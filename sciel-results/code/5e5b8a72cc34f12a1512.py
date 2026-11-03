import time
import numpy as np

N = 50
# grid: 50 equal intervals over [-1/4, 1/4]
X0 = -0.25
X1 = 0.25
step = (X1 - X0) / N
# sample points (midpoints of each interval)
X = X0 + step * (np.arange(N) + 0.5)


def _autoconv_peak(h):
    # circular convolution of h with itself (periodic domain)
    hc = np.fft.fft(h)
    conv = np.fft.ifft(hc * hc).real
    return float(np.max(conv))


def _score(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -1e9
    l2sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    smax = float(np.max(h))
    if l1 <= 0 or smax <= 0:
        return 0.0
    peak = _autoconv_peak(h)
    if peak <= 0:
        return 0.0
    return l2sq / (l1 * smax * peak)


def _build_from_params(amps, freqs, phases):
    # f(x) = sum_k amps[k] * exp(i*(freqs[k]*x + phases[k]))
    # h = |f|^2
    n = len(amps)
    H = np.zeros(N, dtype=complex)
    for k in range(n):
        if amps[k] == 0.0:
            continue
        H += amps[k] * np.exp(1j * (freqs[k] * X + phases[k]))
    h = np.real(H * np.conj(H))
    return h


def _optimize_one(freqs, amps0, phases0, t0, budget):
    # gradient flow on (amps, phases) to maximize score with soft-max peak
    amps = amps0.copy()
    phases = phases0.copy()

    def soft_peak(h, tau):
        hc = np.fft.fft(h)
        conv = np.fft.ifft(hc * hc).real
        # soft-max via log-sum-exp
        m = np.max(conv)
        if m <= 0:
            return 0.0, None
        s = np.sum(np.exp((conv - m) / tau))
        return m + tau * np.log(s), conv

    best = None
    best_score = _score(_build_from_params(amps, freqs, phases))
    tau = 0.02
    lr_a = 0.05
    lr_p = 0.05
    it = 0
    while time.time() < t0 + budget:
        h = _build_from_params(amps, freqs, phases)
        h = np.maximum(h, 0.0)
        l2sq = float(np.sum(h * h))
        l1 = float(np.sum(h))
        smax = float(np.max(h))
        if l1 <= 0 or smax <= 0:
            break
        peak, conv = soft_peak(h, tau)
        if peak <= 0:
            break
        # objective to maximize: J = l2sq / (l1 * smax * peak)
        # We'll do a simple projected gradient ascent on J.
        # Numerical gradient is expensive; use analytic-ish via finite diff on a few steps
        # Instead use a heuristic: reduce peak by adjusting phases (sidelobe cancel),
        # keep amplitudes to shape energy.
        # Perturb phases toward reducing peak (stochastic gradient)
        dA = np.zeros_like(amps)
        dP = np.zeros_like(phases)
        eps = 0.01
        for k in range(len(amps)):
            # phase perturbation: try small + and -
            p_plus = phases.copy(); p_plus[k] += eps
            p_minus = phases.copy(); p_minus[k] -= eps
            h_plus = _build_from_params(amps, freqs, p_plus); h_plus = np.maximum(h_plus,0)
            h_minus = _build_from_params(amps, freqs, p_minus); h_minus = np.maximum(h_minus,0)
            pk_plus = soft_peak(h_plus, tau)[0]
            pk_minus = soft_peak(h_minus, tau)[0]
            # want to reduce peak => dP[k] = - (pk_plus - pk_minus)/(2eps)
            dP[k] = -(pk_plus - pk_minus) / (2 * eps)
            # amplitude: increase amplitude of components that add energy at center
            a_plus = amps.copy(); a_plus[k] += eps
            a_minus = amps.copy(); a_minus[k] = max(0, a_minus[k] - eps) if a_minus[k] > 0 else 0
            h_ap = _build_from_params(a_plus, freqs, phases); h_ap = np.maximum(h_ap,0)
            h_am = _build_from_params(a_minus, freqs, phases); h_am = np.maximum(h_am,0)
            l2p = float(np.sum(h_ap*h_ap)); l1p = float(np.sum(h_ap)); sp = float(np.max(h_ap))
            l2m = float(np.sum(h_am*h_am)); l1m = float(np.sum(h_am)); sm = float(np.max(h_am))
            pp = soft_peak(h_ap, tau)[0]; pm = soft_peak(h_am, tau)[0]
            Jp = (l2p/(l1p*sp*pp)) if (l1p>0 and sp>0 and pp>0) else -1e9
            Jm = (l2m/(l1m*sm*pm)) if (l1m>0 and sm>0 and pm>0) else -1e9
            dA[k] = (Jp - Jm) / (2 * eps)
        # normalize gradients
        nrmP = np.linalg.norm(dP)
        if nrmP > 1e-12:
            dP /= nrmP
        nrmA = np.linalg.norm(dA)
        if nrmA > 1e-12:
            dA /= nrmA
        phases += lr_p * dP
        amps = np.maximum(0, amps + lr_a * dA)
        # renormalize amplitudes to keep scale stable
        an = np.linalg.norm(amps)
        if an > 1e-12:
            amps = amps / an * np.linalg.norm(amps0)
        it += 1
        # anneal
        if it % 50 == 0:
            tau *= 0.95
            lr_a *= 0.97
            lr_p *= 0.97
        h_now = _build_from_params(amps, freqs, phases)
        s_now = _score(h_now)
        if s_now > best_score:
            best_score = s_now
            best = h_now.copy()
    return best, best_score


def construct():
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s
    rng = np.random.default_rng(12345)

    # seed result
    best_h = np.ones(N)
    best_score = _score(best_h)

    # try several frequency-count designs
    for K in [3, 4, 5, 6, 7, 8]:
        if time.time() > DEADLINE:
            break
        # base frequency: choose to create a nice bandlimited structure
        # use frequencies that are integer multiples of a base to keep periodicity
        # base omega such that over domain length 0.5 we get a few periods
        base_omega = 2 * np.pi * (K + 2) / 0.5  # (K+2) periods over length 0.5
        # build a candidate frequency set: symmetric around base for real-valued |f|^2
        # Use K distinct frequencies: a set of harmonics
        freqs = np.array([base_omega * (1.0 + 0.35 * j) for j in range(K)])
        # randomize a bit
        freqs = freqs + rng.normal(0, 0.05 * base_omega, size=K)

        # multiple restarts for this K
        n_restart = 4
        budget_per = min(12.0, (DEADLINE - time.time()) / max(1, n_restart))
        for r in range(n_restart):
            if time.time() > DEADLINE:
                break
            amps0 = np.abs(rng.normal(1.0, 0.5, size=K)) + 0.2
            phases0 = rng.uniform(0, 2 * np.pi, size=K)
            h, s = _optimize_one(freqs, amps0, phases0, time.time(), budget_per)
            if h is not None and s > best_score:
                best_score = s
                best_h = h.copy()
        # also try a "chirp" frequency set (linearly spaced)
        if time.time() > DEADLINE:
            break
        freqs2 = np.linspace(2*np.pi*2/0.5, 2*np.pi*(2*K)/0.5, K)
        for r in range(2):
            if time.time() > DEADLINE:
                break
            amps0 = np.abs(rng.normal(1.0, 0.5, size=K)) + 0.2
            phases0 = rng.uniform(0, 2 * np.pi, size=K)
            h, s = _optimize_one(freqs2, amps0, phases0, time.time(), min(8.0, (DEADLINE - time.time())/2))
            if h is not None and s > best_score:
                best_score = s
                best_h = h.copy()

    # final polish: a few gradient steps on the best structure via direct h-space projection
    # (small local refinement of amplitudes/phases is already done; do a light h-space nudge)
    if time.time() < DEADLINE - 2:
        h = best_h.copy()
        # tiny random perturbations + keep if better (coordinate-ish)
        for _ in range(200):
            if time.time() > DEADLINE - 2:
                break
            idx = rng.integers(0, N)
            trial = h.copy()
            trial[idx] = max(0, h[idx] + rng.normal(0, 0.05))
            s = _score(trial)
            if s > best_score:
                best_score = s
                best_h = trial.copy()

    # ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)


# Self-test guard: do not run heavy search on import
if __name__ == "__main__":
    t0 = time.time()
    h = construct()
    print("score:", _score(np.array(h)), "time:", time.time() - t0)
