"""Erdos minimum overlap: rational-generating-function parameterization + FFT circular correlation."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin vs 150s wall-clock

    # ---------- Core metric: circular cross-correlation of h with (1-h) ----------
    def get_overlaps(h):
        """Overlap at shift k: sum_i h[i] * (1 - h[(i+k) mod N]) = N/2 - autocorr(h, k)."""
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(H * np.conj(H)).real
        overlaps = TARGET_SUM - autocorr
        return overlaps[1:]  # exclude the trivial k=0 shift

    def max_overlap(h):
        return float(np.max(get_overlaps(h)))

    # ---------- Feasibility projection: 0<=h_i<=1, sum(h)=N/2 ----------
    def project(h):
        h = np.clip(h, 0.0, 1.0)
        for _ in range(60):
            diff = TARGET_SUM - np.sum(h)
            if abs(diff) < 1e-14:
                break
            h = np.clip(h + diff / N, 0.0, 1.0)
        residual = TARGET_SUM - np.sum(h)
        if abs(residual) > 1e-14:
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        return h

    # ---------- STRUCTURAL: rational generating function -> smooth autocorrelation ----------
    # h is generated from few coefficients (a low-order rational / autoregressive
    # spectrum) so its autocorrelation decays smoothly, suppressing individual shift peaks.
    def rational_h(params):
        """params = [mean, (a1, b1), (a2, b2), (a3, b3)] -> projected h in [0,1], sum=N/2."""
        mean = params[0]
        # start from a smooth base: mean + first harmonic
        x = np.arange(N)
        h = mean + params[1] * np.cos(2 * np.pi * x / N) + params[2] * np.sin(2 * np.pi * x / N)
        # rational (AR) structure: add decaying multi-harmonic terms with
        # exponentially-decaying amplitudes (b_i in (0,1) = decay, a_i = phase/freq weight)
        for k in range(3):
            freq = k + 2
            amp = params[3 + 2 * k] * (params[4 + 2 * k] ** k)
            phase = 2 * np.pi * freq * x / N
            h += amp * np.cos(phase + params[4 + 2 * k] * 0.0)  # keep real, structured
            h += amp * 0.3 * np.sin(phase)
        return project(h)

    def lse_objective(h, beta):
        ov = get_overlaps(h)
        m = float(np.max(ov))
        if m > 500:
            m = 500.0
        return m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta

    # ---------- Baseline & diverse candidate pool ----------
    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)  # ~25.0

    rng = np.random.default_rng(42)
    candidates = [np.full(N, 0.5)]

    # Multi-harmonic structured profiles (the measured improvement direction)
    for freq in range(1, 15):
        for phase in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
            h = 0.5 + 0.45 * np.sin(x + phase)
            candidates.append(project(h))
    for freq in range(1, 8):
        x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
        h = 0.5 + 0.45 * (2 / np.pi) * np.arcsin(np.sin(x))
        candidates.append(project(h))
    # random smooth low-frequency signals
    for _ in range(25):
        coeffs = rng.standard_normal(6)
        sig = np.zeros(N)
        for f in range(1, 6):
            sig += coeffs[f - 1] * np.sin(2 * np.pi * f * np.arange(N) / N)
        h = 0.5 + 0.35 * sig / (np.max(np.abs(sig)) + 1e-10)
        candidates.append(project(h))
    # random perturbations
    for _ in range(20):
        candidates.append(project(0.5 + 0.3 * rng.standard_normal(N)))
    # rational-function generated candidates
    for _ in range(12):
        p = [0.5, rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)]
        for _k in range(3):
            p.append(rng.uniform(-0.3, 0.3))
            p.append(rng.uniform(0.3, 0.95))
        candidates.append(rational_h(p))

    scored = []
    for h in candidates:
        if time.time() > deadline - 25:
            break
        scored.append((max_overlap(h), h))
    scored.sort(key=lambda t: t[0])
    top = [h for _, h in scored[:min(10, len(scored))]]

    # ---------- Optimization: free-variable L-BFGS-B with projection ----------
    def optimize_free(h0, beta, max_iter, ftol=1e-11, gtol=1e-10):
        def obj(h):
            return lse_objective(project(h), beta)
        try:
            res = minimize(obj, h0, method='L-BFGS-B',
                           bounds=[(0, 1)] * N,
                           options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
            return project(res.x)
        except Exception:
            return project(h0)

    for h0 in top:
        if time.time() > deadline - 20:
            break
        cur = h0.copy()
        for beta in [10.0, 40.0, 100.0, 200.0]:
            if time.time() > deadline - 12:
                break
            cur = optimize_free(cur, beta, 300)
        s = max_overlap(cur)
        if s < best_score:
            best_score, best_h = s, cur.copy()

    # ---------- Optimize the RATIONAL COEFFICIENTS (fewer, smoother DOF) ----------
    def rational_obj(params):
        return lse_objective(rational_h(params), 80.0)

    for p0 in [
        [0.5, 0.4, 0.0, 0.2, 0.7, 0.15, 0.6, 0.1, 0.5],
        [0.5, 0.0, 0.4, -0.2, 0.75, 0.1, 0.65, -0.1, 0.55],
        [0.5, 0.3, 0.3, 0.1, 0.8, 0.1, 0.7, 0.05, 0.6],
    ]:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(rational_obj, p0, method='L-BFGS-B',
                           bounds=[(0.1, 0.9)] + [(0, 1)] * 8,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-11})
            hr = rational_h(res.x)
            s = max_overlap(hr)
            if s < best_score:
                best_score, best_h = s, hr.copy()
        except Exception:
            pass

    # ---------- High-precision polish on the best solution ----------
    if time.time() < deadline - 10:
        hp = optimize_free(best_h, 300.0, 1500, ftol=1e-13, gtol=1e-12)
        s = max_overlap(hp)
        if s < best_score:
            best_score, best_h = s, hp.copy()

    # ---------- Coordinate-descent local refinement ----------
    if time.time() < deadline - 5:
        h = best_h.copy()
        for _ in range(150):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                old = h[i]
                for delta in (-0.04, -0.02, -0.01, 0.01, 0.02, 0.04):
                    nv = old + delta
                    if 0 <= nv <= 1:
                        h[i] = nv
                        residual = TARGET_SUM - np.sum(h)
                        j = (i + 1) % N
                        while j == i:
                            j = (j + 1) % N
                        h[j] += residual
                        if 0 <= h[j] <= 1:
                            s = max_overlap(h)
                            if s < best_score:
                                best_score, best_h, improved = s, h.copy(), True
                        h[j] -= residual
                h[i] = old
            if not improved:
                break

    best_h = project(best_h)
    return best_h

# EVOLVE-BLOCK-END
