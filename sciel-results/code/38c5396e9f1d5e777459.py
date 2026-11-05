"""Model-and-optimize: exploit scale invariance + nonnegativity to drive the
objective to its positive-sum regime minimum (a two-coordinate spike), then
polish with a small negative tail to shave the value slightly below the bound.
Uses FFT-vectorized gradient for fast L-BFGS-B, chaotic logistic-map
initializations for diverse starts, plus coordinate-descent refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145  # wall-clock seconds; evaluator kills at 150s, keep margin


def _evaluate(h):
    """Return (value, valid) for the exact task objective."""
    if not np.all(np.isfinite(h)):
        return np.inf, False
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf, False
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf, False
    return val, True


def _spike_candidate():
    """Deterministic high-quality candidate: two-spike + small negative tail."""
    a = 0.5
    K = 200
    eps = 0.001
    h = np.zeros(N)
    h[0] = a
    h[1] = 1.0 - a
    h[2:K] = -eps
    return h


def _logistic_candidate(r, n_skip=200, n_samples=400):
    """Generate a candidate from a logistic map. Chaotic for r in (3.57, 4.0)."""
    x = 0.5
    for _ in range(n_skip):
        x = r * x * (1.0 - x)
    seq = np.empty(n_samples)
    for i in range(n_samples):
        x = r * x * (1.0 - x)
        seq[i] = x
    seq -= np.mean(seq)
    s = np.sum(seq)
    if abs(s) < 1e-12:
        return None
    return seq / s


def _spectral_candidate(seed, n_sines=5, freq_range=(1, 20), amp_range=(0.01, 0.5)):
    """Generate a candidate as a sum of sine waves with random frequencies/amplitudes."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = np.zeros(N)
    for _ in range(n_sines):
        freq = rng.uniform(*freq_range)
        amp = rng.uniform(*amp_range)
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.sin(freq * t + phase)
    h -= np.mean(h)
    s = np.sum(h)
    if abs(s) > 1e-12:
        h /= s
    return h


def _smooth_objective(x, beta):
    """Smooth (soft-max) surrogate for 2*N*max(conv)/sum^2, sum>0 regime.
    Uses FFT for the convolution."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return 1e10
    nfft = 2 * N - 1
    X = np.fft.rfft(x, n=nfft)
    conv = np.fft.irfft(X * X, n=nfft)[:2 * N - 1]
    m = float(np.max(conv))
    shifted = conv - m
    shifted = np.clip(shifted, -50.0, 0.0)
    log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    if not np.isfinite(val):
        return 1e10
    return float(val)


def _smooth_grad(x, beta):
    """FFT-vectorized gradient of the soft-max surrogate objective."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    nfft = 2 * N - 1
    X = np.fft.rfft(x, n=nfft)
    conv = np.fft.irfft(X * X, n=nfft)[:2 * N - 1]
    m = float(np.max(conv))
    # Clip to avoid overflow in exp
    shifted = np.clip(conv - m, -50.0, 0.0)
    w = np.exp(beta * shifted)
    w_sum = float(np.sum(w))
    if w_sum < 1e-300:
        w = np.zeros_like(conv)
        w[np.argmax(conv)] = 1.0
        w_sum = 1.0
    w /= w_sum
    # grad_lse[i] = 2 * sum_j w[i+j] * x[j] for j in [0, N-1], i in [0, N-1]
    W = np.fft.rfft(w, n=nfft)
    grad_lse = 2.0 * np.fft.irfft(W * np.conj(X), n=nfft)[:N]
    lse = m + np.log(w_sum) / beta
    f = 2.0 * N * lse / (s * s)
    df = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    return df


def _exact_local_search(h, n_rounds=50, step=0.01, deadline=140.0):
    """Coordinate-descent local search on the true nonsmooth objective.
    Tries small perturbations on individual coordinates and accepts improvements."""
    best_h = h.copy()
    best_val, _ = _evaluate(best_h)
    rng = np.random.RandomState(42)
    
    # Pre-compute for speed: use the fact that only one coordinate changes
    # conv[k] = sum_i h[i]*h[k-i], so changing h[idx] by delta changes
    # conv by delta*(h[idx] + sum of relevant h's)
    # Actually, let's just be smart about when to recompute fully
    
    for _ in range(n_rounds):
        if time.time() > deadline:
            break
        improved = False
        for _ in range(30):
            if time.time() > deadline:
                break
            idx = rng.randint(N)
            for delta in (step, -step, step*0.5, -step*0.5):
                # In-place modification to avoid copy
                old_val = best_h[idx]
                best_h[idx] = old_val + delta
                val, valid = _evaluate(best_h)
                if valid and val < best_val:
                    best_val = val
                    improved = True
                    break
                else:
                    best_h[idx] = old_val
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
    return best_h, best_val


def construct():
    start = time.time()

    # Candidate pool: deterministic spike variants + structured starts.
    candidates = [_spike_candidate()]

    # alternate-spike (h[0]=1, rest 0): value exactly 2*N = 800, valid fallback
    alt = np.zeros(N); alt[0] = 1.0
    candidates.append(alt)

    # single spike with tail of varying strength
    for K in (50, 100, 200, 300, 399):
        for eps in (0.0005, 0.001, 0.002, 0.005):
            if K - 2 <= 0:
                continue
            h = np.zeros(N)
            h[0] = 0.5
            h[1] = 0.5
            h[2:K] = -eps
            candidates.append(h)

    # two-spike with unequal split
    for a in (0.3, 0.4, 0.5, 0.6, 0.7):
        h = np.zeros(N)
        h[0] = a
        h[1] = 1.0 - a
        h[2:120] = -0.001
        candidates.append(h)

    # structured: positive spike on a short block, small negative elsewhere
    rng = np.random.RandomState(0)
    for _ in range(12):
        h = np.zeros(N)
        h[0] = 0.5
        h[1] = 0.5
        neg_mask = np.zeros(N, dtype=bool)
        neg_mask[2:] = True
        idx = np.where(neg_mask)[0]
        pick = rng.choice(idx, size=rng.randint(20, 300), replace=False)
        h[pick] = -rng.uniform(0.0003, 0.004)
        candidates.append(h)

    # spectral (sine-wave) candidates
    for seed in range(20):
        for n_sines in (3, 5, 8):
            candidates.append(_spectral_candidate(seed, n_sines=n_sines))

    # Chaotic logistic-map candidates
    for r in (3.6, 3.7, 3.8, 3.85, 3.9, 3.95, 3.99, 3.999):
        for n_skip in (100, 500, 2000):
            c = _logistic_candidate(r, n_skip=n_skip)
            if c is not None:
                candidates.append(c)

    # Chebyshev polynomial based candidates
    for degree in (5, 8, 12):
        x = np.linspace(-1, 1, N)
        h = np.polynomial.chebyshev.chebval(x, np.array([1.0] + [0.0] * degree))
        h -= np.mean(h)
        s = np.sum(h)
        if abs(s) > 1e-12:
            h /= s
        candidates.append(h)

    # Evaluate all candidates, keep the best.
    best_h = None
    best_value = np.inf
    for h in candidates:
        if h is None:
            continue
        val, valid = _evaluate(h)
        if valid and val < best_value:
            best_value = val
            best_h = h.copy()

    # Guard: ensure we always have a valid finite candidate.
    if best_h is None or not np.isfinite(best_value):
        best_h = _spike_candidate()
        best_value, _ = _evaluate(best_h)

    # Multi-stage L-BFGS-B polish with increasing beta (sharper approximation to max).
    if time.time() - start < DEADLINE - 10:
        for beta in (40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0):
            if time.time() - start > DEADLINE - 5:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_h.copy(),
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'maxfun': 5000, 'gtol': 1e-13},
                )
                val, valid = _evaluate(res.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Exact local search on the true nonsmooth objective.
    if time.time() - start < DEADLINE - 4:
        best_h, best_value = _exact_local_search(best_h, n_rounds=100, step=0.01, deadline=DEADLINE - 4)

    # Final high-precision polish with very tight tolerance.
    if time.time() - start < DEADLINE - 2:
        for beta in (4096.0, 8192.0):
            if time.time() - start > DEADLINE - 1:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_h.copy(),
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'maxfun': 8000, 'gtol': 1e-15},
                )
                val, valid = _evaluate(res.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Final exact local search pass.
    if time.time() - start < DEADLINE - 0.5:
        best_h, best_value = _exact_local_search(best_h, n_rounds=50, step=0.005, deadline=DEADLINE - 0.5)

    # Final validity guard: never return an invalid vector.
    val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = _spike_candidate()
        val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return best_h
