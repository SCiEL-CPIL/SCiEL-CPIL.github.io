"""First autocorrelation inequality: minimize max autoconv / ||f||_2^2 for step function on [-1/4,1/4]."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 600
DEADLINE = 150.0  # seconds
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _time_ok(margin=5.0):
    return _elapsed() < DEADLINE - margin

def _score(f):
    """Score = max(autoconv) / (sum(f)^2)  (normalized max autoconvolution)."""
    f = np.asarray(f, dtype=np.float64)
    if f.size != N or np.any(f < -1e-12):
        return float('inf')
    f = np.maximum(f, 0.0)
    s = f.sum()
    if s <= 0:
        return float('inf')
    # Circular autoconvolution via FFT
    F = np.fft.rfft(f, n=2 * N)
    ac = np.fft.irfft(F * np.conj(F), n=2 * N)[:N]
    return float(np.max(ac) / (s * s))

def _fft_grad(f):
    """Gradient of score w.r.t. f (for coordinate descent)."""
    f = np.asarray(f, dtype=np.float64)
    s = f.sum()
    if s <= 0:
        return np.zeros_like(f)
    F = np.fft.rfft(f, n=2 * N)
    ac = np.fft.irfft(F * np.conj(F), n=2 * N)[:N]
    # d(max_ac)/df = 2 * f (since ac = f * f circular, d(ac_k)/df_j = f_{k-j})
    # d(score)/df_j = (2 * f * (1/s^2) - 2*ac_max*f/s^3) ... simplified
    # Use: score = ac_max / s^2
    # d score / d f_j = (2 f * s^2 - 2 ac_max * s) / s^4 * (indicator j in argmax) approx
    # Simpler: use subgradient: d score = (2 f_j / s^2) * I[j in argmax] - 2*score*f_j/s^2
    idx = np.argmax(ac)
    grad = np.zeros_like(f)
    grad[idx] = 2 * f[idx] / (s * s) - 2 * ac[idx] * f / (s ** 3)
    return grad

def _coord_descent(f, max_iters=20000, tol=1e-10):
    """Coordinate descent to minimize score."""
    f = np.array(f, dtype=np.float64)
    cur = _score(f)
    if not np.isfinite(cur):
        return f, cur
    n = f.size
    for it in range(max_iters):
        if not _time_ok(10.0):
            break
        best_j = -1
        best_val = cur
        # Random order for diversity
        for j in np.random.permutation(n):
            if not _time_ok(10.0):
                break
            old = f[j]
            # Try a few candidate values
            for trial in [0.0, old * 0.5, old * 0.8, old * 1.2, old * 1.5,
                          old + 0.05, old + 0.2, old - 0.05, old - 0.2,
                          cur * 0.1, 0.5, 1.0]:
                trial = max(0.0, trial)
                f[j] = trial
                v = _score(f)
                if v < best_val - 1e-14:
                    best_val = v
                    best_j = j
                    best_trial = trial
            f[j] = old
        if best_j >= 0:
            f[best_j] = best_trial
            cur = best_val
        else:
            break
    return f, cur

def _make_pulse(center, width, height):
    """Disjoint pulse: nonzero on a contiguous interval."""
    f = np.zeros(N)
    c = int(center * N)
    w = int(width * N)
    lo = max(0, c - w // 2)
    hi = min(N, lo + w)
    f[lo:hi] = height
    return f

def _make_bell(center, sigma, height):
    """Smooth Gaussian bell."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return np.maximum(f, 0.0)

def _make_random_smooth(scale=1.0):
    """Random smooth function via low-frequency Fourier coefficients."""
    f = np.zeros(N)
    for k in range(1, 20):
        amp = np.random.uniform(0, scale) / k
        phase = np.random.uniform(0, 2 * np.pi)
        x = np.linspace(-0.25, 0.25, N, endpoint=False)
        f += amp * np.cos(2 * np.pi * k * (x + 0.25) + phase)
    f = np.maximum(f, 0.0)
    return f

def _make_box(a, b, height):
    """Box on [a,b] in [-1/4,1/4]."""
    f = np.zeros(N)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    mask = (x >= a) & (x <= b)
    f[mask] = height
    return f

def construct():
    global START_TIME
    START_TIME = time.time()
    rng = np.random.RandomState(42)
    np.random.seed(42)

    best_f = np.ones(N)
    best_score = _score(best_f)

    # Generate diverse initial candidates
    candidates = []

    # 1. Uniform
    candidates.append(np.ones(N))

    # 2. Single central pulse
    candidates.append(_make_pulse(0.0, 0.1, 1.0))
    candidates.append(_make_pulse(0.0, 0.2, 1.0))
    candidates.append(_make_pulse(0.0, 0.05, 1.0))

    # 3. Two disjoint pulses
    for gap in [0.05, 0.1, 0.15]:
        c1 = _make_pulse(-gap / 2, 0.08, 1.0)
        c2 = _make_pulse(gap / 2, 0.08, 1.0)
        candidates.append(np.maximum(c1, c2))

    # 4. Smooth bells
    for sigma in [0.02, 0.05, 0.1]:
        candidates.append(_make_bell(0.0, sigma, 1.0))
    # Two bells
    for sigma in [0.03, 0.06]:
        for off in [0.05, 0.1]:
            b1 = _make_bell(-off, sigma, 1.0)
            b2 = _make_bell(off, sigma, 1.0)
            candidates.append(np.maximum(b1, b2))

    # 5. Random smooth
    for _ in range(10):
        candidates.append(_make_random_smooth(scale=1.0))

    # 6. Boxes of various widths
    for w in [0.05, 0.1, 0.15, 0.2]:
        candidates.append(_make_box(-w / 2, w / 2, 1.0))
    # Two boxes
    for w in [0.05, 0.08]:
        for off in [0.08, 0.12]:
            b1 = _make_box(-off - w / 2, -off + w / 2, 1.0)
            b2 = _make_box(off - w / 2, off + w / 2, 1.0)
            candidates.append(np.maximum(b1, b2))

    # Evaluate all candidates
    for c in candidates:
        if not _time_ok(30.0):
            break
        s = _score(c)
        if s < best_score:
            best_score = s
            best_f = np.array(c)

    # Coordinate descent refinement on best candidates
    top_candidates = []
    for c in candidates:
        top_candidates.append((_score(c), c))
    top_candidates.sort(key=lambda x: x[0])
    top_candidates = top_candidates[:8]

    for s, c in top_candidates:
        if not _time_ok(15.0):
            break
        f_refined, s_refined = _coord_descent(c, max_iters=3000)
        if s_refined < best_score:
            best_score = s_refined
            best_f = f_refined

    # Additional random restarts if time permits
    if _time_ok(20.0):
        for _ in range(20):
            if not _time_ok(15.0):
                break
            # Random mix
            base = _make_random_smooth(scale=rng.uniform(0.5, 2.0))
            if rng.random() < 0.5:
                base = np.maximum(base, _make_pulse(rng.uniform(-0.1, 0.1),
                                                     rng.uniform(0.03, 0.15),
                                                     rng.uniform(0.5, 1.5)))
            f_refined, s_refined = _coord_descent(base, max_iters=2000)
            if s_refined < best_score:
                best_score = s_refined
                best_f = f_refined

    # Final: ensure nonnegativity and return as list
    best_f = np.maximum(best_f, 0.0)
    return [float(v) for v in best_f]
