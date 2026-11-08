"""
Optimization of the second autocorrelation inequality constant.
Uses a combined strategy: diverse starts, multi-beta L-BFGS-B with
log-sum-exp smoothing, Nelder-Mead polish on the exact objective,
and an epigraph-style reformulation for the peak constraint.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def _convolve_fast(h):
    """FFT-based autocorrelation, returns g of length 2*N-1."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n_fft)
    G = H * np.conj(H)
    g = np.fft.irfft(G, n_fft)[:2 * N - 1]
    return g


def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def compute_value_fast(h):
    """Faster version using FFT for high-frequency evaluation (Nelder-Mead)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = _convolve_fast(h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate using Log-Sum-Exp for norm_inf.
    Returns negative value (for minimization).
    """
    h = np.maximum(0.0, h)
    g = _convolve_fast(h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)

    if norm_1 <= 0:
        return 0.0

    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    if lse <= 0:
        return 0.0

    val = l2 / (norm_1 * lse)
    return -val


def _neg_exact_value(h):
    """Negative of exact value (for minimization) using fast FFT convolution."""
    h = np.maximum(0.0, h)
    return -compute_value_fast(h)


def _epigraph_objective(x, beta=100.0):
    """
    Epigraph formulation: minimize -l2/(norm_1 * t) subject to
    t >= g_i for all i, t >= 0.
    Here x = [h (N vars), t (1 var)].
    We use a penalty for the constraint t >= max(g).
    """
    h = x[:N]
    t = x[N]
    h = np.maximum(0.0, h)
    g = _convolve_fast(h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)

    if norm_1 <= 0 or t <= 0:
        return 1e6

    # Smooth constraint violation: t - max(g) via log-sum-exp
    g_max = np.max(g)
    if g_max > 0:
        lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    else:
        lse = 0.0

    # Penalty if t < lse (constraint violation)
    violation = np.maximum(0.0, lse - t)
    penalty = 1e4 * violation ** 2

    val = l2 / (norm_1 * t)
    return -val + penalty


def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Gaussian
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    # Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Power-law profiles
    for p in [0.5, 1.5, 2.0, 3.0]:
        starts.append(np.power(np.abs(2 * x / (N - 1) - 1), p))
    # Sparse spike patterns
    for frac in [0.05, 0.1, 0.15, 0.2, 0.3]:
        spike = np.zeros(N)
        idx = max(1, int(N * frac / 2))
        spike[N // 2 - idx:N // 2 + idx + 1] = 1.0
        starts.append(spike)
    # Shifted plateaus (asymmetric)
    for center_frac in [0.3, 0.4, 0.6, 0.7]:
        for width_frac in [0.15, 0.25, 0.35]:
            center = center_frac * (N - 1)
            half_w = width_frac * (N - 1) / 2
            plateau = np.zeros(N)
            plateau[np.abs(x - center) <= half_w] = 1.0
            starts.append(plateau)
    # Asymmetric triangular
    for asym in [0.3, 0.7]:
        t = np.zeros(N)
        for i in range(N):
            if i < N * asym:
                t[i] = 1 - i / (N * asym)
            else:
                t[i] = 1 - (i - N * asym) / (N * (1 - asym))
        starts.append(np.maximum(0, t))
    # Random smooth profiles
    for _ in range(20):
        freqs = RNG.randint(1, 12, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.3, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.05, prof))
    # Random sparse with random positions
    for _ in range(10):
        spike = np.zeros(N)
        n_spikes = RNG.randint(1, 4)
        for _ in range(n_spikes):
            pos = RNG.randint(0, N)
            w = RNG.randint(1, 5)
            lo = max(0, pos - w)
            hi = min(N, pos + w + 1)
            spike[lo:hi] = 1.0
        starts.append(spike)
    return starts


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 12.0)

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    starts = _generate_starts()

    # Initial evaluation
    for s in starts:
        track(s)

    bounds = [(0, None)] * N

    # Phase 1: Progressive-beta L-BFGS-B from diverse starts
    # Use lower betas first for broader exploration, then higher for precision
    betas_progressive = [5.0, 15.0, 30.0, 60.0, 120.0]
    for s in starts:
        if time.time() > deadline - 20:
            break
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        for beta in betas_progressive:
            if time.time() > deadline - 20:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _smooth_value(x, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < 0:
                    track(res.x)
                    x0 = np.maximum(0.0, res.x)
            except Exception:
                pass

    # Phase 2: Epigraph formulation for peak control
    for _ in range(12):
        if time.time() > deadline - 15:
            break
        pert = best_h + RNG.normal(0.0, 0.1, N)
        pert = np.maximum(0.0, pert)
        t0 = np.max(_convolve_fast(pert)) + 0.1
        x0 = np.concatenate([pert, [t0]])
        try:
            res = opt.minimize(
                lambda x: _epigraph_objective(x, beta=100.0),
                x0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N + [(0, None)],
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            track(res.x[:N])
        except Exception:
            pass

    # Phase 3: Nelder-Mead polish on exact objective (using fast FFT version)
    for _ in range(30):
        if time.time() > deadline - 8:
            break
        scale = 0.05 * (1.0 + 0.5 * RNG.rand())
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                _neg_exact_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            track(res.x)
        except Exception:
            pass

    # Phase 4: High-beta L-BFGS-B polish
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        pert = best_h + RNG.normal(0.0, 0.03, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, 300.0),
                pert,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            track(res.x)
        except Exception:
            pass

    # Phase 5: Final high-precision Nelder-Mead with very tight tolerances
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        scale = 0.02 * (1.0 + 0.3 * RNG.rand())
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                _neg_exact_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            track(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
