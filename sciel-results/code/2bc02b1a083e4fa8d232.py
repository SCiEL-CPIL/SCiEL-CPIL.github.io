"""
Combined approach:
  - Diverse structured starts (plateaus, Gaussians, power-law, sparse spikes,
    semicircle, cosine, random harmonics) for global coverage.
  - Progressive-beta L-BFGS-B on a log-sum-exp smoothed surrogate to escape
    local minima of the non-smooth peak.
  - Epigraph + penalty refinement to sharpen the peak structure.
  - Nelder-Mead polish on the exact objective for final refinement.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1                # len(g) = 99
W = 1.0 / (M + 1)            # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


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


def _neg_exact_value(h):
    h = np.maximum(0.0, h)
    return -compute_value(h)


def _plateau(j, k):
    h = np.zeros(N)
    end = min(N, j + k)
    if end > j:
        h[j:end] = 1.0
    return h


def _smooth_value(h, beta=100.0):
    """Smooth surrogate using log-sum-exp for max(g)."""
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
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
    return -(l2 / (norm_1 * lse))


def _generate_starts():
    """Diverse set of initial profiles for global coverage."""
    x = np.linspace(0, N - 1, N)
    starts = []

    # Plateaus of various widths (the seed and its neighborhood)
    for k in [1, 3, 5, 7, 10, 13, 15, 20, 25, 30, 40, 50]:
        h = _plateau(0, k)
        if h.max() > 0:
            starts.append(h)

    # Shifted plateaus
    for k in [10, 15, 20, 25]:
        for j in [0, N // 4, N // 2]:
            if j + k <= N:
                starts.append(_plateau(j, k))

    # Uniform
    starts.append(np.ones(N))

    # Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))

    # Sharp Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 10)) ** 2))

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

    # Power-law profiles (emphasize edges or center)
    for p in [0.5, 1.0, 2.0, 4.0]:
        t = np.abs(x - (N - 1) / 2) / ((N - 1) / 2)
        starts.append(np.maximum(0.01, (1 - t) ** p))

    # Sparse spikes
    for width in [1, 2, 3]:
        for center in [N // 4, N // 2, 3 * N // 4]:
            h = np.zeros(N)
            lo = max(0, center - width // 2)
            hi = min(N, lo + width)
            h[lo:hi] = 1.0
            starts.append(h)

    # Random smooth profiles (low-frequency harmonics)
    for _ in range(15):
        freqs = RNG.randint(1, 6, size=5)
        phases = RNG.uniform(0, 2 * np.pi, size=5)
        amps = RNG.uniform(0.5, 1.0, size=5)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.05, prof))

    # Random sparse profiles
    for _ in range(10):
        h = np.zeros(N)
        n_nonzero = RNG.randint(3, 15)
        idx = RNG.choice(N, size=n_nonzero, replace=False)
        h[idx] = RNG.uniform(0.3, 1.0, size=n_nonzero)
        starts.append(h)

    return starts


def _epigraph_objective(params, penalty_weight=50.0):
    """
    Epigraph objective: minimize -l2/(norm_1 * t) + penalty * sum(max(0, g-t)^2)
    params: [h_0..h_49, t]
    """
    h = np.maximum(0.0, params[:N])
    t = params[N]
    if t <= 0:
        t = 1e-6

    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0:
        return 1e10

    violation = np.maximum(0.0, g - t)
    penalty = penalty_weight * np.sum(violation ** 2)

    return -(l2 / (norm_1 * t)) + penalty


def construct():
    """Construct 50 nonnegative step heights to maximize the value."""
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0)

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.asarray(h, dtype=float).ravel()
        if h.size != N:
            h = h[:N]
            if h.size < N:
                h = np.pad(h, (0, N - h.size), 'constant')
        h = np.maximum(0.0, h)
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # --- Phase 1: Evaluate diverse starts ----------------------------------
    starts = _generate_starts()
    for s in starts:
        track(s)

    # --- Phase 2: Progressive-beta L-BFGS-B on smooth surrogate ------------
    bounds = [(0, None)] * N
    betas = [20.0, 50.0, 100.0, 200.0]

    # For each start, run L-BFGS-B with increasing beta (progressive sharpening)
    for s in starts:
        if time.time() > deadline:
            break
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        for beta in betas:
            if time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _smooth_value(x, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                track(res.x)
            except Exception:
                pass

    # --- Phase 3: Epigraph refinement from best ----------------------------
    if time.time() < deadline:
        for attempt in range(12):
            if time.time() > deadline:
                break
            pert_h = np.maximum(0.0, best_h + RNG.normal(0.0, 0.03, N))
            pert_g = np.convolve(pert_h, pert_h)
            pert_t = np.max(pert_g) + 0.1
            x0 = np.concatenate([pert_h, [pert_t]])
            ep_bounds = [(0, None)] * N + [(1e-4, None)]
            try:
                res = opt.minimize(
                    lambda x: _epigraph_objective(x, penalty_weight=50.0),
                    x0,
                    method='L-BFGS-B',
                    bounds=ep_bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                track(res.x[:N])
            except Exception:
                pass

    # --- Phase 4: L-BFGS-B polish on smooth surrogate from best ------------
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            pert = np.maximum(0.0, best_h + RNG.normal(0.0, 0.04, N))
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, 300.0),
                    pert,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                track(res.x)
            except Exception:
                pass

    # --- Phase 5: Nelder-Mead polish on exact objective ---------------------
    if time.time() < deadline:
        for _ in range(30):
            if time.time() > deadline:
                break
            pert = best_h + RNG.normal(0.0, 0.05, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                )
                track(res.x)
            except Exception:
                pass

    # --- Phase 6: High-precision final polish --------------------------------
    if time.time() < deadline - 5.0:
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            pert = best_h + RNG.normal(0.0, 0.01, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
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
