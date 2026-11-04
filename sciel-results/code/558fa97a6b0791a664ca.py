"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import time

def _compute_ratio(h):
    """Compute the actual ratio for evaluation."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0
    corr = np.correlate(h, h, mode='full')
    peak = np.max(corr)
    if peak < 1e-30:
        return 0.0
    return l2_sq / (l1 * peak)

def _smooth_objective(h, beta=80.0):
    """
    Smooth negative ratio using log-sum-exp soft-max for the autoconvolution peak.
    Minimizing this maximizes the ratio.
    """
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 1e10
    corr = np.correlate(h, h, mode='full')
    corr = np.clip(corr, 1e-30, None)
    log_corr = np.log(corr)
    max_log = np.max(log_corr)
    # soft-max with temperature 1/beta: as beta -> inf, approaches true max
    soft_max = max_log + np.log(np.sum(np.exp(log_corr - max_log))) / beta
    peak = np.exp(soft_max)
    if peak < 1e-30:
        return 1e10
    ratio = l2_sq / (l1 * peak)
    return -ratio

def _local_optimize(h0, bounds, maxiter=400, maxfun=10000, ftol=1e-12, gtol=1e-9, beta=80.0):
    """Run L-BFGS-B local optimization with the smooth objective."""
    def obj(x):
        return _smooth_objective(x, beta=beta)
    try:
        res = minimize(
            obj, h0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'maxfun': maxfun,
                     'ftol': ftol, 'gtol': gtol}
        )
        h_opt = np.clip(res.x, 0, None)
        return h_opt
    except Exception:
        return h0

def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin before 150s kill

    best_val = -np.inf
    best_h = np.ones(N)

    def _update(h):
        nonlocal best_val, best_h
        val = _compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        return val

    # Evaluate seed
    _update(best_h)

    rng = np.random.default_rng(2024)
    bounds = [(0.0, None)] * N
    x = np.linspace(-1, 1, N)

    # Build diverse initializations
    init_list = []

    # 1. Constant
    init_list.append(np.ones(N))

    # 2. Gaussians with various widths
    for sigma in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]:
        init_list.append(np.exp(-x**2 / (2*sigma**2)))

    # 3. Bimodal peaks
    for gap in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for width in [0.1, 0.15, 0.2, 0.3, 0.4]:
            h_init = np.exp(-((x-gap)**2) / (2*width**2)) + np.exp(-((x+gap)**2) / (2*width**2))
            init_list.append(h_init)

    # 4. Triangular
    init_list.append(np.clip(1 - np.abs(x), 0, None))

    # 5. Cosine window
    init_list.append(0.5 * (1 + np.cos(np.pi * x)))

    # 6. Lorentzian / Cauchy
    for gamma in [0.1, 0.2, 0.3, 0.5, 0.8]:
        init_list.append(1.0 / (1.0 + (x/gamma)**2))

    # 7. Flat-top / box
    for w in [0.3, 0.5, 0.7]:
        h_box = np.where(np.abs(x) <= w, 1.0, 0.0)
        init_list.append(h_box)

    # 8. Random smooth profiles
    for _ in range(12):
        h_rand = np.abs(rng.normal(1.0, 0.9, N))
        h_rand = gaussian_filter1d(h_rand, sigma=rng.uniform(1.0, 3.0))
        h_rand = np.clip(h_rand, 0, None)
        if np.sum(h_rand) > 1e-10:
            init_list.append(h_rand)

    # 9. Sparse-ish profiles
    for _ in range(8):
        h_sparse = np.zeros(N)
        k = rng.integers(3, 18)
        idx = rng.choice(N, size=k, replace=False)
        h_sparse[idx] = rng.uniform(0.3, 1.5, size=k)
        h_sparse = gaussian_filter1d(h_sparse, sigma=rng.uniform(0.5, 1.5))
        h_sparse = np.clip(h_sparse, 0, None)
        if np.sum(h_sparse) > 1e-10:
            init_list.append(h_sparse)

    # 10. Sine-based shapes
    for freq in [1, 2, 3]:
        for phase in [0, np.pi/2]:
            h_sin = np.abs(np.sin(freq * np.pi * x + phase))
            init_list.append(h_sin)

    # 11. Multi-peak (3 peaks)
    for _ in range(6):
        centers = rng.uniform(-0.8, 0.8, size=3)
        widths = rng.uniform(0.1, 0.4, size=3)
        h_multi = np.zeros(N)
        for c, w in zip(centers, widths):
            h_multi += np.exp(-((x-c)**2) / (2*w**2))
        init_list.append(h_multi)

    # Normalize each initialization to sum = N for consistent scale
    for h0 in init_list:
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = np.clip(h0, 0, None)
        s = np.sum(h0)
        if s < 1e-10:
            continue
        init_list[-1] = h0 * N / s

    # Remove duplicates (approximate)
    unique_inits = []
    for h0 in init_list:
        if len(unique_inits) == 0:
            unique_inits.append(h0)
            continue
        is_dup = False
        for hu in unique_inits:
            if np.linalg.norm(h0 - hu) < 1e-6 * N:
                is_dup = True
                break
        if not is_dup:
            unique_inits.append(h0)

    # Optimize from each initialization
    for i, h0 in enumerate(unique_inits):
        if time.time() > deadline - 2:
            break
        h_opt = _local_optimize(h0, bounds, maxiter=300, maxfun=8000,
                                ftol=1e-12, gtol=1e-8, beta=80.0)
        _update(h_opt)

    # Final high-precision polish on the best
    if time.time() < deadline - 5:
        h_polish = _local_optimize(best_h, bounds, maxiter=800, maxfun=25000,
                                   ftol=1e-14, gtol=1e-11, beta=150.0)
        _update(h_polish)

    # One more very tight polish
    if time.time() < deadline - 2:
        h_polish2 = _local_optimize(best_h, bounds, maxiter=1000, maxfun=40000,
                                    ftol=1e-15, gtol=1e-12, beta=200.0)
        _update(h_polish2)

    return list(best_h)
# EVOLVE-BLOCK-END
