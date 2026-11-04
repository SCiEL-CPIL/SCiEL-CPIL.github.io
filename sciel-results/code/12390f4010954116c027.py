import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = (N - 1) // 2  # 24

_DEADLINE = 150.0
_SAFETY = 10.0
_start = 0.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same N-grid."""
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    return ac


def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) from a precomputed nonneg h."""
    h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return 0.0
    h_norm = h / l1
    l2sq_norm = float(np.dot(h_norm, h_norm))
    if l2sq_norm <= 1e-300:
        return 0.0
    ac_norm = _autoconv(h_norm)
    acmax = float(np.max(ac_norm))
    if acmax <= 1e-300:
        return 0.0
    return l2sq_norm / acmax


def _logR_and_grad(h, beta=30.0):
    """Compute log R(h) and its gradient w.r.t. h, using smooth max."""
    h = np.maximum(h, 1e-300)
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return -1e30, np.zeros(N)
    l2sq = float(np.dot(h, h))
    if l2sq <= 1e-300:
        return -1e30, np.zeros(N)
    ac = _autoconv(h)
    ac = np.maximum(ac, 1e-300)
    ac_max = float(np.max(ac))
    if ac_max <= 1e-300:
        return -1e30, np.zeros(N)

    # Smooth max via log-sum-exp
    inv_beta = 1.0 / beta
    shifted = ac - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = float(np.sum(exp_shifted))
    if sum_exp <= 1e-300:
        return -1e30, np.zeros(N)
    log_acmax = ac_max + inv_beta * math.log(sum_exp)
    softmax_w = exp_shifted / sum_exp

    logR = math.log(l2sq) - math.log(l1) - log_acmax

    # Gradient: d logR / dh_k = 2*h_k/l2sq - 1/l1 - d log_acmax / dh_k
    # d log_acmax / dh_k = (1/beta) * sum_j softmax_w[j] * dac_j/dh_k
    # dac_j/dh_k = 2*DH*h_{j-k} if 0<=j-k<N else 0
    dlog_acmax_dh = np.zeros(N)
    for k in range(N):
        if N - k > 0:
            dlog_acmax_dh[k] = (2.0 * DH / beta) * float(np.dot(softmax_w[k:], h[:N - k]))

    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh

    # Zero gradient where h is at boundary
    dlogR_dh[h <= 1e-15] = 0.0

    if not np.all(np.isfinite(dlogR_dh)):
        dlogR_dh = np.zeros(N)

    return logR, dlogR_dh


def _neg_logR(h):
    v, _ = _logR_and_grad(h)
    return -v


def _neg_logR_jac(h):
    v, g = _logR_and_grad(h)
    return -v, -g


def _local_optimize(h0, budget, beta=30.0):
    """Local L-BFGS-B optimization from h0."""
    t0 = time.time()
    bounds = [(0.0, None)] * N

    def obj(x):
        return _neg_logR(x)

    def jac(x):
        return _neg_logR_jac(x)

    try:
        res = minimize(
            obj, h0.copy(),
            method='L-BFGS-B',
            jac=jac,
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget else x),
        )
        if res.x is not None:
            h_opt = np.maximum(res.x, 0.0)
            R_opt = _ratio_from_h(h_opt)
            return h_opt, R_opt
    except Exception:
        pass
    return h0.copy(), _ratio_from_h(h0)


def _two_lobe_param_to_h(params):
    """Convert parameter vector [delta, sigma, amp_left, amp_right] to height vector.
    
    Two Gaussian lobes centered at center-delta and center+delta.
    """
    delta, sigma, a1, a2 = params
    delta = max(0.0, min(float(delta), 24.0))
    sigma = max(0.3, min(float(sigma), 15.0))
    a1 = max(0.0, float(a1))
    a2 = max(0.0, float(a2))
    
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0  # 24.5
    
    h = a1 * np.exp(-0.5 * ((x - (center - delta)) / sigma) ** 2)
    h += a2 * np.exp(-0.5 * ((x - (center + delta)) / sigma) ** 2)
    
    return h


def _ratio_from_two_lobe(params):
    """Compute ratio for two-lobe parameterization."""
    h = _two_lobe_param_to_h(params)
    return _ratio_from_h(h)


def _neg_logR_two_lobe(params):
    """Negative log ratio for two-lobe parameterization (for scipy)."""
    h = _two_lobe_param_to_h(params)
    h = np.maximum(h, 1e-300)
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return 1e30
    l2sq = float(np.dot(h, h))
    if l2sq <= 1e-300:
        return 1e30
    ac = _autoconv(h)
    ac = np.maximum(ac, 1e-300)
    ac_max = float(np.max(ac))
    if ac_max <= 1e-300:
        return 1e30
    logR = math.log(l2sq) - math.log(l1) - math.log(ac_max)
    return -logR


def _optimize_two_lobe(budget):
    """Optimize two-lobe parameters using scipy."""
    t0 = time.time()
    
    # Parameter bounds: [delta, sigma, a1, a2]
    bounds = [(1.0, 24.0), (0.5, 15.0), (0.1, 10.0), (0.1, 10.0)]
    
    best_h = None
    best_ratio = 0.0
    
    # Grid of starting points
    rng = np.random.default_rng(123)
    starts = []
    
    # Symmetric starts
    for delta in [5, 8, 10, 12, 15, 18, 20, 22]:
        for sigma in [2, 3, 4, 5, 7, 10]:
            starts.append([delta, sigma, 1.0, 1.0])
    
    # Asymmetric starts
    for delta in [5, 8, 12, 15, 18]:
        for sigma in [2, 4, 6]:
            for a1 in [0.5, 1.0, 2.0]:
                for a2 in [0.5, 1.0, 2.0]:
                    starts.append([delta, sigma, a1, a2])
    
    # Random starts
    for _ in range(30):
        starts.append([
            rng.uniform(3, 22),
            rng.uniform(1, 12),
            rng.uniform(0.5, 3),
            rng.uniform(0.5, 3)
        ])
    
    for p0 in starts:
        if _time_left() < 1.0:
            break
        
        n_remaining = len(starts) - starts.index(p0)
        budget_i = min(1.5, _time_left() / max(n_remaining, 1) - 0.2)
        if budget_i < 0.3:
            budget_i = 0.3
        
        try:
            res = minimize(
                _neg_logR_two_lobe, np.array(p0, dtype=float),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-12,
                         'gtol': 1e-8, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - t0) > budget_i else x),
            )
            if res.x is not None:
                h = _two_lobe_param_to_h(res.x)
                r = _ratio_from_h(h)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h.copy()
        except Exception:
            pass
    
    if best_h is not None:
        return best_h, best_ratio
    return None, 0.0


def _generate_starts():
    """Generate diverse starting points."""
    rng = np.random.default_rng(42)
    starts = []

    # Uniform
    starts.append(np.ones(N))

    # Power-law distributions (concentrated on one side)
    for alpha in [1.0, 2.0, 3.0, 4.0, 5.0]:
        x = np.linspace(0.1, 1.0, N)
        starts.append(x ** alpha)
        starts.append((1.0 - x) ** alpha)
        starts.append(x ** (-alpha))  # concentrated at 0
    
    # Gaussian bumps at various positions and widths
    for center in range(0, N, 5):
        for width in [2, 4, 6, 8, 12]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width) ** 2)
            starts.append(h)

    # Two-lobe structures (the key structural insight)
    for delta in [5, 8, 10, 12, 15, 18, 20]:
        for sigma in [2, 3, 4, 5, 7]:
            x = np.arange(N, dtype=float)
            center = (N - 1) / 2.0
            h = np.exp(-0.5 * ((x - (center - delta)) / sigma) ** 2)
            h += np.exp(-0.5 * ((x - (center + delta)) / sigma) ** 2)
            starts.append(h)

    # Multi-bump
    for n_bumps in [2, 3, 4]:
        for _ in range(15):
            h = np.zeros(N)
            centers = rng.choice(N, size=n_bumps, replace=False)
            widths = rng.uniform(2, 8, n_bumps)
            for c, w in zip(centers, widths):
                h += np.exp(-0.5 * ((np.arange(N) - c) / w) ** 2)
            starts.append(h)

    # Random smooth
    for _ in range(20):
        h = rng.uniform(0.1, 1.0, N)
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        h = np.convolve(h, kernel, mode='same')
        starts.append(h)

    # Sparse
    for _ in range(15):
        h = np.zeros(N)
        n_nonzero = rng.integers(3, 10)
        indices = rng.choice(N, size=n_nonzero, replace=False)
        h[indices] = rng.uniform(0.5, 2.0, n_nonzero)
        starts.append(h)

    # Lorentzians
    for center in range(0, N, 5):
        for width in [2, 4, 6]:
            x = np.arange(N)
            h = 1.0 / (1.0 + ((x - center) / width) ** 2)
            starts.append(h)

    # Exponential decay from edges
    for side in [0, 1]:
        for rate in [0.1, 0.2, 0.3, 0.5]:
            if side == 0:
                h = np.exp(-rate * np.arange(N))
            else:
                h = np.exp(-rate * (N - 1 - np.arange(N)))
            starts.append(h)

    return starts


def _coordinate_descent(h, max_rounds=30):
    """Coordinate descent refinement."""
    for _ in range(max_rounds):
        if _time_left() < 2.0:
            break
        
        improved = False
        current_ratio = _ratio_from_h(h)
        
        for i in range(N):
            old_val = h[i]
            # Try multiplicative and additive perturbations
            candidates = set()
            for factor in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]:
                candidates.add(max(0.0, old_val * factor))
            for offset in [-0.2, -0.1, 0.1, 0.2, 0.5]:
                candidates.add(max(0.0, old_val + offset))
            candidates.add(0.0)
            
            best_local = h.copy()
            best_local_ratio = current_ratio
            
            for c in candidates:
                test_h = h.copy()
                test_h[i] = c
                test_ratio = _ratio_from_h(test_h)
                if test_ratio > best_local_ratio:
                    best_local_ratio = test_ratio
                    best_local = test_h.copy()
            
            if best_local_ratio > current_ratio + 1e-12:
                h = best_local
                current_ratio = best_local_ratio
                improved = True
        
        if not improved:
            break
    
    return h, _ratio_from_h(h)


def construct():
    global _start
    _start = time.time()

    best_h = np.ones(N)
    best_ratio = _ratio_from_h(best_h)

    # Phase 1: Two-lobe parameter optimization (structural approach)
    if _time_left() > 30:
        h_lobe, r_lobe = _optimize_two_lobe(min(40.0, _time_left() - 20.0))
        if h_lobe is not None and r_lobe > best_ratio:
            best_ratio = r_lobe
            best_h = h_lobe.copy()

    # Phase 2: General starts with local optimization
    starts = _generate_starts()

    for i, h0 in enumerate(starts):
        if _time_left() < 3.0:
            break

        n_remaining = len(starts) - i
        budget = min(2.0, _time_left() / max(n_remaining, 1) - 0.3)
        if budget < 0.4:
            budget = 0.4

        h_opt, R_opt = _local_optimize(h0, budget)

        if R_opt > best_ratio:
            best_ratio = R_opt
            best_h = h_opt.copy()

    # Phase 3: Final polish with tighter tolerance
    if _time_left() > 5.0:
        budget = min(10.0, _time_left() - 2.0)
        h_polished, R_polished = _local_optimize(best_h, budget, beta=50.0)

        if R_polished > best_ratio:
            best_ratio = R_polished
            best_h = h_polished.copy()

    # Phase 4: Coordinate descent refinement
    if _time_left() > 4.0:
        h_cd, r_cd = _coordinate_descent(best_h, max_rounds=20)
        if r_cd > best_ratio:
            best_ratio = r_cd
            best_h = h_cd

    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)

    return list(best_h)
