import numpy as np
import time
from scipy.optimize import minimize

_N = 10
_EPS = 1e-12
_SLOPES = [-1.0, 0.0, 1.0, 2.0]

def _get_proj_indices(slope):
    n = _N
    indices = np.zeros((n, n), dtype=np.int32)
    v_min = None
    v_max = None
    for y in range(n):
        for x in range(n):
            v = x + slope * y
            if v_min is None:
                v_min = v
                v_max = v
            else:
                v_min = min(v_min, v)
                v_max = max(v_max, v)
    size = int(round(v_max - v_min)) + 1
    for y in range(n):
        for x in range(n):
            v = x + slope * y
            indices[y, x] = int(round(v - v_min))
    return indices, size

PROJ_INDICES = {}
PROJ_SIZES = {}
for s in _SLOPES:
    idx, sz = _get_proj_indices(s)
    PROJ_INDICES[s] = idx
    PROJ_SIZES[s] = sz

PROJ_BINS_FLAT = {s: PROJ_INDICES[s].ravel() for s in _SLOPES}

def _shannon_entropy_bits(counts):
    total = counts.sum()
    if total <= _EPS:
        return 0.0
    p = counts / total
    p = p[p > _EPS]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _compute_entropies(weights, support):
    """
    weights: array of length len(support), sum 1.
    support: array of flat indices (y*N + x).
    Returns (H_minus1, H_0, H_1, H_2).
    """
    Hs = []
    for s in _SLOPES:
        counts = np.zeros(PROJ_SIZES[s], dtype=np.float64)
        bins = PROJ_BINS_FLAT[s][support]
        # Use bincount for speed: weights are non-negative
        counts = np.bincount(bins, weights=weights, minlength=PROJ_SIZES[s])
        Hs.append(_shannon_entropy_bits(counts))
    return Hs[0], Hs[1], Hs[2], Hs[3]

def _ratio_from_entropies(Hm1, H0, H1, H2):
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return Hm1 / M

def _log_objective(theta, support):
    """
    theta: free parameters for softmax.
    Returns negative log-ratio (for minimization).
    Objective = -ln(D/M) = ln(M) - ln(D)
    Uses smooth approximation for max via log-sum-exp to handle non-differentiability.
    """
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    if s < _EPS:
        weights = np.ones(len(support)) / len(support)
    else:
        weights = exp_z / s
        
    Hm1, H0, H1, H2 = _compute_entropies(weights, support)
    if Hm1 <= 1e-12:
        return 1e10
    
    # Smooth max of H0, H1, H2 using log-sum-exp
    # log(sum(exp(H))) approximates log(max) when differences are large
    # We want to minimize ln(M) - ln(D), so we maximize D/M
    # Using soft-max: M_smooth = logsumexp([H0, H1, H2])
    # This is differentiable and smooth
    H_vals = np.array([H0, H1, H2])
    max_H = np.max(H_vals)
    if max_H <= 1e-12:
        return 1e10
    # logsumexp for stability
    M_smooth = max_H + np.log(np.sum(np.exp(H_vals - max_H)))
    
    return M_smooth - np.log(Hm1)

def _objective(theta, support):
    """
    Returns negative ratio (for minimization).
    """
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    if s < _EPS:
        weights = np.ones(len(support)) / len(support)
    else:
        weights = exp_z / s
        
    Hm1, H0, H1, H2 = _compute_entropies(weights, support)
    r = _ratio_from_entropies(Hm1, H0, H1, H2)
    return -r

def _optimize_support(support, deadline, max_iter=200, n_restarts=5, use_log=True):
    """
    Optimize weights for a given support set.
    Returns (best_ratio, best_P_full).
    """
    K = len(support)
    if K == 0:
        return 0.0, None
    
    best_ratio = 0.0
    best_theta = np.zeros(K)
    
    rng = np.random.default_rng(42)
    obj_func = _log_objective if use_log else _objective
    
    # Initial points
    inits = [np.zeros(K)]
    for _ in range(n_restarts - 1):
        inits.append(rng.standard_normal(K) * 0.5)
    
    for theta_start in inits:
        if time.time() > deadline - 2.0:
            break
        try:
            # Use L-BFGS-B
            res = minimize(
                obj_func,
                theta_start,
                args=(support,),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Evaluate ratio on result
            z = res.x - np.max(res.x)
            exp_z = np.exp(z)
            s = exp_z.sum()
            if s < _EPS:
                weights = np.ones(K) / K
            else:
                weights = exp_z / s
                
            Hm1, H0, H1, H2 = _compute_entropies(weights, support)
            ratio = _ratio_from_entropies(Hm1, H0, H1, H2)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_theta = res.x.copy()
                
        except Exception:
            pass

    if best_ratio <= 0:
        return 0.0, None
        
    # Construct full P
    z = best_theta - np.max(best_theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    weights = exp_z / s
    
    P = np.zeros(_N * _N, dtype=np.float64)
    P[support] = weights
    P = P.reshape(_N, _N)
    
    return best_ratio, P

def _greedy_support_search(start_support, deadline, max_additions=8, penalty=0.5):
    """
    Greedy submodular-style support expansion.
    Start from start_support, iteratively add the cell that maximizes
    marginal gain in D while penalizing increase in M.
    Then optimize weights on the final support.
    Returns (best_ratio, best_P).
    """
    current_support = list(start_support)
    current_set = set(current_support)
    
    # Compute initial entropies with uniform weights
    K = len(current_support)
    if K == 0:
        return 0.0, None
    
    uniform_w = np.ones(K) / K
    Hm1, H0, H1, H2 = _compute_entropies(uniform_w, np.array(current_support))
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0, None
    current_ratio = Hm1 / M
    
    # Precompute all cells
    all_cells = np.arange(_N * _N)
    
    for add_iter in range(max_additions):
        if time.time() > deadline - 30.0:
            break
        
        best_gain = -np.inf
        best_cell = None
        best_new_ratio = current_ratio
        
        # For each candidate cell not in current support
        for cell in all_cells:
            if cell in current_set:
                continue
            
            # Temporarily add this cell
            new_support = current_support + [cell]
            new_K = len(new_support)
            new_w = np.ones(new_K) / new_K
            new_Hm1, new_H0, new_H1, new_H2 = _compute_entropies(new_w, np.array(new_support))
            new_M = max(new_H0, new_H1, new_H2)
            if new_M <= 1e-9:
                continue
            new_ratio = new_Hm1 / new_M
            
            # Marginal gain in D
            delta_D = new_Hm1 - Hm1
            # Increase in M
            delta_M = new_M - M
            
            # Score: gain in D minus penalty * increase in M
            gain = delta_D - penalty * max(0, delta_M)
            
            if gain > best_gain:
                best_gain = gain
                best_cell = cell
                best_new_ratio = new_ratio
        
        if best_cell is None or best_new_ratio <= current_ratio * (1 + 1e-6):
            break
        
        # Add the best cell
        current_support.append(best_cell)
        current_set.add(best_cell)
        current_ratio = best_new_ratio
    
    # Optimize weights on the final support
    support_arr = np.array(current_support)
    ratio, P = _optimize_support(support_arr, deadline, max_iter=100, n_restarts=3, use_log=True)
    return ratio, P

def _random_support(rng, min_size=5, max_size=20):
    K = rng.integers(min_size, max_size + 1)
    return rng.choice(_N * _N, size=K, replace=False)

def _structured_supports():
    """Generate some structured supports."""
    supports = []
    n = _N
    # Diagonals x-y = d
    for d in range(-n+1, n):
        cells = []
        for y in range(n):
            x = y + d
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Anti-diagonals x+y = s
    for s in range(0, 2*n-1):
        cells = []
        for y in range(n):
            x = s - y
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Slope 2 lines x - 2y = c -> x = 2y + c
    for c in range(-2*n, 2*n):
        cells = []
        for y in range(n):
            x = 2*y + c
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Rows and Columns
    for i in range(n):
        supports.append(np.array([i * n + x for x in range(n)]))
        supports.append(np.array([y * n + i for y in range(n)]))
        
    return supports

def _get_P_from_params(params):
    """
    Construct P from quadratic parameters.
    params: [A11, A12, A22, b1, b2, c]
    W[y, x] = A11*x^2 + 2*A12*x*y + A22*y^2 + b1*x + b2*y + c
    """
    A11, A12, A22, b1, b2, c = params
    n = _N
    xs = np.arange(n)
    ys = np.arange(n)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    
    W = A11 * X**2 + 2 * A12 * X * Y + A22 * Y**2 + b1 * X + b2 * Y + c
    
    # Softmax
    W_shifted = W - np.max(W)
    P_flat = np.exp(W_shifted)
    P_flat /= np.sum(P_flat)
    P = P_flat.reshape(_N, _N)
    return P

def _objective_full(params):
    P = _get_P_from_params(params)
    # Convert to support/weights for entropy calc
    P_flat = P.flatten()
    # Find support (non-zero)
    mask = P_flat > 1e-10
    support = np.where(mask)[0]
    weights = P_flat[support]
    weights = weights / np.sum(weights)
    
    Hm1, H0, H1, H2 = _compute_entropies(weights, support)
    r = _ratio_from_entropies(Hm1, H0, H1, H2)
    return -r

def _log_objective_full(params):
    P = _get_P_from_params(params)
    P_flat = P.flatten()
    mask = P_flat > 1e-10
    support = np.where(mask)[0]
    weights = P_flat[support]
    weights = weights / np.sum(weights)
    
    Hm1, H0, H1, H2 = _compute_entropies(weights, support)
    if Hm1 <= 1e-12:
        return 1e10
    
    H_vals = np.array([H0, H1, H2])
    max_H = np.max(H_vals)
    if max_H <= 1e-12:
        return 1e10
    M_smooth = max_H + np.log(np.sum(np.exp(H_vals - max_H)))
    
    return M_smooth - np.log(Hm1)

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    timeout = 630.0
    safety_margin = 10.0
    deadline = start_time + timeout - safety_margin
    
    best_P = np.ones((10, 10)) / 100.0
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Full Grid Optimization (Quadratic Params)
    # This explores the full 10x10 space with a parametric family.
    bounds = [
        (-5.0, 5.0),   # A11
        (-5.0, 5.0),   # A12
        (-5.0, 5.0),   # A22
        (-5.0, 5.0),   # b1
        (-5.0, 5.0),   # b2
        (0.0, 0.0)     # c
    ]
    
    init_points = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Uniform
        [2.0, -2.0, 2.0, 0.0, 0.0, 0.0], # Spread x-y
        [2.0, -2.0, 2.0, 1.0, -1.0, 0.0], # Shifted
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],   # Spread x+y
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],   # Mild spread
        [3.0, -3.0, 3.0, 0.0, 0.0, 0.0],  # Strong spread x-y
        [1.0, -0.5, 1.0, 0.0, 0.0, 0.0],  # Moderate
        [-2.0, 2.0, -2.0, 0.0, 0.0, 0.0], # Concentrate x-y
        [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],   # Concentrate x+y
        [4.0, -4.0, 4.0, 0.0, 0.0, 0.0],  # Very strong x-y spread
        [0.0, -3.0, 0.0, 0.0, 0.0, 0.0],  # Only cross term
        [1.5, -1.5, 1.5, 0.5, 0.5, 0.0],  # Shifted diagonal
    ]
    
    for _ in range(15):
        p = rng.uniform([-5, -5, -5, -5, -5, 0], [5, 5, 5, 5, 5, 0])
        init_points.append(p)
    
    for p_init in init_points:
        if time.time() > deadline - 60:
            break
        try:
            # Use log objective for better conditioning
            res = minimize(_log_objective_full, p_init, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < 1e9:
                P = _get_P_from_params(res.x)
                P_flat = P.flatten()
                mask = P_flat > 1e-10
                support = np.where(mask)[0]
                weights = P_flat[support]
                weights = weights / np.sum(weights)
                Hm1, H0, H1, H2 = _compute_entropies(weights, support)
                score = _ratio_from_entropies(Hm1, H0, H1, H2)
                if score > best_score:
                    best_score = score
                    best_P = P.copy()
        except Exception:
            continue

    # Phase 2: Structured Supports + Greedy Expansion
    # Generate structured supports
    supports = _structured_supports()
    
    # Add some random supports
    for _ in range(8):
        supports.append(_random_support(rng, 5, 12))
    
    # Shuffle supports to ensure variety
    rng.shuffle(supports)
    
    for supp in supports:
        if time.time() > deadline - 30:
            break
        if len(supp) < 3:
            continue
            
        # Try greedy expansion from this support
        ratio, P = _greedy_support_search(supp, deadline, max_additions=4, penalty=0.5)
        if P is not None and ratio > best_score:
            best_score = ratio
            best_P = P.copy()
            
        # Also try direct optimization on the initial support (without expansion)
        # This might be faster and find a local optimum
        if time.time() < deadline - 5:
            ratio_direct, P_direct = _optimize_support(supp, deadline - 5, max_iter=40, n_restarts=2, use_log=True)
            if P_direct is not None and ratio_direct > best_score:
                best_score = ratio_direct
                best_P = P_direct.copy()

    # Phase 3: Random Supports with Greedy
    for _ in range(15):
        if time.time() > deadline - 10:
            break
        supp = _random_support(rng, 5, 15)
        ratio, P = _greedy_support_search(supp, deadline, max_additions=4, penalty=0.5)
        if P is not None and ratio > best_score:
            best_score = ratio
            best_P = P.copy()

    # Phase 4: Polish - high-precision optimization on best support found
    if best_score > 0 and time.time() < deadline - 2:
        P_flat = best_P.flatten()
        mask = P_flat > 1e-10
        support = np.where(mask)[0]
        if len(support) >= 3:
            weights = P_flat[support]
            weights = weights / np.sum(weights)
            # Convert to log-weights for optimization
            log_w = np.log(weights + 1e-300)
            
            # Try polishing with tighter tolerances
            try:
                res = minimize(_log_objective, log_w, args=(support,), method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < 1e9:
                    z = res.x - np.max(res.x)
                    exp_z = np.exp(z)
                    s_exp = exp_z.sum()
                    if s_exp > _EPS:
                        weights_new = exp_z / s_exp
                        P_new = np.zeros(_N * _N, dtype=np.float64)
                        P_new[support] = weights_new
                        P_new = P_new.reshape(_N, _N)
                        P_new_flat = P_new.flatten()
                        mask_new = P_new_flat > 1e-10
                        support_new = np.where(mask_new)[0]
                        weights_new = P_new_flat[support_new]
                        weights_new = weights_new / np.sum(weights_new)
                        Hm1, H0, H1, H2 = _compute_entropies(weights_new, support_new)
                        score_new = _ratio_from_entropies(Hm1, H0, H1, H2)
                        if score_new > best_score:
                            best_score = score_new
                            best_P = P_new.copy()
            except Exception:
                pass

    # Final Check and Cleanup
    best_P = np.asarray(best_P, dtype=float)
    if best_P.shape != (10, 10):
        best_P = np.ones((10, 10)) / 100.0
        
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s <= 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s
        
    if not np.all(np.isfinite(best_P)):
        best_P = np.ones((10, 10)) / 100.0

    return best_P
