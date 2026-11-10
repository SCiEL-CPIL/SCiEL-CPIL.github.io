# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_EPS = 1e-12
_SLOPES = [-1.0, 0.0, 1.0, 2.0]

# Precompute projection indices for each slope
# P[y, x] -> index y*N + x
# v = x + slope * y
# We need to map each cell to a bin index for counts.
# v_min and v_max depend on slope.
# Slope -1: v = x - y. Range: -9 to 9. Size 19.
# Slope 0: v = x. Range: 0 to 9. Size 10.
# Slope 1: v = x + y. Range: 0 to 18. Size 19.
# Slope 2: v = x + 2y. Range: 0 to 27. Size 28.

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

# Precompute flat bin indices for fast access
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
        np.add.at(counts, bins, weights)
        Hs.append(_shannon_entropy_bits(counts))
    return Hs[0], Hs[1], Hs[2], Hs[3]

def _ratio_from_entropies(Hm1, H0, H1, H2):
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return Hm1 / M

def _objective_on_support(theta, support):
    """
    theta: free parameters for softmax.
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

def _optimize_support(support, deadline, max_iter=150, n_restarts=5):
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
    attempts = 0
    
    while time.time() < deadline - 1.0 and attempts < n_restarts:
        attempts += 1
        if attempts == 1:
            theta_start = np.zeros(K)
        else:
            theta_start = rng.standard_normal(K) * 0.5
        
        try:
            res = minimize(
                _objective_on_support,
                theta_start,
                args=(support,),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun is not None:
                ratio = -res.fun
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_theta = res.x
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
    ratio, P = _optimize_support(support_arr, deadline, max_iter=100, n_restarts=3)
    return ratio, P

def _random_support(rng, min_size=5, max_size=20):
    K = rng.integers(min_size, max_size + 1)
    return rng.choice(_N * _N, size=K, replace=False)

def _structured_supports():
    """Generate some structured supports."""
    supports = []
    n = _N
    # Diagonals
    for d in range(-n+1, n):
        cells = []
        for y in range(n):
            x = y + d
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Anti-diagonals
    for s in range(0, 2*n-1):
        cells = []
        for y in range(n):
            x = s - y
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Slope -2 lines (x + 2y = c)
    for c in range(0, 2*n):
        cells = []
        for y in range(n):
            x = c - 2*y
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Slope 2 lines (x - 2y = c) -> x = 2y + c
    for c in range(-2*n, 2*n):
        cells = []
        for y in range(n):
            x = 2*y + c
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Rows and Cols
    for i in range(n):
        supports.append(np.array([i * n + j for j in range(n)])) # Row i
        supports.append(np.array([j * n + i for j in range(n)])) # Col i
        
    return supports

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # 10s margin
    
    best_ratio = 0.0
    best_P = np.ones((_N, _N)) / (_N * _N)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Try structured supports with direct optimization
    struct_supports = _structured_supports()
    for support in struct_supports:
        if time.time() > deadline - 60.0:
            break
        ratio, P = _optimize_support(support, deadline, max_iter=150, n_restarts=5)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
    
    # Phase 2: Greedy support search from various starting points
    # Start from some structured supports and random supports
    greedy_starts = []
    
    # Add some structured starts (smaller ones for more room to grow)
    for support in struct_supports:
        if 3 <= len(support) <= 8:
            greedy_starts.append(support)
    
    # Add random starts
    for _ in range(20):
        greedy_starts.append(_random_support(rng, min_size=3, max_size=8))
    
    for start_sup in greedy_starts:
        if time.time() > deadline - 30.0:
            break
        ratio, P = _greedy_support_search(start_sup, deadline, max_additions=8, penalty=0.5)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
    
    # Phase 3: Random support optimization with varying sizes
    num_random = 0
    while time.time() < deadline - 10.0:
        num_random += 1
        if num_random % 10 == 0:
            min_sz, max_sz = 5, 10
        elif num_random % 10 == 5:
            min_sz, max_sz = 10, 15
        else:
            min_sz, max_sz = 15, 25
            
        support = _random_support(rng, min_sz, max_sz)
        ratio, P = _optimize_support(support, deadline, max_iter=150, n_restarts=5)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
        
        # Periodically try greedy from current best
        if num_random % 30 == 0 and best_ratio > 0:
            nz = np.nonzero(best_P.ravel() > 1e-6)[0]
            if len(nz) > 0:
                # Use a subset as starting point for greedy
                if len(nz) > 10:
                    subset = rng.choice(nz, size=8, replace=False)
                else:
                    subset = nz
                ratio, P = _greedy_support_search(subset, deadline, max_additions=6, penalty=0.5)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P

    # Final validation and normalization
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = P / s
        
    return P

# EVOLVE-BLOCK-END
