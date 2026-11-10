# EVOLVE-BLOCK-START
import time
import numpy as np
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

def _soft_max_smooth(H0, H1, H2, beta=10.0):
    H = np.array([H0, H1, H2])
    H_max = np.max(H)
    if H_max <= _EPS:
        return _EPS
    z = beta * (H - H_max)
    z_clipped = np.clip(z, -50, 50)
    log_sum = np.log(np.sum(np.exp(z_clipped)))
    return H_max + log_sum / beta

def _objective_smooth(theta, support, beta=10.0):
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    if s < _EPS:
        weights = np.ones(len(support)) / len(support)
    else:
        weights = exp_z / s
        
    Hm1, H0, H1, H2 = _compute_entropies(weights, support)
    M_smooth = _soft_max_smooth(H0, H1, H2, beta)
    if M_smooth <= 1e-9:
        return 0.0
    r = Hm1 / M_smooth
    return -r

def _optimize_support(support, deadline, max_iter=200, n_restarts=8, use_smooth=True):
    K = len(support)
    if K == 0:
        return 0.0, None
    
    best_ratio = 0.0
    best_theta = np.zeros(K)
    
    rng = np.random.default_rng(42)
    attempts = 0
    
    # First pass with smooth objective
    if use_smooth:
        for beta in [5.0, 10.0, 20.0]:
            while time.time() < deadline - 1.0 and attempts < n_restarts:
                attempts += 1
                if attempts == 1:
                    theta_start = np.zeros(K)
                else:
                    theta_start = rng.standard_normal(K) * 0.5
                
                try:
                    res = minimize(
                        _objective_smooth,
                        theta_start,
                        args=(support, beta),
                        method='L-BFGS-B',
                        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.fun is not None:
                        # Evaluate with true objective
                        z = res.x - np.max(res.x)
                        exp_z = np.exp(z)
                        s = exp_z.sum()
                        weights = exp_z / s
                        Hm1, H0, H1, H2 = _compute_entropies(weights, support)
                        ratio = _ratio_from_entropies(Hm1, H0, H1, H2)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_theta = res.x
                except Exception:
                    pass
    
    # Second pass with exact objective for polishing
    for _ in range(3):
        if time.time() > deadline - 1.0:
            break
        try:
            res = minimize(
                _objective_on_support,
                best_theta,
                args=(support,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
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
        
    z = best_theta - np.max(best_theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    weights = exp_z / s
    
    P = np.zeros(_N * _N, dtype=np.float64)
    P[support] = weights
    P = P.reshape(_N, _N)
    
    return best_ratio, P

def _simulated_annealing(support, deadline, max_iters=5000, T0=1.0, T_end=0.01):
    """
    Simulated annealing that swaps probability mass between cells on the support.
    """
    K = len(support)
    if K < 2:
        return 0.0, None
    
    # Start with uniform weights
    theta = np.zeros(K)
    rng = np.random.default_rng(123)
    
    def _get_weights(th):
        z = th - np.max(th)
        exp_z = np.exp(z)
        s = exp_z.sum()
        if s < _EPS:
            return np.ones(K) / K
        return exp_z / s
    
    def _get_ratio(th):
        w = _get_weights(th)
        Hm1, H0, H1, H2 = _compute_entropies(w, support)
        return _ratio_from_entropies(Hm1, H0, H1, H2)
    
    current_theta = theta.copy()
    current_ratio = _get_ratio(current_theta)
    best_theta = current_theta.copy()
    best_ratio = current_ratio
    
    n_iters = max_iters
    for i in range(n_iters):
        if time.time() > deadline - 2.0:
            break
        
        # Cooling schedule
        T = T0 * (T_end / T0) ** (i / max(1, n_iters - 1))
        
        # Random swap: pick two indices and transfer some mass
        idx1, idx2 = rng.integers(0, K, size=2)
        if idx1 == idx2:
            continue
        
        # Transfer a small amount of mass
        delta = rng.uniform(-0.5, 0.5)
        new_theta = current_theta.copy()
        new_theta[idx1] += delta
        new_theta[idx2] -= delta
        
        new_ratio = _get_ratio(new_theta)
        delta_ratio = new_ratio - current_ratio
        
        # Accept if better, or with probability exp(delta/T)
        if delta_ratio > 0 or rng.random() < np.exp(delta_ratio / max(T, 1e-10)):
            current_theta = new_theta
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_theta = current_theta.copy()
    
    z = best_theta - np.max(best_theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    weights = exp_z / s
    
    P = np.zeros(_N * _N, dtype=np.float64)
    P[support] = weights
    P = P.reshape(_N, _N)
    
    return best_ratio, P

def _polish_support(support, P_current, deadline):
    """
    High-precision polish on the support using the current P as initialization.
    """
    K = len(support)
    if K == 0:
        return 0.0, None
    
    # Extract current weights on support
    weights = P_current.ravel()[support]
    total = weights.sum()
    if total < _EPS:
        weights = np.ones(K) / K
    else:
        weights = weights / total
    
    # Convert to theta
    theta = np.log(weights + _EPS)
    theta = theta - np.max(theta)
    
    # Multiple restarts with tight tolerance
    best_ratio = 0.0
    best_theta = theta.copy()
    rng = np.random.default_rng(99)
    
    for restart in range(5):
        if time.time() > deadline - 1.0:
            break
        
        if restart == 0:
            theta_start = theta.copy()
        else:
            theta_start = theta + rng.standard_normal(K) * 0.1
        
        try:
            res = minimize(
                _objective_on_support,
                theta_start,
                args=(support,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
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
        
    z = best_theta - np.max(best_theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    weights = exp_z / s
    
    P = np.zeros(_N * _N, dtype=np.float64)
    P[support] = weights
    P = P.reshape(_N, _N)
    
    return best_ratio, P

def _structured_supports():
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
            
    # Slope -2 lines
    for c in range(0, 2*n):
        cells = []
        for y in range(n):
            x = c - 2*y
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Slope 2 lines
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
        supports.append(np.array([i * n + j for j in range(n)]))
        supports.append(np.array([j * n + i for j in range(n)]))
        
    return supports

def _random_support(rng, min_size=5, max_size=20):
    K = rng.integers(min_size, max_size + 1)
    return rng.choice(_N * _N, size=K, replace=False)

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # 10s margin before 630s kill
    
    best_ratio = 0.0
    best_P = np.ones((_N, _N)) / (_N * _N)
    best_support = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Structured supports with smooth + polish optimization
    struct_supports = _structured_supports()
    for support in struct_supports:
        if time.time() > deadline - 120.0:
            break
        ratio, P = _optimize_support(support, deadline, max_iter=150, n_restarts=5, use_smooth=True)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
            best_support = support
    
    # Phase 2: Simulated annealing on best support and structured supports
    if best_support is not None:
        ratio, P = _simulated_annealing(best_support, deadline, max_iters=3000)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
    
    for support in struct_supports[:10]:
        if time.time() > deadline - 60.0:
            break
        ratio, P = _simulated_annealing(support, deadline, max_iters=2000)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
            best_support = support
    
    # Phase 3: Random support optimization
    num_random = 0
    while time.time() < deadline - 30.0:
        num_random += 1
        if num_random % 5 == 0:
            min_sz, max_sz = 5, 12
        elif num_random % 5 == 3:
            min_sz, max_sz = 10, 18
        else:
            min_sz, max_sz = 15, 25
            
        support = _random_support(rng, min_sz, max_sz)
        ratio, P = _optimize_support(support, deadline, max_iter=100, n_restarts=3, use_smooth=True)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
            best_support = support
    
    # Phase 4: Final high-precision polish
    if best_support is not None and time.time() < deadline - 5.0:
        ratio, P = _polish_support(best_support, best_P, deadline)
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
