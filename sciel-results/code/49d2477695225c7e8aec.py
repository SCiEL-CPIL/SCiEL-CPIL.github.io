"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _soft_min_1d(vals, beta=20.0):
    """
    Smooth approximation to min using log-sum-exp.
    For a 1D array, returns a scalar soft-min.
    soft_min ≈ -(1/beta) * log(sum(exp(-beta * x)))
    """
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return 0.0
    # Use the minimum as reference for numerical stability
    m = np.min(v)
    # If all values are very close, return the min
    if m > 1e15:  # all inf
        return m
    shifted = v - m
    # For large beta*shifted, exp(-beta*shifted) underflows to 0, which is fine
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return m - log_sum / beta

def _soft_min_rows(mat, beta=20.0):
    """
    Apply soft-min along axis=1 of a 2D matrix.
    mat: shape (n, m) -> returns shape (n,)
    """
    if mat.ndim == 1:
        return np.array([_soft_min_1d(mat, beta)])
    n, m = mat.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _soft_min_1d(mat[i], beta)
    return out

def _objective_smooth(pos, beta=15.0):
    """
    Smooth objective: minimize negative sum of clearances.
    Uses soft-min for differentiability.
    pos: shape (2*N,)
    """
    p = pos.reshape(N, 2)
    
    # Pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    # Distance to box boundaries
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    # For each circle i, clearance is min(dist_to_box[i], 0.5 * min_j dist(i,j))
    # The inner min over j is where we need soft-min
    # dist[i, :] has inf at diagonal, so min over j≠i
    min_dist_to_others = _soft_min_rows(dist, beta=beta)
    
    # Clearance for each circle
    # r_i <= min(dist_to_box[i], 0.5 * min_dist_to_others[i])
    # We want to maximize sum of r_i, so minimize -sum(r_i)
    # r_i = min(dist_to_box[i], 0.5 * min_dist_to_others[i])
    # Use soft-min for the outer min too
    clearances = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    
    # For a smoother objective, we can also soften the outer min
    # But for now, let's keep it simple and just use the soft inner min
    return -np.sum(clearances)

def _objective_exact(pos):
    """
    Exact (nonsmooth) objective for final polish.
    """
    p = pos.reshape(N, 2)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    clearances = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    return -np.sum(clearances)

def _get_valid_packing(pos, safety=0.99995):
    """
    Convert positions to a valid packing with radii.
    """
    p = pos.reshape(N, 2).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 1.0 - 0.0, 1.0)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    radii = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    radii *= safety
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    return p, radii

def _hex_initialization():
    """Generate a hexagonal grid initialization with N points."""
    pts = []
    # 6x6 grid with staggering
    for i in range(6):
        for j in range(6):
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            if i % 2 == 1:
                y += 0.5 / 6.0 * 0.5
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append([x, y])
    
    if len(pts) < N:
        np.random.seed(42)
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
    
    return np.array(pts[:N]).flatten()

def _rect_initialization():
    """Rectangular grid initialization."""
    pts = []
    n_per_side = int(np.ceil(np.sqrt(N)))
    for i in range(n_per_side):
        for j in range(n_per_side):
            if len(pts) >= N:
                break
            x = (i + 0.5) / n_per_side
            y = (j + 0.5) / n_per_side
            pts.append([x, y])
    
    while len(pts) < N:
        pts.append([0.5, 0.5])
    
    return np.array(pts[:N]).flatten()

def _random_initialization(seed=None):
    """Random initialization."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(2 * N)

def _optimize_phase1(init_pos, deadline):
    """
    Phase 1: Coarse optimization with smooth objective.
    """
    best_x = init_pos.copy()
    best_fun = _objective_smooth(init_pos)
    
    # Try multiple beta values for soft-min
    for beta in [10.0, 15.0, 25.0]:
        if time.time() > deadline - 15.0:
            break
        
        def obj_smooth(pos):
            return _objective_smooth(pos, beta=beta)
        
        try:
            res = minimize(obj_smooth, best_x, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x
        except Exception:
            continue
    
    return best_x

def _optimize_phase2(init_pos, deadline):
    """
    Phase 2: Polish with exact objective.
    """
    best_x = init_pos.copy()
    best_fun = _objective_exact(init_pos)
    
    try:
        res = minimize(_objective_exact, best_x, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Multiple initializations
    inits = [
        _hex_initialization(),
        _rect_initialization(),
        _random_initialization(1),
        _random_initialization(2),
        _random_initialization(3),
        _random_initialization(4),
    ]
    
    # Phase 1: Coarse optimization for each init
    phase1_results = []
    for init_pos in inits:
        if time.time() > deadline - 30.0:
            break
        
        result = _optimize_phase1(init_pos, deadline)
        phase1_results.append(result)
    
    # Phase 2: Polish the best phase-1 results
    # Sort by phase-1 objective and polish top few
    scored_phase1 = []
    for r in phase1_results:
        if r is not None:
            fun = _objective_exact(r)
            scored_phase1.append((fun, r))
    
    scored_phase1.sort(key=lambda x: x[0])
    
    # Polish top 3
    for i, (fun, pos) in enumerate(scored_phase1[:3]):
        if time.time() > deadline - 10.0:
            break
        
        polished = _optimize_phase2(pos, deadline)
        
        pos_arr, radii = _get_valid_packing(polished)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack((pos_arr, radii))
    
    # Fallback
    if best_result is None:
        pos, radii = _get_valid_packing(phase1_results[0])
        best_result = np.column_stack((pos, radii))
    
    return best_result

# EVOLVE-BLOCK-END
