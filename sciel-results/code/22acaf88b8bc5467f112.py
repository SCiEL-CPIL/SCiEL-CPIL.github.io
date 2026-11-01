"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Hard deadline guard: evaluator kills after 150s, return best with margin
_DEADLINE = time.time() + 120.0

def _time_ok():
    return time.time() < _DEADLINE

def _pairwise_distances(P):
    """Return all pairwise squared distances as a flat array and indices."""
    n = len(P)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = P[idx_i] - P[idx_j]
    sq_dists = np.sum(diffs * diffs, axis=1)
    return sq_dists, idx_i, idx_j

def _score_from_sq_dists(sq_dists):
    """Score is (max dist / min dist)^2 = max_sq / min_sq."""
    dmin = np.min(sq_dists)
    dmax = np.max(sq_dists)
    if dmin < 1e-12:
        return 1e12
    return dmax / dmin

def _objective(params, idx_i, idx_j):
    """Objective: (max pairwise dist / min pairwise dist)^2, with smooth approximation option."""
    P = params.reshape(-1, 3)
    diffs = P[idx_i] - P[idx_j]
    sq_dists = np.sum(diffs * diffs, axis=1)
    dmin = np.min(sq_dists)
    dmax = np.max(sq_dists)
    if dmin < 1e-12:
        return 1e12
    return dmax / dmin

def _objective_smooth(params, idx_i, idx_j, alpha=50.0):
    """Soft-max/soft-min version for smoother gradients."""
    P = params.reshape(-1, 3)
    diffs = P[idx_i] - P[idx_j]
    sq_dists = np.sum(diffs * diffs, axis=1)
    # Use log-sum-exp style soft max/min
    # soft_max = log(sum(exp(alpha * x))) / alpha
    # soft_min = -log(sum(exp(-alpha * x))) / alpha
    # But we want ratio, so let's just use a softened version
    m = np.max(sq_dists)
    M = np.min(sq_dists)
    if M < 1e-12:
        return 1e12
    # Smooth approximation: use a tempered version
    # For the ratio, we can use:
    # soft_ratio = (sum exp(alpha*d)/sum exp(-alpha*d))^(1/alpha) approx
    # Simpler: just use the exact one but add small regularization
    return m / M

def _objective_grad(params, idx_i, idx_j):
    """Gradient of (max/min) w.r.t. parameters. Approximate via finite differences if needed."""
    # For simplicity, use finite differences
    eps = 1e-7
    f0 = _objective(params, idx_i, idx_j)
    grad = np.zeros_like(params)
    n_params = len(params)
    for i in range(n_params):
        params[i] += eps
        f1 = _objective(params, idx_i, idx_j)
        params[i] -= eps
        grad[i] = (f1 - f0) / eps
    return f0, grad

def _make_octahedron_cube(scale=1.0, rot_angle=0.0):
    """
    Create 14 points: 6 octahedron vertices + 8 cube vertices.
    Octahedron: (±1,0,0), (0,±1,0), (0,0,±1) scaled by scale
    Cube: (±1,±1,±1) scaled by scale, rotated around z-axis by rot_angle
    """
    # Octahedron vertices (6 points)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]) * scale
    
    # Cube vertices (8 points)
    cube = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]) * scale
    
    # Rotate cube around z-axis by rot_angle
    c = np.cos(rot_angle)
    s = np.sin(rot_angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    cube = (R @ cube.T).T
    
    return np.vstack([octa, cube])

def _optimize_from_initial(P0, max_iters=5000, timeout=100.0):
    """Optimize starting from P0 using L-BFGS-B."""
    n = len(P0)
    idx_i, idx_j = np.triu_indices(n, k=1)
    params0 = P0.flatten().copy()
    
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > timeout:
            # Return a bad value to stop
            return 1e12, np.zeros_like(x)
        P = x.reshape(-1, 3)
        diffs = P[idx_i] - P[idx_j]
        sq_dists = np.sum(diffs * diffs, axis=1)
        dmin = np.min(sq_dists)
        dmax = np.max(sq_dists)
        if dmin < 1e-12:
            return 1e12, np.zeros_like(x)
        val = dmax / dmin
        
        # Gradient computation
        # f = dmax / dmin
        # df/dx = (ddmax/dx * dmin - dmax * ddmin/dx) / dmin^2
        # Need to find which pair gives max and min
        i_max = np.argmax(sq_dists)
        i_min = np.argmin(sq_dists)
        
        # Gradient of squared distance for pair (a,b): 2*(P[a]-P[b]) for P[a], -2*(P[a]-P[b]) for P[b]
        grad = np.zeros_like(x)
        
        # For max pair
        a, b = idx_i[i_max], idx_j[i_max]
        diff_max = P[a] - P[b]
        grad[a*3:(a+1)*3] += 2 * diff_max
        grad[b*3:(b+1)*3] -= 2 * diff_max
        
        # For min pair
        a, b = idx_i[i_min], idx_j[i_min]
        diff_min = P[a] - P[b]
        grad[a*3:(a+1)*3] += -2 * diff_min * dmax / dmin
        grad[b*3:(b+1)*3] += 2 * diff_min * dmax / dmin
        
        # Full gradient: df = (ddmax * dmin - dmax * ddmin) / dmin^2
        # ddmax/dx = 2*(P[a]-P[b]) for max pair
        # ddmin/dx = 2*(P[a]-P[b]) for min pair
        # So df/dx = [2*(P[a_max]-P[b_max]) * dmin - dmax * 2*(P[a_min]-P[b_min])] / dmin^2
        # = 2/dmin^2 * [(P[a_max]-P[b_max]) * dmin - dmax * (P[a_min]-P[b_min])]
        
        # Let me redo this more carefully
        # f = dmax / dmin
        # Let dm = dmax, dn = dmin
        # df/dx = (dm' * dn - dm * dn') / dn^2
        # where dm' = d(dm)/dx, dn' = d(dn)/dx
        # dm = ||P[a]-P[b]||^2 for max pair, so dm' = 2(P[a]-P[b]) for P[a], -2(P[a]-P[b]) for P[b]
        # dn = ||P[c]-P[d]||^2 for min pair, so dn' = 2(P[c]-P[d]) for P[c], -2(P[c]-P[d]) for P[d]
        
        grad = np.zeros_like(x)
        
        # Contribution from dm' * dn / dn^2 = dm' / dn
        a, b = idx_i[i_max], idx_j[i_max]
        diff_max = P[a] - P[b]
        grad[a*3:(a+1)*3] += 2 * diff_max / dmin
        grad[b*3:(b+1)*3] += -2 * diff_max / dmin
        
        # Contribution from -dm * dn' / dn^2 = -dm * dn' / dn^2
        a, b = idx_i[i_min], idx_j[i_min]
        diff_min = P[a] - P[b]
        grad[a*3:(a+1)*3] += -2 * dmax * diff_min / (dmin * dmin)
        grad[b*3:(b+1)*3] += 2 * dmax * diff_min / (dmin * dmin)
        
        return val, grad
    
    result = minimize(obj_and_grad, params0, method='L-BFGS-B', 
                      options={'maxiter': max_iters, 'maxfun': max_iters * 10,
                               'gtol': 1e-10, 'ftol': 1e-12})
    
    if result.success or result.fun < 1e12:
        P_opt = result.x.reshape(-1, 3)
        score = _score_from_sq_dists(_pairwise_distances(P_opt)[0])
        return P_opt, score
    
    return P0, _score_from_sq_dists(_pairwise_distances(P0)[0])

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 120.0
    
    best_P = None
    best_score = 1e12
    
    # Generate initial candidates using octahedron + cube parameterization
    candidates = []
    
    # Try various scale and rotation angles
    for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
        for rot in [0.0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            if not _time_ok():
                break
            P = _make_octahedron_cube(scale, rot)
            candidates.append(P)
    
    # Also try some random perturbations of a base configuration
    base = _make_octahedron_cube(1.0, np.pi/6)
    for trial in range(20):
        if not _time_ok():
            break
        noise = np.random.randn(14, 3) * 0.1
        candidates.append(base + noise)
    
    # Try a pure random initialization
    for trial in range(5):
        if not _time_ok():
            break
        P = np.random.randn(14, 3)
        candidates.append(P)
    
    # Optimize each candidate
    for P0 in candidates:
        if not _time_ok():
            break
        P_opt, score = _optimize_from_initial(P0, max_iters=2000, timeout=8.0)
        if score < best_score:
            best_score = score
            best_P = P_opt.copy()
    
    # Final high-precision polish
    if best_P is not None and _time_ok():
        P_polished, score_polished = _optimize_from_initial(best_P, max_iters=5000, timeout=60.0)
        if score_polished < best_score:
            best_score = score_polished
            best_P = P_polished.copy()
    
    if best_P is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_P = np.array(grid, dtype=float)
    
    return best_P
# EVOLVE-BLOCK-END
