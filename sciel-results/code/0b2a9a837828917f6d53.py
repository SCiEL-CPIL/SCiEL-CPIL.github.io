"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid region."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _objective(x):
    """Objective: negative min triangle area. x is flat array of 2N barycentric coords."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _make_constraints():
    """Create linear inequality constraints for barycentric coords."""
    constraints = []
    for i in range(N):
        u_idx = 2 * i
        v_idx = 2 * i + 1
        # u_i >= 0
        A_u = np.zeros(2 * N)
        A_u[u_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_u: A @ x})
        # v_i >= 0
        A_v = np.zeros(2 * N)
        A_v[v_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_v: A @ x})
        # u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
        A_sum = np.zeros(2 * N)
        A_sum[u_idx] = -1.0
        A_sum[v_idx] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_sum: 1.0 + A @ x})
    return constraints

def _make_initial_bary_structured():
    """A well-structured initial configuration."""
    bary_list = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1.0/3, 1.0/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.25],
    ]
    return np.array(bary_list[:N])

def _make_random_bary(seed):
    """Generate random barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for _ in range(1000):
        u = rng.uniform(0.0, 1.0)
        v = rng.uniform(0.0, 1.0)
        if u + v <= 1.0:
            bary_list.append([u, v])
            if len(bary_list) == N:
                break
    while len(bary_list) < N:
        u = rng.uniform(0.1, 0.6)
        v = rng.uniform(0.1, 0.6)
        if u + v < 0.8:
            bary_list.append([u, v])
    return np.array(bary_list[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    constraints = _make_constraints()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_score = -np.inf
    best_bary = None
    best_points = None
    
    # Generate diverse initializations
    inits = []
    
    bary_struct = _make_initial_bary_structured()
    inits.append(bary_struct)
    
    for seed in range(20):
        if time_left() < 5.0:
            break
        bary_r = _make_random_bary(seed)
        inits.append(bary_r)
    
    rng_pert = np.random.RandomState(42)
    for scale in [0.02, 0.05, 0.10, 0.20]:
        if time_left() < 5.0:
            break
        bary_pert = bary_struct + rng_pert.normal(0, scale, size=bary_struct.shape)
        bary_pert = _clamp_bary(bary_pert)
        inits.append(bary_pert)
    
    # Optimize each initialization with SLSQP
    for i, bary0 in enumerate(inits):
        if time_left() < 10.0:
            break
        
        x0 = bary0.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            bary_opt = _clamp_bary(result.x.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            score_opt = _min_triangle_area_vectorized(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
                best_points = points_opt.copy()
                
        except Exception:
            pass
    
    # Additional refinement passes from the best solution
    if time_left() > 15.0 and best_bary is not None:
        rng_refine = np.random.RandomState(123)
        for scale in [0.01, 0.03, 0.08]:
            if time_left() < 10.0:
                break
            bary_pert = best_bary + rng_refine.normal(0, scale, size=best_bary.shape)
            bary_pert = _clamp_bary(bary_pert)
            x0 = bary_pert.flatten()
            
            try:
                result = minimize(
                    _objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 150, 'ftol': 1e-13}
                )
                
                bary_opt = _clamp_bary(result.x.reshape(-1, 2))
                points_opt = _bary_to_cart(bary_opt)
                score_opt = _min_triangle_area_vectorized(points_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_bary = bary_opt.copy()
                    best_points = points_opt.copy()
                    
            except Exception:
                pass
    
    if best_points is None:
        best_points = _bary_to_cart(_make_initial_bary_structured())
    
    return best_points
