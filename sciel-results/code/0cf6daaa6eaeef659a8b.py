"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_M = len(_idx_i)  # 165


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)


def _soft_min_area(points, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    min_val = np.min(areas)
    return min_val - np.log(np.sum(np.exp(-beta * (areas - min_val)))) / beta


def _is_inside(points):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -1e-9) and 
            np.all(y <= np.sqrt(3) * x + 1e-9) and 
            np.all(y <= np.sqrt(3) * (1 - x) + 1e-9))


def _params_to_points(params):
    """Convert 22 parameters to 11 points using smooth barycentric mapping.
    Each point uses 3 parameters (a,b,c) mapped via softmax to barycentric coords."""
    points = np.zeros((N, 2))
    for i in range(N):
        # Use 3 params per point, but we only have 22 total for 11 points
        # So use 2 params: u, v in [0,1] mapped to barycentric
        u = params[2 * i]
        v = params[2 * i + 1]
        # Smooth mapping: ensure u,v >= 0 and u+v <= 1
        # Use: a = 1-u-v, b = v, c = u, then normalize if needed
        # Better: use sigmoid to keep in range, then adjust
        # Simple approach: clamp u,v to [0, 0.99] and ensure u+v < 1
        u = np.clip(u, 0.0, 0.99)
        v = np.clip(v, 0.0, 0.99)
        if u + v > 1.0:
            # Reflect smoothly: scale down
            s = 1.0 / (u + v)
            u *= s
            v *= s
        # Barycentric: (1-u-v)*C + u*A + v*B
        # A=(0,0), B=(1,0), C=(0.5, H)
        points[i, 0] = v + 0.5 * (1 - u - v)
        points[i, 1] = H * (1 - u - v)
    return points


def _objective(params):
    """Objective function: negative soft-min area (to minimize)."""
    points = _params_to_points(params)
    if not _is_inside(points):
        return 1e6
    return -_soft_min_area(points, beta=50.0)


def _objective_exact(params):
    """Objective function: negative exact min area (to minimize)."""
    points = _params_to_points(params)
    if not _is_inside(points):
        return 1e6
    return -_min_area(points)


def _soft_min_area_beta(points, beta=100.0):
    """Smooth approximation of min area with higher beta for sharper min."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    min_val = np.min(areas)
    # Use stable log-sum-exp
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - log_sum / beta


def _objective_high_beta(params):
    """Objective with higher beta soft-min for sharper optimization."""
    points = _params_to_points(params)
    if not _is_inside(points):
        return 1e6
    return -_soft_min_area_beta(points, beta=100.0)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple random restarts with differential evolution
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        bounds = [(0, 1)] * (2 * N)
        
        # Use higher beta for first few restarts, then standard
        beta = 100.0 if n_restarts <= 3 else 50.0
        obj_func = _objective_high_beta if beta == 100.0 else _objective
        
        # Run differential evolution with slightly more iterations
        max_iter = 100 if n_restarts == 1 else 60
        pop_size = 45 if n_restarts == 1 else 30
        
        try:
            result = differential_evolution(
                obj_func,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-8,
                seed=n_restarts * 100 + 7,
                polish=True,
                init='latinhypercube'
            )
        except Exception:
            continue
        
        # Evaluate exact objective
        exact_score = -_objective_exact(result.x)
        if exact_score > best_score:
            best_score = exact_score
            best_params = result.x.copy()
        
        # Refine with Nelder-Mead on exact objective
        if time.time() < deadline - 10:
            try:
                res2 = minimize(
                    _objective_exact,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                exact_score2 = -res2.fun
                if exact_score2 > best_score:
                    best_score = exact_score2
                    best_params = res2.x.copy()
            except Exception:
                pass
        
        # Additional local refinement with different initial perturbation
        if time.time() < deadline - 15 and best_params is not None:
            try:
                # Small random perturbation to escape potential local minima
                perturbed = best_params + np.random.normal(0, 0.01, size=(2 * N,))
                perturbed = np.clip(perturbed, 0, 1)
                res3 = minimize(
                    _objective_exact,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                )
                exact_score3 = -res3.fun
                if exact_score3 > best_score:
                    best_score = exact_score3
                    best_params = res3.x.copy()
            except Exception:
                pass
        
        # Check time
        if time.time() > deadline - 20:
            break
    
    if best_params is None:
        # Fallback: symmetric grid placement
        points = np.zeros((N, 2))
        # Place points in a triangular lattice pattern
        positions = [
            (0.5, H * 0.8),    # near top
            (0.25, H * 0.5),   # mid-left
            (0.75, H * 0.5),   # mid-right
            (0.1, H * 0.2),    # bottom-left
            (0.5, H * 0.2),    # bottom-center
            (0.9, H * 0.2),    # bottom-right
            (0.35, H * 0.65),  # upper-mid-left
            (0.65, H * 0.65),  # upper-mid-right
            (0.2, H * 0.35),   # lower-mid-left
            (0.8, H * 0.35),   # lower-mid-right
            (0.5, H * 0.45),   # center
        ]
        for i, (x, y) in enumerate(positions):
            points[i] = [x, y]
        return points
    
    points = _params_to_points(best_params)
    
    # Ensure points are inside triangle (clamp if needed)
    for i in range(N):
        x, y = points[i]
        if y < 0:
            y = 0
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x if x > 0 else 0
            if x == 0:
                y = 0
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x) if x < 1 else 0
            if x == 1:
                y = 0
        points[i] = [x, y]
    
    return points
