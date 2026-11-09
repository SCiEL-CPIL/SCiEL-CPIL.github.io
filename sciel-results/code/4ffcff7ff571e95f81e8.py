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
    # Use stable log-sum-exp
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - log_sum / beta


def _params_to_points(params):
    """
    Convert 22 parameters to 11 points.
    Params are in [0, 1].
    Mapping: s = p[2i], t = p[2i+1]
    u = s * (1 - t)
    v = s * t
    P = (1-u-v)A + uB + vC
    """
    points = np.zeros((N, 2))
    s = params[0::2]
    t = params[1::2]
    u = s * (1 - t)
    v = s * t
    
    # Vectorized barycentric to cartesian
    # A=(0,0), B=(1,0), C=(0.5, H)
    # P_x = u*1 + v*0.5
    # P_y = v*H
    points[:, 0] = u + 0.5 * v
    points[:, 1] = H * v
    return points


def _objective_soft(params, beta=50.0):
    """Objective function: negative soft-min area (to minimize)."""
    points = _params_to_points(params)
    return -_soft_min_area(points, beta)


def _objective_exact(params):
    """Objective function: negative exact min area (to minimize)."""
    points = _params_to_points(params)
    return -_min_area(points)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    best_params = None
    best_score = -np.inf
    
    bounds = [(0, 1)] * (2 * N)
    
    n_restarts = 0
    
    # We will run DE multiple times.
    # Each DE run is followed by NM refinement.
    
    while time.time() < deadline - 10:
        n_restarts += 1
        
        # Use a beta that is smooth enough for DE but sharp enough to guide towards min
        # Start with lower beta, increase? Or just use a fixed moderate one.
        beta = 40.0
        
        # DE Parameters
        # 22 dimensions. Popsize 40 is reasonable.
        # Maxiter 150.
        max_iter = 150
        pop_size = 45
        
        try:
            result = differential_evolution(
                _objective_soft,
                bounds,
                args=(beta,),
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-9,
                seed=n_restarts * 1000 + 42,
                polish=False,
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
        if time.time() < deadline - 5:
            try:
                res2 = minimize(
                    _objective_exact,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                exact_score2 = -res2.fun
                if exact_score2 > best_score:
                    best_score = exact_score2
                    best_params = res2.x.copy()
            except Exception:
                pass
        
        # Additional local refinement with different initial perturbation
        if time.time() < deadline - 5 and best_params is not None:
            try:
                # Small random perturbation to escape potential local minima
                perturbed = best_params + np.random.normal(0, 0.02, size=(2 * N,))
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
        if time.time() > deadline - 15:
            break
    
    if best_params is None:
        # Fallback: symmetric grid placement
        points = np.zeros((N, 2))
        # Place points in a triangular lattice pattern
        # Approximate lattice
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
    
    # Ensure points are inside triangle (clamp if needed, though mapping should keep them inside)
    # The mapping u=s(1-t), v=st with s,t in [0,1] guarantees u,v >= 0 and u+v <= 1.
    # So points are strictly inside or on the boundary.
    
    return points
