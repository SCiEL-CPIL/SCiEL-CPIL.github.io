"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, LinearConstraint

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


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross).min()


def _is_inside(points, eps=1e-9):
    """Check if all points are inside or on the boundary of the triangle."""
    # For point (x,y) to be in triangle with vertices (0,0), (1,0), (0.5, H):
    # y >= 0, y <= H*(1-x) for x in [0, 0.5], y <= H*x for x in [0.5, 1]
    # More generally: barycentric coords all >= 0
    # Point is inside if:
    # 1) y >= 0
    # 2) y <= H (above bottom)
    # 3) For the left edge (0,0)-(0.5,H): x >= 0 and y <= H * 2 * x => 2x - y/H >= 0... 
    # Actually let's use the standard barycentric test.
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5, H)
    # For point P, compute barycentric coordinates.
    # A simpler check: 
    #   y >= -eps
    #   y <= H * (1 - x) + eps  ... no, that's not right either.
    
    # Let's use the half-plane method:
    # Edge AB: from (0,0) to (1,0). Interior is y >= 0.
    # Edge BC: from (1,0) to (0.5, H). Direction: (-0.5, H). Normal pointing inward: (H, 0.5)
    #   Line: H*(x-1) + 0.5*(y-0) <= 0  => H*x + 0.5*y - H <= 0
    # Edge CA: from (0.5, H) to (0,0). Direction: (-0.5, -H). Normal pointing inward: (H, -0.5)
    #   Line: H*(x-0) + (-0.5)*(y-0) >= 0  => H*x - 0.5*y >= 0
    
    xs = points[:, 0]
    ys = points[:, 1]
    
    cond1 = ys >= -eps
    cond2 = H * xs + 0.5 * ys - H <= eps
    cond3 = H * xs - 0.5 * ys >= -eps
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)


def _params_to_points(params):
    """Convert 22 parameters (11 x-coords, 11 y-coords) to point array."""
    pts = params.reshape(N, 2)
    return pts


def _objective(params):
    """Objective: negative of min area (to minimize). Returns large penalty if invalid."""
    pts = _params_to_points(params)
    if not _is_inside(pts):
        return 1e6
    return -_min_area(pts)


def _refine_local(pts, time_limit=None):
    """Local refinement using Nelder-Mead on the non-smooth min-area objective."""
    start_time = time.time()
    best_pts = pts.copy()
    best_val = _min_area(best_pts)
    
    # Try Nelder-Mead
    try:
        res = minimize(_objective, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if _is_inside(res.x.reshape(N, 2)):
            val = _min_area(res.x.reshape(N, 2))
            if val > best_val:
                best_pts = res.x.reshape(N, 2)
                best_val = val
    except Exception:
        pass
    
    if time_limit is not None and time.time() - start_time > time_limit:
        return best_pts
    
    # Try a few random perturbations and local optimize
    rng = np.random.RandomState(42)
    for _ in range(20):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        perturbed = best_pts + rng.normal(0, 0.01, size=best_pts.shape)
        if _is_inside(perturbed):
            try:
                res = minimize(_objective, perturbed.flatten(), method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if _is_inside(res.x.reshape(N, 2)):
                    val = _min_area(res.x.reshape(N, 2))
                    if val > best_val:
                        best_pts = res.x.reshape(N, 2)
                        best_val = val
            except Exception:
                pass
    
    return best_pts


def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N
    
    best_pts = None
    best_val = -1
    
    # Use differential evolution with multiple restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if best_pts is not None and restart > 0:
            # Use best found so far as seed for next restart
            seed = best_pts.flatten()
        else:
            seed = None
        
        try:
            de_result = differential_evolution(
                _objective, bounds, seed=seed, maxiter=100, popsize=15,
                tol=1e-6, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, init='latinhypercube' if restart == 0 else 'sobol'
            )
            if _is_inside(de_result.x.reshape(N, 2)):
                val = _min_area(de_result.x.reshape(N, 2))
                if val > best_val:
                    best_val = val
                    best_pts = de_result.x.reshape(N, 2)
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Local refinement
    if best_pts is not None:
        remaining = max(0, deadline - time.time() - 5)
        if remaining > 1:
            best_pts = _refine_local(best_pts, time_limit=remaining)
    
    if best_pts is None:
        # Fallback: random points
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        best_pts = np.array(points)
    
    return best_pts
