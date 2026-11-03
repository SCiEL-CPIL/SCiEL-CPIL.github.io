"""Heilbronn-11: Differential Evolution Global Search with Nelder-Mead Polish."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

DEADLINE = 140.0

def _bary_to_cart(uv_flat):
    """
    Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    u, v are coordinates relative to V0, V1, V2.
    P = u*V0 + v*V1 + w*V2, where w = 1-u-v.
    V0=(0,0), V1=(1,0), V2=(0.5, S3H)
    x = v*1 + w*0.5 = v + 0.5(1-u-v) = 0.5 + 0.5v - 0.5u
    y = w*S3H = (1-u-v)*S3H
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = 0.5 + 0.5 * v - 0.5 * u
    y = w * S3H
    return np.column_stack((x, y))

def _cart_to_bary(pts):
    """
    Convert Cartesian points to barycentric coords (u, v).
    x = 0.5 + 0.5v - 0.5u
    y = (1-u-v)S3H => 1-u-v = y/S3H => u+v = 1 - y/S3H
    Let s = u+v.
    x = 0.5 + 0.5v - 0.5(s-v) = 0.5 + 0.5v - 0.5s + 0.5v = 0.5 + v - 0.5s
    v = x - 0.5 + 0.5s
    u = s - v = s - (x - 0.5 + 0.5s) = 0.5s - x + 0.5
    """
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        w = y / S3H
        s = 1.0 - w
        v = x - 0.5 + 0.5 * s
        u = 0.5 * s - x + 0.5
        uv[i] = [u, v]
    return uv

def _vectorized_min_area_cart(pts):
    """Vectorized computation of min triangle area from Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _project_to_simplex(uv_flat):
    """Project barycentric coordinates to the valid simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(-1, 2).copy()
    for i in range(len(uv)):
        u, v = uv[i]
        # Simple projection: clamp to [0,1] and then adjust if sum > 1
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        if u + v > 1.0:
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        uv[i] = [u, v]
    return uv.flatten()

def _objective_bary(uv_flat):
    """
    Objective function for minimization: -min_area.
    Includes a penalty if points are slightly outside due to numerical issues, 
    though DE bounds should keep them in.
    """
    # Project to ensure validity in case of numerical drift
    uv_proj = _project_to_simplex(uv_flat)
    pts = _bary_to_cart(uv_proj)
    
    # Check validity strictly
    # If any point is significantly outside, return a large penalty
    # For DE, bounds are [0,1] for u and v. 
    # The condition u+v<=1 is not a simple box bound.
    # The projection handles it.
    
    min_area = _vectorized_min_area_cart(pts)
    return -min_area

def construct():
    start_time = time.time()
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)

    # Initial best
    # Use a symmetric starting point
    init_uv = np.zeros((N, 2))
    # Place points roughly in a grid or symmetric pattern
    # 3 corners, 3 edge midpoints, 3 inner points?
    # Let's just use a random start for DE, but keep a good initial guess
    # Symmetric points in barycentric:
    # (0,0), (1,0), (0,1) -> Corners
    # (0.5, 0), (0.5, 0.5), (0, 0.5) -> Edge midpoints? No, (0.5, 0.5) is center.
    # Let's use a known good configuration or random.
    
    # Random start for DE
    rng = np.random.default_rng(42)
    
    # Define bounds for DE: u in [0,1], v in [0,1]
    # The constraint u+v<=1 is handled by the projection in the objective.
    # However, DE works best with box constraints. The projection makes the landscape 
    # non-smooth at the boundaries u+v=1. 
    # Alternative: Parameterize with u, v, w normalized? No, 2 DOF is enough.
    # Let's stick to box constraints and projection.
    
    bounds = [(0, 1)] * (2 * N)
    
    best_pts = None
    best_area = 0.0

    def _update_best(uv_flat):
        nonlocal best_pts, best_area
        uv_proj = _project_to_simplex(uv_flat)
        pts = _bary_to_cart(uv_proj)
        
        # Strict validity check
        for i in range(len(pts)):
            x, y = pts[i]
            if y < -1e-9 or y > S3H + 1e-9:
                return
            x_min = y / S3
            x_max = 1 - y / S3
            if x < x_min - 1e-9 or x > x_max + 1e-9:
                return
        
        area = _vectorized_min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Phase 1: Differential Evolution
    # We will run DE for a fixed number of iterations or time.
    # To respect the deadline, we run DE in a loop, checking time.
    # DE in SciPy runs to convergence or maxiter. 
    # We can set maxiter to a large number and rely on the time check? 
    # No, DE is a black box. 
    # Strategy: Run DE for a specific number of iterations that fits in the time budget.
    # Estimation: Each iteration evaluates pop_size * (N+1) points.
    # Pop size 50, N=22 vars. 50 * 23 = 1150 evaluations per generation.
    # Each evaluation is fast (~100 us). 1150 * 100us = 0.1s per generation.
    # We can afford ~500 generations in 50s.
    # Let's run DE with maxiter=200 first.
    
    if _elapsed_ok(10):
        try:
            # Run DE
            # seed=42 for reproducibility
            de_result = differential_evolution(
                _objective_bary,
                bounds,
                maxiter=200,
                popsize=20, # 20 * 22 = 440 points in population
                tol=1e-6,
                seed=42,
                polish=False, # We will do our own polish
                init='sobol' # Better initial spread
            )
            _update_best(de_result.x)
        except Exception as e:
            print(f"DE failed: {e}")
            
    # Phase 2: Local Refinement with Nelder-Mead
    # Take the best point found so far and polish it.
    if best_pts is not None and _elapsed_ok(20):
        # Convert best_pts to barycentric for optimization
        best_uv = _cart_to_bary(best_pts).flatten()
        
        # Run Nelder-Mead
        # Nelder-Mead is local, so start from the DE result.
        try:
            nm_result = minimize(
                _objective_bary,
                best_uv,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            _update_best(nm_result.x)
        except Exception as e:
            print(f"NM failed: {e}")

    # Phase 3: Multi-start Nelder-Mead from DE population?
    # If time remains, we can try polishing other points from the DE population.
    # However, DE result is usually the best. 
    # Let's try a few random starts for NM if time allows.
    
    if _elapsed_ok(30):
        # Try 5 random starts for NM
        for i in range(5):
            if not _elapsed_ok(10):
                break
            uv_rand = rng.random(2 * N)
            uv_rand = _project_to_simplex(uv_rand)
            try:
                nm_res = minimize(
                    _objective_bary,
                    uv_rand,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                _update_best(nm_res.x)
            except Exception:
                pass

    if best_pts is None:
        # Fallback to initial random
        uv_rand = rng.random(2 * N)
        best_pts = _bary_to_cart(_project_to_simplex(uv_rand))
        
    return best_pts

# EVOLVE-BLOCK-END
