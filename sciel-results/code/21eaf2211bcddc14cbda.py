"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 115  # Safety margin from 120s limit

def _bary_to_cart(abc):
    """Convert (a,b,c) barycentric coords to (x,y) cartesian points."""
    # x = b + c*0.5, y = c*S3H
    pts = np.zeros((len(abc), 2))
    pts[:, 0] = abc[:, 1] + 0.5 * abc[:, 2]
    pts[:, 1] = abc[:, 2] * S3H
    return pts

def _cart_to_bary(pts):
    """Convert (x,y) cartesian points to (a,b,c) barycentric coords."""
    n = len(pts)
    abc = np.zeros((n, 3))
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    abc[:, 0] = a
    abc[:, 1] = b
    abc[:, 2] = c
    return abc

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
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

def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    # Shift to prevent underflow/overflow in exp
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip shifted values to avoid log(0) if sum is 0 (shouldn't happen with max shift)
    # exp(shifted) will be <= 1.
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _objective_bary_soft(x, beta):
    """
    Objective function for L-BFGS-B.
    x: flattened barycentric coords (3*N)
    beta: temperature for soft min
    """
    abc = x.reshape(-1, 3)
    # Project to valid barycentric space (a,b,c >= 0, sum=1)
    abc = np.clip(abc, 0, 1)
    sums = abc.sum(axis=1, keepdims=True)
    # Handle case where all are 0 (unlikely but safe)
    sums[sums == 0] = 1
    abc = abc / sums
    
    pts = _bary_to_cart(abc)
    return -_soft_min_area(pts, beta)

def _objective_bary_hard(x):
    """
    Objective function for final polishing (non-smooth but exact).
    """
    abc = x.reshape(-1, 3)
    abc = np.clip(abc, 0, 1)
    sums = abc.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    abc = abc / sums
    
    pts = _bary_to_cart(abc)
    return -_min_triangle_area(pts)

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        # Ensure points are strictly inside/on boundary
        # Projection to triangle in Cartesian
        pts = np.array(pts, dtype=float)
        for i in range(len(pts)):
            x, y = pts[i]
            y = max(0.0, min(S3H, y))
            x_min = y / S3
            x_max = 1 - y / S3
            x = max(x_min, min(x_max, x))
            pts[i] = [x, y]
        
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # 1. Generate Initial Candidates
    candidates = []
    
    # Candidate 1: Symmetric "3-4-3" or similar heuristic
    # Corners, midpoints, centroid
    init_sym = np.array([
        [0.0, 0.0],          # 0
        [1.0, 0.0],          # 1
        [0.5, S3H],          # 2
        [0.5, 0.0],          # 3: mid bottom
        [0.75, S3H / 2],     # 4: mid right
        [0.25, S3H / 2],     # 5: mid left
        [1/3, S3H/3],        # 6: centroid
        [0.2, 0.2],          # 7
        [0.8, 0.2],          # 8
        [0.5, 0.4],          # 9
        [0.5, 0.1],          # 10
    ])
    candidates.append(init_sym)
    
    # Candidate 2: Random
    for _ in range(3):
        # Random points in triangle
        r1 = rng.random(N)
        r2 = rng.random(N)
        # Transform to triangle
        # If r1+r2 > 1, reflect
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        # Barycentric: a=1-r1-r2, b=r1, c=r2
        abc = np.column_stack([1-r1-r2, r1, r2])
        pts = _bary_to_cart(abc)
        candidates.append(pts)

    # Candidate 3: Grid-like
    # Try to spread points out
    grid_x = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid_y = [0.0, S3H/2, S3H]
    grid_pts = []
    for y in grid_y:
        x_min = y / S3
        x_max = 1 - y / S3
        if x_max < x_min: continue
        # Sample x
        for x in grid_x:
            if x_min <= x <= x_max:
                grid_pts.append([x, y])
    # Pick 11 from grid
    if len(grid_pts) >= N:
        # Simple selection: take first N
        candidates.append(np.array(grid_pts[:N]))

    # Evaluate initial candidates
    for c in candidates:
        if not _elapsed_ok(5): break
        _update_best(c)

    if best_pts is None:
        # Fallback
        return candidates[0]

    # 2. Global Search: Differential Evolution
    # We optimize in barycentric space (2 independent coords per point, a=1-b-c)
    # Bounds: b in [0,1], c in [0,1], a=1-b-c >= 0 => b+c <= 1
    # DE works with box constraints. We use b, c and project.
    
    def _de_obj(flat_bc):
        bc = flat_bc.reshape(-1, 2)
        b = bc[:, 0]
        c = bc[:, 1]
        a = 1.0 - b - c
        abc = np.column_stack([a, b, c])
        # Project to valid barycentric
        abc = np.clip(abc, 0, 1)
        sums = abc.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        abc = abc / sums
        pts = _bary_to_cart(abc)
        ma = _min_triangle_area(pts)
        return -ma

    bounds_de = [(0, 1), (0, 1)] * N
    
    if _elapsed_ok(30):
        try:
            de_result = differential_evolution(
                _de_obj,
                bounds_de,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if de_result.fun < 0:
                bc = de_result.x.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                abc = np.clip(abc, 0, 1)
                sums = abc.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc = abc / sums
                de_pts = _bary_to_cart(abc)
                _update_best(de_pts)
        except Exception:
            pass

    # 3. Local Refinement: Multi-start L-BFGS-B with Soft Min Annealing
    # Start from best found so far, plus some random perturbations
    
    base_pts = best_pts.copy()
    base_abc = _cart_to_bary(base_pts)
    # Normalize base_abc
    base_abc = np.clip(base_abc, 0, 1)
    sums = base_abc.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    base_abc = base_abc / sums

    # Prepare list of starting points for local search
    starts = [base_abc.flatten()]
    for _ in range(5):
        # Perturb best
        noise = rng.normal(0, 0.05, size=(N, 3))
        pert_abc = base_abc + noise
        pert_abc = np.clip(pert_abc, 0, 1)
        sums = pert_abc.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        pert_abc = pert_abc / sums
        starts.append(pert_abc.flatten())
        
    # Also add starts from other candidates if they were good
    for c in candidates:
        abc_c = _cart_to_bary(c)
        abc_c = np.clip(abc_c, 0, 1)
        sums = abc_c.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        abc_c = abc_c / sums
        starts.append(abc_c.flatten())

    # Annealing schedule for beta
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for x0 in starts:
        if not _elapsed_ok(10):
            break
            
        x_curr = x0.copy()
        
        for beta in betas:
            if not _elapsed_ok(5):
                break
                
            try:
                res = minimize(
                    _objective_bary_soft,
                    x_curr,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x_curr = res.x
                
                # Update best with current result
                abc_res = x_curr.reshape(-1, 3)
                abc_res = np.clip(abc_res, 0, 1)
                sums = abc_res.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc_res = abc_res / sums
                pts_res = _bary_to_cart(abc_res)
                _update_best(pts_res)
                
            except Exception:
                break

    # 4. Final Polish: Nelder-Mead on Hard Objective
    # This handles the non-smoothness of the true min-area function
    if best_pts is not None and _elapsed_ok(15):
        best_abc = _cart_to_bary(best_pts)
        best_abc = np.clip(best_abc, 0, 1)
        sums = best_abc.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        best_abc = best_abc / sums
        
        # Run a few NM restarts
        for i in range(5):
            if not _elapsed_ok(5):
                break
            x0 = best_abc.flatten() + rng.normal(0, 0.01, size=(N, 3)).flatten()
            try:
                res = minimize(
                    _objective_bary_hard,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                abc_res = res.x.reshape(-1, 3)
                abc_res = np.clip(abc_res, 0, 1)
                sums = abc_res.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc_res = abc_res / sums
                pts_res = _bary_to_cart(abc_res)
                _update_best(pts_res)
            except Exception:
                pass

    if best_pts is None:
        return candidates[0]

    return best_pts
