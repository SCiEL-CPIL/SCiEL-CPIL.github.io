import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def bary_to_cart(bary):
    """
    bary: (N, 2) array of (u, v). w is implicit as 1-u-v.
    Returns (N, 2) Cartesian coordinates.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = H * w
    return np.column_stack((x, y))

def _calc_areas_bary(bary_pts):
    """
    Calculate areas of all triangles formed by N points in barycentric coords.
    Returns array of shape (165,) with area values.
    Note: In barycentric coords, the area of a triangle formed by 3 points is 
    proportional to the absolute value of the determinant of their barycentric coordinates.
    Specifically, Area = |det| * Area_Triangle / 2? 
    Actually, the determinant of the 3x3 matrix of barycentric coordinates gives the ratio of the area 
    of the sub-triangle to the area of the main triangle.
    Let's verify:
    If points are vertices, det is 1. Area is 1 (relative to main triangle area).
    So relative_area = |det|.
    Since we are maximizing the minimum relative area, we can just use |det|.
    The evaluator likely computes Cartesian area. 
    Area_cart = 0.5 * |cross|.
    For equilateral triangle of side 1, Area_total = sqrt(3)/4.
    If we use barycentric det, Area_sub = det * Area_total.
    So maximizing min(Area_sub) is equivalent to maximizing min(det).
    We will calculate the actual Cartesian areas to be safe and consistent with the evaluator's metric.
    """
    pts = bary_to_cart(bary_pts)
    Pi = pts[I]
    Pj = pts[J]
    Pk = pts[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _min_area_bary(bary_pts):
    """
    Calculate min triangle area given points in barycentric coords (u, v).
    """
    areas = _calc_areas_bary(bary_pts)
    return np.min(areas)

def _soft_min_area_bary(bary_pts, beta=100.0):
    """
    Smooth approximation of min area using log-sum-exp.
    """
    areas = _calc_areas_bary(bary_pts)
    # Log-sum-exp trick for stability
    min_area = -np.log(np.sum(np.exp(-beta * areas))) / beta
    return min_area

def _project_bary(bary):
    """
    Project a point (u, v) into the valid barycentric simplex (u>=0, v>=0, u+v<=1).
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    # Ensure strict positivity if needed for some optimizers, but SLSQP/L-BFGS-B handle bounds.
    # For Nelder-Mead, we might want to keep them strictly inside to avoid numerical issues with area calc?
    # Area calc is fine on boundary.
    return np.column_stack((u, v))

def _generate_hex_init():
    """
    Generate a hexagonal/triangular lattice initialization in barycentric coordinates.
    """
    # A standard 11-point configuration often starts near a triangular lattice.
    # We'll create a few candidates and pick the best.
    candidates = []
    
    # Candidate 1: Regular grid
    # Rows in (u,v) space. Note: v is "height" in the (u,v) plane corresponding to y-axis in cart?
    # No, u, v are barycentric weights.
    # Let's just use a simple random-ish grid or known good start.
    # Known good start for 11 points is often close to a 4-3-2-1 or 3-4-3-1 arrangement.
    
    # Let's try a few manual placements
    # 1. Center + Ring
    # 2. Grid
    
    # Grid approach in barycentric (u, v) where u+v<=1
    # Let's try a 4-3-2-1 grid
    pts = []
    # Row 0: v=0.1, u in [0.1, 0.3, 0.5, 0.7]
    for u in [0.1, 0.3, 0.5, 0.7]:
        if u + 0.1 <= 1.0:
            pts.append([u, 0.1])
            
    # Row 1: v=0.3, u in [0.15, 0.35, 0.55]
    for u in [0.15, 0.35, 0.55]:
        if u + 0.3 <= 1.0:
            pts.append([u, 0.3])
            
    # Row 2: v=0.5, u in [0.25, 0.45]
    for u in [0.25, 0.45]:
        if u + 0.5 <= 1.0:
            pts.append([u, 0.5])
            
    # Row 3: v=0.7, u in [0.35]
    for u in [0.35]:
        if u + 0.7 <= 1.0:
            pts.append([u, 0.7])
            
    if len(pts) == 11:
        candidates.append(np.array(pts))
        
    # Candidate 2: Perturbed random
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        candidates.append(np.column_stack([u, v]))
        
    best_candidate = None
    best_score = -np.inf
    for c in candidates:
        c = _project_bary(c)
        score = _min_area_bary(c)
        if score > best_score:
            best_score = score
            best_candidate = c
            
    return best_candidate

def _refine_two_stage(bary_init, time_limit, rng):
    """
    Refine using two stages:
    1. L-BFGS-B on soft-min surrogate.
    2. Nelder-Mead or SLSQP on hard min.
    """
    start_time = time.time()
    best_bary = bary_init.copy()
    best_score = _min_area_bary(best_bary)
    
    bounds = [(0, 1)] * (2 * N)
    
    # Stage 1: Smooth optimization
    def soft_obj(flat_bary):
        b = _project_bary(flat_bary.reshape(-1, 2))
        # Add penalty for being far outside? _project_bary handles it.
        # But L-BFGS-B might step outside. The projection is non-differentiable.
        # Better to use bounds and let the function handle interior.
        # However, _project_bary introduces kinks.
        # Alternative: Use a penalty method or just rely on the fact that the minimum is likely inside.
        # Let's try a simple approach: minimize -soft_min_area.
        # If points go outside, areas might be weird? No, bary_to_cart is linear.
        # But if u+v > 1, the point is outside the triangle.
        # We must ensure validity.
        # Let's use a penalty in the objective for violation.
        u = flat_bary.reshape(-1, 2)[:, 0]
        v = flat_bary.reshape(-1, 2)[:, 1]
        viol = np.sum(np.maximum(0, u + v - 1.0)**2)
        area = _soft_min_area_bary(flat_bary.reshape(-1, 2), beta=100.0)
        return -area + 1000.0 * viol

    try:
        res1 = minimize(
            soft_obj,
            best_bary.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res1.success or res1.fun < soft_obj(best_bary.flatten()):
            new_bary = _project_bary(res1.x.reshape(-1, 2))
            new_score = _min_area_bary(new_bary)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
    except Exception:
        pass
    
    if time.time() - start_time > time_limit - 2:
        return best_bary, best_score
        
    # Stage 2: Hard optimization
    def hard_obj(flat_bary):
        b = _project_bary(flat_bary.reshape(-1, 2))
        u = b[:, 0]
        v = b[:, 1]
        viol = np.sum(np.maximum(0, u + v - 1.0)**2)
        area = _min_area_bary(b)
        return -area + 1000.0 * viol

    # Try Nelder-Mead for non-smooth hard min
    try:
        res2 = minimize(
            hard_obj,
            best_bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if res2.fun < hard_obj(best_bary.flatten()):
            new_bary = _project_bary(res2.x.reshape(-1, 2))
            new_score = _min_area_bary(new_bary)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
    except Exception:
        pass
        
    # Try SLSQP as well if time permits
    if time.time() - start_time < time_limit - 0.5:
        try:
            res3 = minimize(
                hard_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res3.fun < hard_obj(best_bary.flatten()):
                new_bary = _project_bary(res3.x.reshape(-1, 2))
                new_score = _min_area_bary(new_bary)
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary
        except Exception:
            pass
            
    return best_bary, best_score

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -np.inf
    
    # Phase 1: Global Search with Differential Evolution
    # DE is good for finding a good basin.
    de_time_budget = 40.0
    
    def de_obj(x):
        bary = _project_bary(x.reshape(N, 2))
        return -_min_area_bary(bary)
        
    bounds = [(0, 1)] * (2 * N)
    
    try:
        # Use a small population and few iterations to stay within time
        # popsize=10, maxiter=20 should be fast enough
        result_de = differential_evolution(
            de_obj,
            bounds,
            seed=42,
            maxiter=20,
            popsize=10,
            tol=1e-5,
            polish=False,
            init='sobol'
        )
        
        if result_de.success or result_de.fun < de_obj(best_bary.flatten() if best_bary is not None else np.random.random(2*N)):
            de_bary = _project_bary(result_de.x.reshape(N, 2))
            de_score = _min_area_bary(de_bary)
            if best_bary is None or de_score > best_score:
                best_score = de_score
                best_bary = de_bary
    except Exception:
        pass
    
    # Phase 2: Local Refinement of DE result and Hex Init
    # We have ~100s left.
    # Let's refine the DE result.
    if best_bary is not None:
        time_left = timeout - (time.time() - start_time)
        if time_left > 10:
            refined_bary, refined_score = _refine_two_stage(best_bary, time_left - 5, rng)
            if refined_score > best_score:
                best_score = refined_score
                best_bary = refined_bary
                
    # Phase 3: Multi-start Local Refinement
    # Try a few random starts and the hex init
    init_strategies = [
        _generate_hex_init,
        lambda: _project_bary(np.column_stack([rng.random(N), rng.random(N) * (1 - rng.random(N))])),
        lambda: _project_bary(np.column_stack([rng.random(N), rng.random(N) * (1 - rng.random(N))])),
        lambda: _project_bary(np.column_stack([rng.random(N), rng.random(N) * (1 - rng.random(N))])),
    ]
    
    for i in range(len(init_strategies)):
        if time.time() - start_time > timeout - 15:
            break
            
        init_pts = init_strategies[i]()
        time_left = timeout - 15 - (time.time() - start_time)
        if time_left < 5:
            continue
            
        refined_bary, refined_score = _refine_two_stage(init_pts, time_left, rng)
        if refined_score > best_score:
            best_score = refined_score
            best_bary = refined_bary
            
    # Final Polish
    if best_bary is not None and time.time() - start_time < timeout - 2:
        # One last high-precision SLSQP run
        try:
            def final_obj(flat_bary):
                b = _project_bary(flat_bary.reshape(-1, 2))
                return -_min_area_bary(b)
                
            res_final = minimize(
                final_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            if res_final.fun < final_obj(best_bary.flatten()):
                new_bary = _project_bary(res_final.x.reshape(-1, 2))
                new_score = _min_area_bary(new_bary)
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary
        except Exception:
            pass

    # Convert to Cartesian
    if best_bary is None:
        best_bary = _generate_hex_init()
        
    cart_pts = bary_to_cart(best_bary)
    
    # Final validation and clipping
    x = cart_pts[:, 0]
    y = cart_pts[:, 1]
    
    # Clip to triangle
    y = np.maximum(y, 0)
    viol1 = y > np.sqrt(3) * x
    if np.any(viol1):
        x[viol1] = y[viol1] / np.sqrt(3)
    viol2 = y > np.sqrt(3) * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / np.sqrt(3)
        
    cart_pts[:, 0] = x
    cart_pts[:, 1] = y
    
    return cart_pts
