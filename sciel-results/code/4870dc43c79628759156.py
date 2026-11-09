import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 600.0

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    # Vectorized computation of all triangle areas
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """Compute A_min / A_hull."""
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _soft_min_area(points, temperature=1.0):
    """
    Differentiable approximation of the minimum triangle area.
    Uses log-sum-exp trick for numerical stability.
    """
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    
    # Calculate areas (2 * area to avoid division by 2 in log space initially, 
    # but standard area is fine. We'll use the formula directly)
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    
    # Soft min: -T * log(sum(exp(-area/T)))
    # To avoid underflow/overflow, subtract max (which is min of -areas)
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -temperature * log_sum_exp
    
    return soft_min

def _objective_interior(x_int, hull_pts, temperature=1.0):
    """
    Objective function to minimize: - (soft_min_area / hull_area)
    x_int: flattened interior points
    hull_pts: fixed hull points
    """
    n_int = len(x_int) // 2
    pts_int = x_int.reshape(n_int, 2)
    pts = np.vstack([hull_pts, pts_int])
    
    ha = _hull_area(pts)
    if ha <= 0:
        return 1e10
    
    sm = _soft_min_area(pts, temperature)
    
    # We want to maximize sm / ha, so minimize -sm/ha
    return -sm / ha

def _optimize_interior(hull_pts, init_int, temperature=1.0):
    """
    Optimize interior points given fixed hull points.
    Uses L-BFGS-B with gradient estimation or direct method if gradient is tricky.
    Since _soft_min_area is differentiable, we could use gradient, but 
    scipy's minimize with L-BFGS-B can use numerical gradient if jac is not provided.
    However, numerical gradient on 26 dims might be slow. 
    Let's stick to Powell or Nelder-Mead for robustness if gradient is not provided, 
    OR provide a simple gradient.
    
    Actually, let's use a differentiable approach with L-BFGS-B and let it compute gradients numerically 
    but limit iterations. Or better, use COBYLA or Powell which don't need gradients.
    Given the low dimension (interior points only), Powell is often very effective.
    """
    n_int = len(init_int)
    bounds = [(0.01, 0.99)] * n_int
    
    best_x = init_int.copy()
    best_val = _objective_interior(best_x, hull_pts, temperature)
    
    methods = ['Powell', 'Nelder-Mead', 'COBYLA']
    
    for method in methods:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_objective_interior, best_x, args=(hull_pts, temperature),
                           method=method, bounds=bounds,
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    return best_x

def _generate_regular_hull(n_hull, radius=0.45, center=(0.5, 0.5)):
    """Generate a regular polygon hull."""
    angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    # Rotate slightly to vary orientation
    angles += np.pi / n_hull 
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _generate_interior_init(n_int, hull_pts, rng):
    """Generate initial interior points inside the hull."""
    # Simple random points within the bounding box of the hull, then check if inside
    # For simplicity, use a smaller circle
    cx = np.mean(hull_pts[:, 0])
    cy = np.mean(hull_pts[:, 1])
    r = 0.25
    angles = rng.uniform(0, 2*np.pi, n_int)
    radii = rng.uniform(0, r, n_int)
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)
    return np.column_stack([x, y])

def heilbronn_convex13() -> np.ndarray:
    n_total = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    # Strategy: Vary the number of hull points (k) and optimize the rest (n-k)
    # k=13: Regular 13-gon
    # k=12: 12-gon + 1 center
    # k=11: 11-gon + 2 interior
    # ... down to k=5
    
    possible_hull_sizes = [13, 12, 11, 10, 9, 8, 7, 6, 5]
    
    for k in possible_hull_sizes:
        if time.time() > _DEADLINE - 20:
            break
            
        n_int = n_total - k
        
        # Try a few different hull radii and orientations
        for trial in range(5):
            if time.time() > _DEADLINE - 20:
                break
            
            # Vary radius slightly
            radius = 0.45 + rng.uniform(-0.05, 0.05)
            # Vary rotation
            base_hull = _generate_regular_hull(k, radius=radius)
            
            if n_int == 0:
                pts = base_hull
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            else:
                # Generate initial interior points
                int_init = _generate_interior_init(n_int, base_hull, rng)
                int_flat = int_init.flatten()
                
                # Optimize interior points
                # Use a soft-min temperature that starts high and cools? 
                # For simplicity, fix temperature to a value that approximates min well.
                # Temperature too high = average, too low = min. 
                # Let's try a few temperatures.
                
                for temp in [0.05, 0.1, 0.2]:
                    if time.time() > _DEADLINE - 10:
                        break
                    
                    opt_int_flat = _optimize_interior(base_hull, int_init.copy().flatten(), temperature=temp)
                    opt_int = opt_int_flat.reshape(n_int, 2)
                    
                    pts = np.vstack([base_hull, opt_int])
                    
                    # Validate
                    if not np.all(np.isfinite(pts)):
                        continue
                    
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
                        
    # Also try a fully random search with local optimization as a fallback/complement
    # This ensures we don't miss asymmetric solutions
    if time.time() < _DEADLINE - 30:
        for trial in range(20):
            if time.time() > _DEADLINE - 10:
                break
            pts = rng.random((n_total, 2)) * 0.9 + 0.05
            # Local optimization on all points
            # We can't easily use the interior-only optimizer here.
            # Let's just do a quick Powell on all 26 coords with a soft-min objective.
            
            def obj_all(x, temp=0.1):
                p = x.reshape(n_total, 2)
                ha = _hull_area(p)
                if ha <= 0: return 1e10
                sm = _soft_min_area(p, temp)
                return -sm/ha
            
            try:
                res = minimize(obj_all, pts.flatten(), method='Powell', 
                               options={'maxiter': 100, 'xatol': 1e-5})
                if res.fun < 1e9:
                    p_opt = res.x.reshape(n_total, 2)
                    s = _score(p_opt)
                    if s > best_score:
                        best_score = s
                        best_points = p_opt.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback: regular 13-gon
        best_points = _generate_regular_hull(13)
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = _generate_regular_hull(13)
        
    return best_points

# EVOLVE-BLOCK-END
