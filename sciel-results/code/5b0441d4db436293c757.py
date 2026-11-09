# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0.
    """
    n = len(vertices)
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    other_dists = dists[1:]
    d = np.sort(other_dists)
    
    if d[0] < 0.05:
        return 0.0
    
    gaps = d[1:] - d[:-1]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
        
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    # Check no repeated points / zero length edges
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 0.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    # Check all turns are same sign and nonzero
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
        
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _make_polygon_from_radii(radii, angles=None):
    """
    Create a polygon from radial perturbations.
    radii: (20,) array of radii.
    Returns (20, 2) array or None if invalid.
    """
    n = 20
    if angles is None:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Ensure radii are positive
    if np.any(radii <= 0):
        return None
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    poly = np.stack([x, y], axis=1)
    
    # Quick convexity check
    # For a star-shaped polygon with vertices in angular order, 
    # convexity is not guaranteed just by positive radii.
    # We must check the cross products of consecutive edges.
    edges = np.diff(np.vstack([poly, poly[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    
    # All cross products must have the same sign and be non-zero
    if np.any(np.abs(cross) < 1e-10):
        return None
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return None
        
    return poly

def _objective(radii):
    """
    Objective function for Differential Evolution.
    We want to maximize the minimum vertex score.
    DE minimizes, so we return -score.
    We add a penalty if the polygon is invalid.
    """
    poly = _make_polygon_from_radii(radii)
    if poly is None:
        return 1e6 # Large penalty for invalid polygon
    
    score = _polygon_min_score(poly)
    return -score

def _search_radial_de():
    """
    Search over radial modulations using Differential Evolution.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Bounds for radii. 
    # We start with regular polygon radii = 1.
    # To maintain convexity, radii cannot vary too wildly.
    # Let's try a range [0.5, 1.5] initially, maybe wider if needed.
    # Actually, for a regular 20-gon, the ratio of max to min radius 
    # that maintains convexity is constrained.
    # If r_i varies too much, the polygon becomes non-convex.
    # Let's use bounds [0.8, 1.2] to be safe and effective.
    bounds = [(0.8, 1.2) for _ in range(n)]
    
    # Differential Evolution
    # maxiter and popsize need to be balanced with time limit.
    # Each function evaluation is O(n^2) ~ 400 ops, very fast.
    # 1000 iterations * 50 popsize = 50k evaluations. Should be fast enough.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-5,
            mutation=(0.4, 1.0),
            recombination=0.8,
            polish=True, # Use L-BFGS-B at the end
            init='sobol',
            maxfev=5000, # Limit function evaluations to ensure we finish in time
            x0=np.ones(n) # Start near regular
        )
        
        if result.success and result.fun < 0:
            best_radii = result.x
            best_score = -result.fun
        else:
            # Fallback to regular if DE fails or finds nothing better
            best_radii = np.ones(n)
            best_score = _polygon_min_score(_make_polygon_from_radii(best_radii))
            
    except Exception as e:
        # If DE fails, try a simple random search or fallback
        best_radii = np.ones(n)
        best_score = _polygon_min_score(_make_polygon_from_radii(best_radii))

    # If time permits, do a local refinement with Nelder-Mead on the best result
    if time.time() < deadline - 30:
        from scipy.optimize import minimize
        
        def objective_nm(radii):
            poly = _make_polygon_from_radii(radii)
            if poly is None:
                return 1e6
            return -_polygon_min_score(poly)
            
        try:
            res_nm = minimize(objective_nm, best_radii, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_nm.fun < -best_score:
                best_radii = res_nm.x
                best_score = -res_nm.fun
        except Exception:
            pass

    # Final check and return
    poly = _make_polygon_from_radii(best_radii)
    if poly is None:
        # Fallback to regular
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
    return poly

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    
    polygon = _search_radial_de()
    
    if time.time() - start_time > 625:
        n = 20
        radius = 1.0
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
    return polygon

# EVOLVE-BLOCK-END
