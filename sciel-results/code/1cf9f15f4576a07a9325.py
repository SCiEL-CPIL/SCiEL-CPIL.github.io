"""Heilbronn convex regions n=14: maximize minimum triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas_vec(pts):
    """
    Vectorized computation of areas of all triangles formed by triples of points.
    pts: (N, 2)
    Returns: array of shape (N*(N-1)*(N-2)/6,)
    """
    n = len(pts)
    # We need combinations i < j < k
    # Broadcasting approach:
    # pts[i, :] - pts[j, :] and pts[i, :] - pts[k, :]
    # This creates large intermediate arrays. For N=14, it's manageable.
    
    # Efficient way:
    # i indices: 0..N-1
    # j indices: i+1..N-1
    # k indices: j+1..N-1
    
    # Let's just use a loop for clarity and memory safety, N=14 is small.
    # But vectorization can help if we do it in chunks or use itertools.
    # For N=14, total triangles = 14*13*12/6 = 364.
    # A loop is fast enough. But let's try to be slightly more efficient with numpy.
    
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                # Cross product magnitude in 2D
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas[idx] = area
                idx += 1
    return areas

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the point set."""
    try:
        # Check for degenerate cases (all points collinear or same)
        # ConvexHull requires at least 3 non-collinear points
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas_vec(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min(areas, temperature):
    """
    Compute the soft minimum of an array.
    soft_min(a, T) = -1/T * log(sum(exp(-T * a)))
    """
    # Numerical stability
    min_a = np.min(areas)
    # If T is large, exp(-T * a) can underflow.
    # We can shift by min_a
    # sum(exp(-T * a)) = exp(-T * min_a) * sum(exp(-T * (a - min_a)))
    # log(sum(exp(-T * a))) = -T * min_a + log(sum(exp(-T * (a - min_a))))
    # soft_min = -1/T * ( -T * min_a + log(sum(exp(-T * (a - min_a)))) )
    #          = min_a - 1/T * log(sum(exp(-T * (a - min_a))))
    
    shifted = areas - min_a
    # shifted >= 0
    # exp(-T * shifted) <= 1
    log_sum = np.log(np.sum(np.exp(-temperature * shifted)))
    return min_a - (1.0 / temperature) * log_sum

def _objective_smooth(pts_flat, temperature, hull_area_cache=None):
    """
    Objective function to minimize: -soft_min(areas) / hull_area
    We want to maximize the ratio, so we minimize the negative.
    """
    pts = pts_flat.reshape(-1, 2)
    areas = _triangle_areas_vec(pts)
    
    # Calculate hull area
    # To avoid recomputing hull area too often or handling non-differentiability,
    # we compute it here.
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10 # Large penalty
    
    sm = _soft_min(areas, temperature)
    # We want to maximize sm / hull_area
    # So we minimize -sm / hull_area
    return -sm / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for Python overhead
    
    best_pts = None
    best_score = -1.0
    
    # Initializations
    initializations = []
    
    # 1. Regular Polygon
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(theta), np.sin(theta)])
    initializations.append(init_reg)
    
    # 2. Random Convex Sets
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Generate random angles, sort them to ensure convexity (points on convex hull)
        # This creates points on the boundary of their convex hull
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        # Add some radius variation to avoid being on a circle
        # Actually, for Heilbronn, interior points are often not optimal for small N?
        # For N=14, it's likely all points are on the hull.
        # Let's generate points on a circle with jittered radii
        radii = rng.uniform(0.5, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)
        
    # 3. Jittered Grid
    # Place points on a grid and jitter
    # 14 points... maybe 4x4 grid minus 2?
    # Let's just use random points in a square, then take convex hull?
    # No, we need 14 points.
    for _ in range(10):
        pts = rng.uniform(-1, 1, (N, 2))
        # Ensure they are somewhat spread out
        # Normalize to unit area? Not necessary for init
        initializations.append(pts)

    # Optimization Loop
    # We will use a two-stage approach:
    # Stage 1: Moderate temperature (e.g., 10.0) to find a good basin
    # Stage 2: High temperature (e.g., 100.0) to sharpen the minimum
    
    temperatures = [10.0, 50.0, 200.0]
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
            
        current_pts = init_pts.copy()
        
        # Evaluate initial score
        current_score = _score(current_pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
        # Try to optimize from this initialization
        # We'll try the first temperature for all, and subsequent temperatures for the best ones
        # Or just run a short optimization for each T.
        
        for T in temperatures:
            if time.time() > deadline:
                break
                
            x0 = current_pts.flatten()
            
            # Define objective for this T
            def obj(x, T=T):
                return _objective_smooth(x, T)
                
            try:
                # Use L-BFGS-B
                # maxiter depends on remaining time. 
                # Let's cap it to avoid long runs.
                max_iter = 100
                result = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
                    
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
                    
            except Exception as e:
                pass
    
    # Final Polishing: 
    # Take the best point found and do a few more steps with very high temperature
    # to ensure we are at the local minimum of the hard min function.
    if best_pts is not None and time.time() < deadline:
        T_final = 1000.0
        x0 = best_pts.flatten()
        
        def obj_final(x):
            return _objective_smooth(x, T_final)
            
        try:
            result = minimize(
                obj_final,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            final_pts = result.x.reshape(-1, 2)
            final_score = _score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass

    # If we still have time, try some random perturbations of the best point
    if best_pts is not None and time.time() < deadline:
        rng_polish = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > deadline:
                break
            # Perturb
            noise = rng_polish.normal(0, 0.01, best_pts.shape)
            perturbed_pts = best_pts + noise
            perturbed_score = _score(perturbed_pts)
            
            if perturbed_score > best_score:
                best_score = perturbed_score
                best_pts = perturbed_pts.copy()
                
            # Quick local optimization from perturbed point
            T_polish = 50.0
            x0 = perturbed_pts.flatten()
            def obj_polish(x):
                return _objective_smooth(x, T_polish)
            try:
                result = minimize(
                    obj_polish,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10}
                )
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            except Exception:
                pass

    if best_pts is None:
        # Fallback
        best_pts = initializations[0]
        
    return best_pts

# EVOLVE-BLOCK-END
