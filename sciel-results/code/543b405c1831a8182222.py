"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from scipy.optimize import minimize, check_grad

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points, 
    normalized by the area of the container triangle.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    min_area = float('inf')
    # Vectorized calculation for speed
    # points shape (n, 2)
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    # We can compute all areas for a fixed i by vectorizing j and k
    
    # Precompute all pairwise differences? No, n is small (11), so O(n^3) is fine but vectorized O(n^3) is better.
    # Let's stick to the loop over i with vectorized j,k as in the original, it's efficient enough.
    
    for i in range(n):
        pi = points[i]
        # Vectors from pi to all other points
        vecs = points - pi  # shape (n, 2)
        
        X = vecs[:, 0]
        Y = vecs[:, 1]
        
        # Cross product matrix C[j,k] = X[j]*Y[k] - Y[j]*X[k]
        C = np.outer(X, Y) - np.outer(Y, X)
        
        C_abs = np.abs(C)
        
        # We need min of 0.5 * |C[j,k]| for j < k, j != i, k != i
        # Set diagonal and i-th row/col to inf
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        np.fill_diagonal(C_abs, float('inf'))
        
        # Find min in the remaining part
        # Since C is antisymmetric, C[j,k] = -C[k,j], so |C| is symmetric.
        # We can just take the global min of the modified matrix.
        current_min = np.min(C_abs)
        
        if current_min < min_area:
            min_area = current_min
            if min_area < 1e-12: # Early exit if degenerate
                return 0.0

    return (0.5 * min_area) / AREA


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _clamp_points(points):
    """
    Clamp points to be inside the triangle.
    This is a simple heuristic projection. If a point is outside, 
    we pull it back towards the centroid or the nearest edge.
    For robustness in direct search, we can just reject invalid moves 
    in the optimizer, but clamping helps with stability.
    """
    clamped = np.copy(points)
    for i in range(len(points)):
        p = points[i]
        if not _is_inside_triangle(p):
            # Simple projection: find the closest point on the triangle boundary?
            # Or just move it towards the center?
            # Moving towards center is safer for feasibility.
            center = (V0 + V1 + V2) / 3.0
            # We can do a binary search for t such that p + t*(center-p) is inside
            # Or just use a fixed small step towards center if outside
            # Actually, for Nelder-Mead, it's better to have a penalty or rejection.
            # Let's use a penalty function approach in the objective instead of hard clamping
            # which might distort the geometry. 
            # However, if we use a penalty, the "min area" might be computed on invalid points.
            # Better: define the objective to return a large negative value (or small area) if invalid.
            # But _min_triangle_area assumes valid points.
            # Let's implement a soft penalty in the objective function wrapper.
            pass
    return clamped


def _objective_with_penalty(params):
    """
    Objective function for minimization (we maximize area, so we minimize negative area).
    Includes a penalty for points outside the triangle.
    """
    # Convert params to points
    # We use a simple mapping: params are (x, y) directly? 
    # No, the interface expects construct() to return points.
    # The optimization variables are the coordinates of the points.
    # We need to ensure they stay inside. 
    # Using a penalty method is robust for direct search.
    
    points = params.reshape(-1, 2)
    
    # Check validity
    penalty = 0.0
    for p in points:
        if not _is_inside_triangle(p):
            # Calculate distance to triangle
            # Simple approximation: distance to nearest vertex or edge
            # Or just a large penalty
            penalty += 10.0
            # Add distance to center to guide back?
            # Let's just use a large constant penalty for simplicity in direct search.
            # Nelder-Mead will likely reject this move.
            
    if penalty > 0:
        return -100.0 # Very bad score (minimizing negative area, so -100 is very high cost)
    
    area = _min_triangle_area(points)
    return -area


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    # Vertices, midpoints, centroid, etc.
    # A good starting point is often a grid or specific symmetric layout.
    # For 11 points, a 3x3 grid plus center? Or vertices + midpoints + centroid + 4 more?
    # Let's use a simple heuristic: place points at vertices, midpoints, centroid, and 4 points 
    # slightly offset from centroid or edges.
    
    # Vertices
    pts = [V0 + np.array([0.01, 0.01]), V1 + np.array([-0.01, 0.01]), V2 + np.array([0.0, -0.01])]
    # Midpoints
    pts.append((V0 + V1) / 2.0)
    pts.append((V1 + V2) / 2.0)
    pts.append((V2 + V0) / 2.0)
    # Centroid
    pts.append((V0 + V1 + V2) / 3.0)
    
    # 4 more points to make 11
    # Maybe near the 1/3 points of the medians?
    # Or just random jitter around the centroid?
    # Let's add 4 points in a small cluster around the centroid, slightly spread out.
    for i in range(4):
        angle = i * np.pi / 2
        r = 0.1
        p = pts[6] + np.array([r * np.cos(angle), r * np.sin(angle)])
        # Ensure inside
        if _is_inside_triangle(p):
            pts.append(p)
        else:
            pts.append(pts[6]) # Fallback to centroid
    
    return np.array(pts)


def _nelder_mead_optimize(points, max_iter=500, timeout=10.0):
    """
    Run Nelder-Mead optimization on the points.
    """
    start_time = time.time()
    
    # Flatten points to 1D array for scipy.optimize.minimize
    x0 = points.flatten()
    
    # Options for Nelder-Mead
    # initial_simplex_size is important. 
    # If too small, it gets stuck. If too large, it might jump out of bounds (handled by penalty).
    # Let's try a moderate size.
    options = {
        'maxiter': max_iter,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'adaptive': True # Helps with scaling issues
    }
    
    try:
        res = minimize(
            _objective_with_penalty,
            x0,
            method='Nelder-Mead',
            options=options
        )
        
        if time.time() - start_time > timeout:
            return points # Return original if timeout
        
        best_params = res.x
        best_points = best_params.reshape(-1, 2)
        
        # Check if the result is valid
        valid = True
        for p in best_points:
            if not _is_inside_triangle(p):
                valid = False
                break
        
        if not valid:
            # If invalid, try to repair or return original
            # For now, return original
            return points
            
        return best_points
        
    except Exception:
        return points


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy: Multi-start Nelder-Mead with Simulated Annealing-like restarts
    # 1. Start with structured initial
    # 2. Start with several random initializations
    # 3. For each start, run Nelder-Mead.
    # 4. If time permits, do a "polish" phase with smaller simplex size.
    
    initial_configs = []
    initial_configs.append(_generate_structured_initial())
    
    # Add random starts
    for _ in range(20):
        initial_configs.append(_generate_random_points(rng))
    
    # Run Nelder-Mead on each
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        # Determine max iterations based on time left
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Allocate time per start
        # If many starts, give each less time.
        # 21 starts total. 145s / 21 ~ 6.9s per start.
        # Let's cap per start at 5s to be safe, allowing for overhead.
        per_start_timeout = min(5.0, time_left / max(1, 21 - idx))
        
        # Run Nelder-Mead
        # We can try different initial simplex sizes.
        # Small size for polish, larger for exploration.
        # Let's just use adaptive NM which handles this somewhat.
        
        optimized_pts = _nelder_mead_optimize(init_pts, max_iter=300, timeout=per_start_timeout)
        
        current_min = _min_triangle_area(optimized_pts)
        
        if current_min > best_min:
            best_min = current_min
            best_points = optimized_pts.copy()
            
            # If we have found a good solution, we can try to polish it further
            # with a tighter tolerance or a second round of NM.
            if time.time() < deadline - 10 and best_min > 0.01:
                # Polish phase
                # Use a smaller initial simplex size implicitly by setting xatol/fatol tighter?
                # Or just run NM again on the best point found so far.
                # Since NM is local, running it again on the same point might not help unless 
                # we perturb it slightly.
                
                # Let's do a quick local search with smaller steps.
                # We can simulate this by running NM with a very small xatol.
                polish_timeout = min(5.0, deadline - time.time() - 1)
                if polish_timeout > 1.0:
                    polished_pts = _nelder_mead_optimize(best_points, max_iter=200, timeout=polish_timeout)
                    polished_min = _min_triangle_area(polished_pts)
                    if polished_min > best_min:
                        best_min = polished_min
                        best_points = polished_pts.copy()

    # If no valid points found (should not happen), fallback
    if best_points is None:
        best_points = _generate_random_points(rng)

    return best_points
