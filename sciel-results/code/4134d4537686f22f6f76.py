import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _triangle_areas(points):
    """
    Compute the areas of all triangles formed by triples of points.
    Returns an array of shape (M,) where M = N*(N-1)*(N-2)/6
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Vector from i to j
                v1 = points[j] - points[i]
                # Vector from i to k
                v2 = points[k] - points[i]
                # Cross product magnitude (2 * Area)
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        # If points are collinear or degenerate, return a small value to penalize
        return 1e-10

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    areas = _triangle_areas(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _objective(points_flat, tau=50.0):
    """
    Objective function for minimization.
    We want to maximize score, so we minimize -score.
    We use a soft-min approximation to make the function smoother.
    """
    points = points_flat.reshape(-1, 2)
    
    # Check for degenerate hull
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6 # Large penalty

    areas = _triangle_areas(points)
    
    # Soft min approximation: -1/tau * log(sum(exp(-tau * areas)))
    # To avoid overflow, subtract max area? No, we want min.
    # min(A) approx -1/tau * log(sum(exp(-tau * A)))
    # If A is large, exp(-tau*A) is small.
    # Let's use the standard log-sum-exp trick for stability.
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -tau * A.
    # max(x) = -tau * min(A).
    # So softmin = -1/tau * ( -tau * min(A) + log(sum(exp(-tau * A + tau * min(A)))) )
    #           = min(A) - 1/tau * log(sum(exp(-tau * (A - min(A)))))
    # This is always >= min(A).
    
    min_a = np.min(areas)
    # Shift to prevent underflow/overflow in exp
    shifted = areas - min_a
    # exp(-tau * shifted)
    # shifted >= 0, so -tau * shifted <= 0. exp is in (0, 1].
    # Sum is at most M.
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    soft_min = min_a - (1.0 / tau) * log_sum
    
    score = soft_min / hull_area
    return -score

def _normalize_points(points):
    """
    Normalize points so that their convex hull area is 1.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return points
    # Scale factor: area scales with scale^2.
    # We want new_area = 1.
    # new_points = points * s
    # new_area = area * s^2 = 1 => s = 1/sqrt(area)
    s = 1.0 / np.sqrt(hull_area)
    return points * s

def _generate_regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    best_points = None
    best_score = -np.inf

    # List of initial configurations
    inits = []

    # 1. Regular 13-gon
    inits.append(_generate_regular_polygon(N))

    # 2. Regular 7-gon + 6 interior points (center + 5 random? or structured?)
    # Let's try a few variations of "boundary + interior"
    for _ in range(10):
        boundary = _generate_regular_polygon(7, radius=1.0)
        interior = np.random.rand(6, 2) * 0.8 - 0.4 # Inside the heptagon roughly
        pts = np.vstack([boundary, interior])
        inits.append(pts)

    # 3. Random points
    for _ in range(30):
        pts = np.random.rand(N, 2) * 10 - 5
        inits.append(pts)

    # 4. Lattice-like
    # 5x3 grid clipped?
    for _ in range(5):
        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 3)
        grid = np.array([[xi, yi] for xi in x for yi in y])
        # Shuffle and take 13
        idx = np.random.choice(len(grid), N, replace=False)
        pts = grid[idx]
        # Add small noise
        pts += np.random.randn(N, 2) * 0.1
        inits.append(pts)

    # Optimization Loop
    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break

        # Normalize initial points to unit area to help the optimizer
        init_pts_norm = _normalize_points(init_pts)
        
        # Flatten for optimizer
        init_flat = init_pts_norm.flatten()

        # Run Nelder-Mead
        # maxiter: 200 is usually enough for local refinement
        # xatol/fatol: Tight enough
        try:
            res = minimize(
                _objective, 
                init_flat, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 300, 
                    'xatol': 1e-5, 
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            opt_pts = res.x.reshape(-1, 2)
        except Exception:
            continue

        # Evaluate true score on the optimized points
        # Note: The optimizer used soft-min, but we score with true min
        true_score = _score(opt_pts)
        
        if true_score > best_score:
            best_score = true_score
            best_points = opt_pts.copy()
            
            # If we found a very good score, we might want to do a finer local search
            # But for now, just proceed.

    # If no points were found (shouldn't happen), return random
    if best_points is None:
        best_points = np.array([[rng.random(), rng.random()] for rng in [np.random.RandomState(i) for i in range(N)]])

    return best_points
