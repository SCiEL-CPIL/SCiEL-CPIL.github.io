import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 14

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized computation for speed
    # i, j, k
    # We can use combinations but for N=14, loops are fine and simple.
    # 14*13*12/6 = 364 triangles.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    
    min_area = _min_triangle_area(points)
    
    # Optional: Check if all points are on the hull
    # If not, the configuration is likely suboptimal, but the formula still holds.
    # The problem says "Place 14 points in a planar convex region... normalized by the point set convex hull area".
    # It doesn't strictly require points to be vertices, but optimal solutions have them on the boundary.
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Parameterization: 7 points (x, y) for the first half, the other 7 are negatives.
    # This enforces central symmetry.
    # We optimize 14 variables: x0, y0, x1, y1, ..., x6, y6
    
    def unpack(params):
        # params: (14,)
        pts = np.zeros((14, 2))
        for i in range(7):
            x = params[2*i]
            y = params[2*i+1]
            pts[i] = [x, y]
            pts[i+7] = [-x, -y]
        return pts

    def objective(params):
        pts = unpack(params)
        s = _score(pts)
        # We want to maximize s, so minimize -s
        return -s

    # Bounds for differential evolution
    # We can scale the points to be within [-1, 1]
    bounds = [(-1.0, 1.0)] * 14
    
    # Run Differential Evolution
    # popsize=10, maxiter=50, tol=0.01
    # We check time in a callback or just run a fixed number of iterations
    # DE is stochastic, so we rely on the maxiter limit.
    
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4,
            polish=False  # We will do our own polish
        )
    except Exception as e:
        # Fallback to random if DE fails
        return np.array([[np.random.uniform(-1, 1), np.random.uniform(-1, 1)] for _ in range(14)])

    best_params = de_result.x
    best_score = -de_result.fun
    
    # Check time
    if time.time() > deadline - 20.0:
        return unpack(best_params)

    # Local refinement with L-BFGS-B
    # L-BFGS-B requires bounds
    try:
        res = minimize(
            objective, 
            best_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        if res.fun < -best_score: # res.fun is -score, so lower is better
            best_params = res.x
            best_score = -res.fun
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 10.0:
        return unpack(best_params)

    # Optional: Nelder-Mead for final polish (no bounds needed if we stay close)
    try:
        res_nm = minimize(
            objective, 
            best_params, 
            method='Nelder-Mead',
            options={'maxiter': 50, 'fatol': 1e-10}
        )
        if res_nm.fun < -best_score:
            best_params = res_nm.x
            best_score = -res_nm.fun
    except Exception:
        pass

    final_points = unpack(best_params)
    
    # Final check: ensure points are distinct and valid
    # If score is 0, try a known good configuration or random
    if best_score <= 1e-6:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return final_points
