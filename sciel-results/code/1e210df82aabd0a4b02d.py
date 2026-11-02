"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _convex_hull_area(points):
    """Compute convex hull area of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: Andrew's monotone chain
        pts = np.array(points)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        if len(pts) <= 1:
            return 0.0
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            return 0.0
        x = hull[:, 0]
        y = hull[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _soft_min_objective(params, beta=100.0):
    """
    Smooth approximation of the negative minimum triangle area divided by hull area.
    Uses a soft-min to approximate the minimum triangle area with a differentiable function.
    params: 28-dimensional vector of point coordinates.
    """
    points = params.reshape(-1, 2)
    
    # Compute all triangle areas
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    areas = np.array(areas)
    
    # Soft-min approximation: -log-sum-exp(-beta * areas) / beta
    # This approximates min(areas) as beta -> infinity
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-beta * (areas - max_area))))
    soft_min = max_area - np.log(np.sum(np.exp(-beta * (areas - max_area)))) / beta
    
    # Compute hull area
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 1e10  # Large penalty for degenerate hull
    
    # We want to maximize soft_min / hull_area, so minimize -soft_min / hull_area
    return -soft_min / hull_area

def _soft_min_objective_grad(params, beta=100.0):
    """
    Gradient of the soft-min objective.
    For simplicity, we use numerical differentiation if analytical is too complex.
    """
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = _soft_min_objective(params, beta)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _soft_min_objective(params_plus, beta)
        grad[i] = (f_plus - f0) / eps
    return grad

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Uses gradient-based optimization with a soft-min approximation of the objective.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, keep 10s safety margin
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Start with regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_params = initial_points.flatten()
    
    # Normalize to unit hull area
    hull_area = _convex_hull_area(initial_points)
    if hull_area > 1e-15:
        scale = np.sqrt(1.0 / hull_area)
        initial_points = initial_points * scale
        initial_params = initial_points.flatten()
    
    best_points = initial_points.copy()
    best_score = _score(initial_points)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_points
    
    # Strategy 2: Gradient-based optimization using Powell's method
    # Powell's method does not require gradients, but we can try with the soft-min objective
    # We'll use multiple restarts with different initial configurations
    
    rng = np.random.RandomState(42)
    
    # Try several different initial configurations
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
        elif restart == 1:
            # Perturbed regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            angles += rng.normal(0, 0.1, N)
            angles = np.sort(angles)
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
        elif restart == 2:
            # Ellipse
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            pts = np.column_stack([1.5 * np.cos(angles), 0.8 * np.sin(angles)])
        elif restart == 3:
            # Random convex points
            angles = np.sort(rng.uniform(0, 2 * np.pi, N))
            radii = rng.uniform(0.5, 1.0, N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Random points in unit disk, then take convex hull vertices
            pts = rng.uniform(-1, 1, (N, 2))
            try:
                hull = ConvexHull(pts)
                pts = pts[hull.vertices]
                # If hull has fewer than N points, add remaining points inside
                if len(pts) < N:
                    # This is tricky; for now, just use the hull points and duplicate
                    # Actually, we need exactly N points. Let's just use random points
                    # and hope they are in convex position
                    pts = rng.uniform(-1, 1, (N, 2))
            except Exception:
                pts = rng.uniform(-1, 1, (N, 2))
        
        # Normalize to unit hull area
        hull_area = _convex_hull_area(pts)
        if hull_area > 1e-15:
            scale = np.sqrt(1.0 / hull_area)
            pts = pts * scale
        
        params = pts.flatten()
        
        # Optimize using Powell's method
        try:
            result = minimize(
                _soft_min_objective,
                params,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.success or result.fun < 0:  # result.fun is negative of score
                opt_points = result.x.reshape(-1, 2)
                # Normalize to unit hull area
                hull_area = _convex_hull_area(opt_points)
                if hull_area > 1e-15:
                    scale = np.sqrt(1.0 / hull_area)
                    opt_points = opt_points * scale
                
                opt_score = _score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    # Strategy 3: Local refinement using Nelder-Mead from best point
    if time.time() < deadline - 10:
        current_params = best_points.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.success:
                opt_points = result.x.reshape(-1, 2)
                hull_area = _convex_hull_area(opt_points)
                if hull_area > 1e-15:
                    scale = np.sqrt(1.0 / hull_area)
                    opt_points = opt_points * scale
                
                opt_score = _score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
