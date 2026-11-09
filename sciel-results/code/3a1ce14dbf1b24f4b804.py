# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(n,3) triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return float('inf')

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    if len(points) != 13:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 0 or not np.isfinite(hull_area):
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _gradient_ascent(points, max_iters=2000, step_size=0.01, time_limit=None):
    """
    Gradient ascent to maximize the minimum triangle area.
    Uses subgradient of the minimum triangle area.
    Normalizes by hull area implicitly by working on the ratio.
    """
    n = len(points)
    pts = points.astype(float).copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    if time_limit is None:
        time_limit = 300.0
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find the minimum area triangle and compute its gradient
        min_area = float('inf')
        min_idx = (0, 1, 2)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
        
        i, j, k = min_idx
        p1, p2, p3 = pts[i], pts[j], pts[k]
        
        # Signed area: A = 0.5 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
        # Gradient of |A| with respect to points
        # dA/dx1 = 0.5*(y2-y3), dA/dy1 = 0.5*(x3-x2)
        # dA/dx2 = 0.5*(y3-y1), dA/dy2 = 0.5*(x1-x3)
        # dA/dx3 = 0.5*(y1-y2), dA/dy3 = 0.5*(x2-x1)
        
        signed_area = 0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        sign = 1.0 if signed_area >= 0 else -1.0
        
        grad = np.zeros_like(pts)
        grad[i, 0] += sign * 0.5 * (p2[1] - p3[1])
        grad[i, 1] += sign * 0.5 * (p3[0] - p2[0])
        grad[j, 0] += sign * 0.5 * (p3[1] - p1[1])
        grad[j, 1] += sign * 0.5 * (p1[0] - p3[0])
        grad[k, 0] += sign * 0.5 * (p1[1] - p2[1])
        grad[k, 1] += sign * 0.5 * (p2[0] - p1[0])
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        # Adaptive step size
        new_pts = pts + step_size * grad
        
        # Check if improvement
        new_score = _score(new_pts)
        if new_score > best_score:
            best_pts = new_pts.copy()
            best_score = new_score
            pts = new_pts
            # Decrease step size slightly
            step_size *= 0.995
        else:
            # Decrease step size more aggressively
            step_size *= 0.95
            if step_size < 1e-6:
                break
    
    return best_pts, best_score

def _initialization_regular_polygon(n=13):
    """Points on a regular polygon with one center point."""
    angles = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
    radius = 1.0
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    return np.vstack([pts, center])

def _initialization_grid(n=13):
    """Grid-like initialization."""
    # 4x4 grid minus one corner, scaled
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.array([[x, y] for x in xs for y in ys])
    # Take 13 points
    return grid[:13]

def _initialization_halton(n=13, base=2):
    """Halton sequence initialization."""
    def halton(index, base):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result
    
    pts = np.array([[halton(i + 1, 2), halton(i + 1, 3)] for i in range(n)])
    # Center and scale
    pts = (pts - pts.mean(axis=0)) * 2.0
    return pts

def _initialization_random(n=13, seed=42):
    """Random initialization in unit square."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    return pts

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s deadline with safety margin
    
    best_points = None
    best_score = 0.0
    
    # List of initializations to try
    initializations = [
        _initialization_regular_polygon,
        _initialization_grid,
        _initialization_halton,
        lambda: _initialization_random(seed=1),
        lambda: _initialization_random(seed=2),
        lambda: _initialization_random(seed=3),
        lambda: _initialization_random(seed=4),
        lambda: _initialization_random(seed=5),
    ]
    
    # Try each initialization with gradient ascent
    for init_func in initializations:
        if time.time() > deadline - 60:  # Leave 60s safety margin
            break
        
        pts = init_func()
        
        # Ensure points are well-spread (not all collinear)
        if not np.all(np.isfinite(pts)):
            continue
        hull_area = _hull_area(pts)
        if hull_area <= 0 or not np.isfinite(hull_area):
            continue
        
        # Run gradient ascent with time limit
        time_remaining = deadline - time.time() - 30  # 30s safety margin
        if time_remaining <= 0:
            break
        
        optimized_pts, score = _gradient_ascent(
            pts, 
            max_iters=5000, 
            step_size=0.05, 
            time_limit=min(time_remaining, 120.0)
        )
        
        if score > best_score:
            best_score = score
            best_points = optimized_pts.copy()
        
        # Also try with smaller step sizes from the same start
        if time.time() > deadline - 60:
            break
        
        for ss in [0.02, 0.01, 0.005]:
            if time.time() > deadline - 30:
                break
            time_remaining = deadline - time.time() - 20
            if time_remaining <= 0:
                break
            optimized_pts, score = _gradient_ascent(
                pts,
                max_iters=3000,
                step_size=ss,
                time_limit=min(time_remaining, 60.0)
            )
            if score > best_score:
                best_score = score
                best_points = optimized_pts.copy()
    
    # If no valid result found, return a default
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback: regular polygon with center
        best_points = _initialization_regular_polygon(13)
    
    # Final validation
    if len(best_points) != 13 or not np.all(np.isfinite(best_points)):
        best_points = _initialization_regular_polygon(13)
    
    return best_points

# EVOLVE-BLOCK-END
