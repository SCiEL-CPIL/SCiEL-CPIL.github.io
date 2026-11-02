"""Program for Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 13

# EVOLVE-BLOCK-START
def _triangle_area(p1, p2, p3):
    """Compute absolute area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _convex_hull_area(points):
    """Compute area of convex hull of points using monotone chain."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
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
    
    area = 0.0
    n = len(hull)
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) * 0.5

def _objective(x, n_points):
    """
    Objective function: minimize negative of (min_triangle_area / hull_area).
    x is a flat array of 2*n_points coordinates.
    """
    pts = x.reshape(n_points, 2)
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n_points):
        for j in range(i + 1, n_points):
            for k in range(j + 1, n_points):
                area = _triangle_area(pts[i], pts[j], pts[k])
                if area < min_area:
                    min_area = area
    
    # Compute convex hull area
    hull_area = _convex_hull_area(pts)
    
    if hull_area < 1e-12:
        return 1e10  # Degenerate
    
    ratio = min_area / hull_area
    return -ratio

def _gradient(x, n_points, eps=1e-7):
    """Numerical gradient of the objective."""
    grad = np.zeros_like(x)
    f0 = _objective(x, n_points)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (_objective(x_plus, n_points) - f0) / eps
    return grad

def construct():
    """
    Construct 13 points to maximize min triangle area / convex hull area.
    Uses optimization starting from a regular polygon configuration.
    """
    n_points = N
    
    # Start with a regular 13-gon (unit area normalized)
    angles = 2 * np.pi * np.arange(n_points) / n_points
    # Regular polygon with n sides has area = (n/2) * R^2 * sin(2*pi/n)
    # We want area = 1, so R^2 = 2 / (n * sin(2*pi/n))
    R = np.sqrt(2.0 / (n_points * np.sin(2 * np.pi / n_points)))
    x0 = np.array([R * np.cos(a), R * np.sin(a)] for a in angles).flatten()
    
    # Normalize to unit hull area
    hull_area = _convex_hull_area(x0.reshape(n_points, 2))
    if hull_area > 1e-12:
        x0 = x0 * np.sqrt(1.0 / hull_area)
    
    # Try multiple random restarts and pick the best
    best_x = x0.copy()
    best_score = _objective(x0, n_points)
    
    # Optimize from initial point
    result = minimize(
        _objective,
        x0,
        args=(n_points,),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.fun < best_score:
        best_score = result.fun
        best_x = result.x
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    for trial in range(5):
        # Generate random points in a disk
        theta = rng.uniform(0, 2 * np.pi, n_points)
        r = rng.uniform(0.5, 1.0, n_points)
        x_init = np.array([r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i])] for i in range(n_points)).flatten()
        
        # Normalize to unit hull area
        hull_area = _convex_hull_area(x_init.reshape(n_points, 2))
        if hull_area > 1e-12:
            x_init = x_init * np.sqrt(1.0 / hull_area)
        
        result = minimize(
            _objective,
            x_init,
            args=(n_points,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_x = result.x
    
    # Final refinement with more iterations
    result = minimize(
        _objective,
        best_x,
        args=(n_points,),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    if result.fun < best_score:
        best_x = result.x
    
    return best_x.reshape(n_points, 2)
# EVOLVE-BLOCK-END
