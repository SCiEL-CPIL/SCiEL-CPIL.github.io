"""Optimized program for Heilbronn convex regions n=14."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14
DEADLINE = 120.0  # seconds, safety margin from 150s limit
SOFT_BETA = 80.0


def _all_triangle_areas(points):
    """Compute areas of all C(n,3) triangles. Vectorized."""
    n = len(points)
    # Get all triples
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    return areas


def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: use all points if hull fails
        return 0.0


def _soft_min_triangle_area(points, beta=SOFT_BETA):
    """Smooth approximation of min triangle area using log-sum-exp."""
    areas = _all_triangle_areas(points)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum / beta
    return soft_min


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area


def _negative_soft_score(points, beta=SOFT_BETA):
    """Negative of soft-min triangle area / convex hull area for minimization."""
    areas = _all_triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    # Soft min of areas
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum / beta
    return -soft_min / hull_area


def _gradient_negative_soft_score(points, beta=SOFT_BETA):
    """Approximate gradient of negative soft score via finite differences."""
    # Use numerical gradient for simplicity
    grad = np.zeros_like(points)
    eps = 1e-7
    f0 = _negative_soft_score(points, beta)
    for i in range(len(points)):
        for j in range(2):
            points[i, j] += eps
            f1 = _negative_soft_score(points, beta)
            points[i, j] -= eps
            grad[i, j] = (f1 - f0) / eps
    return grad


def construct():
    start_time = time.time()
    best_points = None
    best_score = 0.0
    
    # Initial configurations: try several structured and random starts
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    def _generate_initial():
        """Generate a random point set in unit square."""
        return rng.random((N, 2))
    
    # Also try a grid-like initial configuration
    def _generate_grid():
        """Generate points on a regular grid in unit square."""
        # 14 points: try 4x4 grid minus 2, or 5x3 minus 1, etc.
        # Use 4 columns, 4 rows = 16, take 14
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, 0.9, 4)
        pts = []
        for x in xs:
            for y in ys:
                pts.append([x, y])
        # Remove 2 random points
        pts = np.array(pts[:14])
        return pts
    
    initial_configs = [_generate_grid(), _generate_initial(), _generate_initial(), 
                       _generate_initial(), _generate_initial()]
    
    # Optimize each initial configuration
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        try:
            # Use L-BFGS-B with soft-min surrogate
            result = minimize(
                _negative_soft_score,
                init_pts,
                args=(SOFT_BETA,),
                method='L-BFGS-B',
                jac=_gradient_negative_soft_score,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            opt_pts = result.x
            
            # Compute actual score
            actual_score = _score(opt_pts)
            
            if actual_score > best_score:
                best_score = actual_score
                best_points = opt_pts.copy()
                
        except Exception:
            continue
    
    # If we have time, do a final polish with tighter parameters
    if best_points is not None and time.time() - start_time < DEADLINE - 15:
        try:
            # Polish with higher beta for sharper min approximation
            result2 = minimize(
                _negative_soft_score,
                best_points,
                args=(100.0,),
                method='L-BFGS-B',
                jac=_gradient_negative_soft_score,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            polished_pts = result2.x
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts.copy()
        except Exception:
            pass
    
    # If no optimization succeeded, return best initial or a default
    if best_points is None:
        best_points = _generate_grid()
    
    return best_points
