"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _triangle_areas_fast(points):
    """Compute areas of all triangles formed by triples of points using vectorized operations."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    hull = ConvexHull(points)
    return hull.volume


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = np.min(_triangle_areas_fast(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area


def _generate_points(params):
    """
    Generate 13 points with angular perturbations on a unit circle.
    params: [a1, phi1, a2, phi2]
    - a1, phi1: amplitude and phase for 1st harmonic angular perturbation
    - a2, phi2: amplitude and phase for 2nd harmonic angular perturbation
    
    angle(k) = k*2*pi/13 + a1*cos(k*2*pi/13 + phi1) + a2*cos(2*k*2*pi/13 + phi2)
    """
    n = 13
    a1, phi1 = params[0], params[1]
    a2, phi2 = params[2], params[3]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbations (keep small to ensure convexity)
    angular_pert = a1 * np.cos(1 * base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    
    angles = base_angles + angular_pert
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _is_convex(points):
    """Check if all points are on the convex hull (convex configuration)."""
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception:
        return False


def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    if not _is_convex(points):
        return -0.01
    return -_score(points)


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses angular Fourier perturbation (2 harmonics) with Nelder-Mead optimization.
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [a1, phi1, a2, phi2]
    bounds = [
        (-0.1, 0.1),         # a1: 1st harmonic angular amplitude
        (0, 2 * np.pi),      # phi1: 1st harmonic phase
        (-0.05, 0.05),       # a2: 2nd harmonic angular amplitude
        (0, 2 * np.pi),      # phi2: 2nd harmonic phase
    ]
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    for trial in range(20):
        if trial == 0:
            # Start with regular polygon (no perturbation)
            init_params = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            init_params = np.array([
                rng.uniform(-0.05, 0.05),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.03, 0.03),
                rng.uniform(0, 2 * np.pi),
            ])
        
        try:
            result = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            
            final_params = result.x
            for i in range(len(final_params)):
                final_params[i] = np.clip(final_params[i], bounds[i][0], bounds[i][1])
            
            points = _generate_points(final_params)
            
            if not _is_convex(points):
                continue
            
            score = _score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            continue
    
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
