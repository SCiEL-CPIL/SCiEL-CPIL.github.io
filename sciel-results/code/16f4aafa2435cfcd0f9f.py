"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                areas.append(0.5 * cross)
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    return hull.volume  # For 2D, volume is area


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = np.min(_triangle_areas(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area


def _generate_points(params):
    """
    Generate 13 points from parameters.
    params: [theta0, r1, theta1, r2, theta2]
    - theta0: base angle offset
    - r1, theta1: first harmonic amplitude and phase
    - r2, theta2: second harmonic amplitude and phase
    
    Points are placed on a perturbed circle:
    r(k) = 1 + r1*cos(k*2*pi/13 + theta1) + r2*cos(2*k*2*pi/13 + theta2)
    angle(k) = theta0 + k*2*pi/13
    """
    n = 13
    theta0, r1, theta1, r2, theta2 = params
    
    angles = theta0 + np.arange(n) * 2 * np.pi / n
    radii = 1.0 + r1 * np.cos(angles - theta1) + r2 * np.cos(2 * (angles - theta2))
    
    # Ensure radii are positive
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    return -_score(points)


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES-like optimization via scipy's minimize with multiple restarts.
    """
    best_score = 0.0
    best_points = None
    
    # Try multiple random initializations
    rng = np.random.RandomState(42)
    
    for trial in range(5):
        # Random initial parameters
        if trial == 0:
            # Start with regular polygon
            init_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            init_params = np.array([
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.3, 0.3),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi),
            ])
        
        try:
            result = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True,
                }
            )
            
            final_params = result.x
            points = _generate_points(final_params)
            score = _score(points)
            
            if score > best_score:
                best_score = score
                best_points = points
        except Exception:
            continue
    
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
