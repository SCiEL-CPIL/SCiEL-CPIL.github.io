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
    Generate 13 points from parameters using angular perturbations.
    params: [c1, phi1, c2, phi2]
    - c1, phi1: first harmonic amplitude and phase for angular perturbation
    - c2, phi2: second harmonic amplitude and phase for angular perturbation
    
    Points are placed on the unit circle with perturbed angles:
    theta(k) = 2*pi*k/13 + c1*sin(2*pi*k/13 + phi1) + c2*sin(4*pi*k/13 + phi2)
    """
    n = 13
    c1, phi1, c2, phi2 = params
    
    base_angles = np.arange(n) * 2 * np.pi / n
    perturbation = c1 * np.sin(base_angles + phi1) + c2 * np.sin(2 * base_angles + phi2)
    angles = base_angles + perturbation
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    return -_score(points)


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses Differential Evolution (evolutionary strategy) to optimize angular perturbations.
    """
    # Bounds for parameters: [c1, phi1, c2, phi2]
    # c1, c2 must be small to maintain convexity/ordering
    # phi1, phi2 are angles
    bounds = [
        (-0.2, 0.2),  # c1
        (0, 2 * np.pi), # phi1
        (-0.2, 0.2),  # c2
        (0, 2 * np.pi)  # phi2
    ]
    
    best_score = 0.0
    best_points = None
    
    try:
        from scipy.optimize import differential_evolution
        
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            popsize=15,
            polish=True
        )
        
        final_params = result.x
        points = _generate_points(final_params)
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points
            
    except Exception:
        pass

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
