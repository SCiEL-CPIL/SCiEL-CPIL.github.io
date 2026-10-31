"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _triangle_areas_fast(points):
    """Compute areas of all triangles formed by triples of points using vectorized operations."""
    n = len(points)
    # Use the cross product formula: area = 0.5 * |det([p2-p1, p3-p1])|
    # For all triples (i, j, k) with i < j < k
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
    Generate 13 points from parameters using a Fourier series perturbation.
    params: [theta0, a1, phi1, a2, phi2, a3, phi3, a4, phi4]
    - theta0: base angle offset
    - a_m, phi_m: amplitude and phase for m-th harmonic
    
    Points are placed on a perturbed circle:
    r(k) = 1 + sum_m a_m * cos(m * (k*2*pi/13 + theta0) + phi_m)
    angle(k) = theta0 + k*2*pi/13
    """
    n = 13
    theta0 = params[0]
    # Extract harmonics: (a1, phi1), (a2, phi2), (a3, phi3), (a4, phi4)
    a1, phi1 = params[1], params[2]
    a2, phi2 = params[3], params[4]
    a3, phi3 = params[5], params[6]
    a4, phi4 = params[7], params[8]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    angles = theta0 + base_angles
    
    # Compute radii with Fourier perturbation
    radii = 1.0
    radii += a1 * np.cos(1 * base_angles + phi1)
    radii += a2 * np.cos(2 * base_angles + phi2)
    radii += a3 * np.cos(3 * base_angles + phi3)
    radii += a4 * np.cos(4 * base_angles + phi4)
    
    # Ensure radii are positive and points form a convex shape
    radii = np.maximum(radii, 0.3)
    
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    # Check convexity: all points should be on the convex hull
    try:
        hull = ConvexHull(points)
        # If not all points are on the hull, penalize
        if len(hull.vertices) < 13:
            return -0.01  # Strong penalty for non-convex configurations
    except Exception:
        return -0.01
    
    return -_score(points)


def _cma_es_step(params, sigma, rng, bounds):
    """
    Simple CMA-ES step: sample from Gaussian, evaluate, and update.
    This is a simplified version for demonstration.
    """
    n_params = len(params)
    # Sample a new point
    new_params = params + sigma * rng.standard_normal(n_params)
    # Clip to bounds
    for i in range(n_params):
        new_params[i] = np.clip(new_params[i], bounds[i][0], bounds[i][1])
    return new_params


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES-like optimization with multiple restarts and a 9-parameter Fourier model.
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [theta0, a1, phi1, a2, phi2, a3, phi3, a4, phi4]
    bounds = [
        (0, 2 * np.pi),      # theta0
        (-0.4, 0.4),         # a1
        (0, 2 * np.pi),      # phi1
        (-0.3, 0.3),         # a2
        (0, 2 * np.pi),      # phi2
        (-0.2, 0.2),         # a3
        (0, 2 * np.pi),      # phi3
        (-0.15, 0.15),       # a4
        (0, 2 * np.pi),      # phi4
    ]
    
    rng = np.random.RandomState(42)
    
    # Try multiple random initializations
    for trial in range(8):
        # Random initial parameters
        if trial == 0:
            # Start with regular polygon (all amplitudes zero)
            init_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            init_params = np.array([
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.3, 0.3),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.15, 0.15),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.1, 0.1),
                rng.uniform(0, 2 * np.pi),
            ])
        
        try:
            # Use Nelder-Mead for local refinement
            result = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True,
                }
            )
            
            final_params = result.x
            # Clip to bounds
            for i in range(len(final_params)):
                final_params[i] = np.clip(final_params[i], bounds[i][0], bounds[i][1])
            
            points = _generate_points(final_params)
            
            # Verify convexity
            try:
                hull = ConvexHull(points)
                if len(hull.vertices) < 13:
                    continue
            except Exception:
                continue
            
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
