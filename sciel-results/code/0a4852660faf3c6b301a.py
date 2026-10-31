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
    Generate 13 points from parameters using angular perturbation.
    params: [theta0, a1, phi1, a2, phi2]
    - theta0: global rotation
    - a1, phi1: amplitude and phase for 1st harmonic angular perturbation
    - a2, phi2: amplitude and phase for 2nd harmonic angular perturbation
    
    Points are placed on unit circle with perturbed angles:
    angle(k) = theta0 + k*2*pi/13 + a1*cos(k*2*pi/13 + phi1) + a2*cos(2*k*2*pi/13 + phi2)
    """
    n = 13
    theta0 = params[0]
    a1, phi1 = params[1], params[2]
    a2, phi2 = params[3], params[4]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbation using first 2 harmonics
    angles = theta0 + base_angles + a1 * np.cos(base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    
    # Points on unit circle with perturbed angles
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < 13:
            return -0.01
    except Exception:
        return -0.01
    
    return -_score(points)


def _cma_es(params, sigma, rng, bounds, n_iter=500, population_size=15):
    """
    Simple CMA-ES implementation for optimization.
    params: initial parameters
    sigma: initial step size
    rng: random number generator
    bounds: list of (low, high) for each parameter
    n_iter: number of iterations
    population_size: number of samples per iteration
    """
    n_params = len(params)
    mean = params.copy()
    step_size = sigma
    
    # Covariance matrix (diagonal initially)
    C = np.eye(n_params)
    eigenvalues = np.ones(n_params)
    eigenvectors = np.eye(n_params)
    
    best_x = mean.copy()
    best_f = _objective(best_x)
    
    for it in range(n_iter):
        # Generate population
        for i in range(population_size):
            # Sample from multivariate normal
            z = rng.standard_normal(n_params)
            x = mean + step_size * (eigenvectors @ (eigenvalues ** 0.5 * z))
            
            # Clip to bounds
            for j in range(n_params):
                x[j] = np.clip(x[j], bounds[j][0], bounds[j][1])
            
            f = _objective(x)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Update mean (simple version: move toward best)
        mean = best_x.copy()
        
        # Decay step size
        step_size *= 0.995
        
        # If step size is too small, break
        if step_size < 1e-6:
            break
    
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES optimization with angular perturbation model (first 2 harmonics).
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [theta0, a1, phi1, a2, phi2]
    bounds = [
        (0, 2 * np.pi),      # theta0
        (-0.15, 0.15),       # a1 (angular perturbation, 1st harmonic)
        (0, 2 * np.pi),      # phi1
        (-0.1, 0.1),         # a2 (angular perturbation, 2nd harmonic)
        (0, 2 * np.pi),      # phi2
    ]
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    for trial in range(6):
        if trial == 0:
            # Start with regular polygon (no perturbation)
            init_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            # Random initialization
            init_params = np.array([
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.1, 0.1),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.05, 0.05),
                rng.uniform(0, 2 * np.pi),
            ])
        
        try:
            # Run CMA-ES
            result_params, result_f = _cma_es(
                init_params,
                sigma=0.05,
                rng=rng,
                bounds=bounds,
                n_iter=300,
                population_size=12
            )
            
            # Also try Nelder-Mead for local refinement
            result = minimize(
                _objective,
                result_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            
            final_params = result.x
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
