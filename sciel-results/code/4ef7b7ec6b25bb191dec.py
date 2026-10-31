"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
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
    Generate 13 points from angular perturbation parameters.
    params: [a1, phi1, a2, phi2]
    - a1, phi1: amplitude and phase for 1st angular harmonic
    - a2, phi2: amplitude and phase for 2nd angular harmonic
    
    Points are placed at angles:
    theta(k) = k*2*pi/13 + a1*cos(1*k*2*pi/13 + phi1) + a2*cos(2*k*2*pi/13 + phi2)
    and radii are fixed at 1.0
    """
    n = 13
    a1, phi1, a2, phi2 = params
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbation
    angle_perturbation = a1 * np.cos(1 * base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    
    angles = base_angles + angle_perturbation
    radii = np.ones(n)
    
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _is_convex(points):
    """Check if all points lie on the convex hull (i.e., form a convex polygon)."""
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == 13
    except Exception:
        return False


def _objective(params):
    """Negative score for minimization. Penalize non-convex configurations."""
    points = _generate_points(params)
    if not _is_convex(points):
        return -0.001  # Penalty for non-convex
    return -_score(points)


def _cma_es(params, sigma, bounds, rng, max_iter=200, pop_size=12):
    """
    Simple CMA-ES implementation for low-dimensional optimization.
    """
    n = len(params)
    pop_size = max(pop_size, 4 + int(3 * np.log(n)))
    
    best_x = params.copy()
    best_f = _objective(best_x)
    
    mean = params.copy()
    sigma = sigma
    
    # Eigenvalue decomposition for step size adaptation
    cc = (4 + n / 3) / (n + 4 + 2 / n)
    cs = (2 + 2 / (n + 1)) / (n + 4 + 2 / n)
    
    for iteration in range(max_iter):
        # Generate population
        population = []
        for _ in range(pop_size):
            # Sample direction from N(0, I)
            z = rng.standard_normal(n)
            step = sigma * z
            x = mean + step
            # Clip to bounds
            for i in range(n):
                x[i] = np.clip(x[i], bounds[i][0], bounds[i][1])
            population.append(x)
        
        # Evaluate
        fitnesses = [_objective(x) for x in population]
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        sorted_pop = [population[i] for i in idx]
        sorted_fit = [fitnesses[i] for i in idx]
        
        # Update best
        if sorted_fit[0] < best_f:
            best_f = sorted_fit[0]
            best_x = sorted_pop[0].copy()
        
        # Update mean using weighted average of top half
        w = np.log(np.arange(1, pop_size // 2 + 1)) - np.log(pop_size // 2)
        w = w / np.sum(w)
        
        new_mean = np.zeros(n)
        for i in range(pop_size // 2):
            new_mean += w[i] * sorted_pop[i]
        mean = new_mean
        
        # Update sigma
        sigma *= np.exp(cs / (5 * (n + 1)) * (np.std([f - sorted_fit[0] for f in sorted_fit[:pop_size//2]]) / max(sigma, 1e-10) - 1))
        sigma = np.clip(sigma, 1e-8, 10.0)
        
        # Check convergence
        if sigma < 1e-10:
            break
    
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES optimization on 4 angular perturbation parameters.
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [a1, phi1, a2, phi2]
    bounds = [
        (-0.5, 0.5),    # a1: 1st harmonic amplitude
        (0, 2 * np.pi), # phi1: 1st harmonic phase
        (-0.3, 0.3),    # a2: 2nd harmonic amplitude
        (0, 2 * np.pi), # phi2: 2nd harmonic phase
    ]
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    num_restarts = 6
    for trial in range(num_restarts):
        if trial == 0:
            # Start with regular polygon (zero perturbation)
            init_params = np.array([0.0, 0.0, 0.0, 0.0])
            sigma = 0.1
        else:
            # Random initialization
            init_params = np.array([
                rng.uniform(-0.3, 0.3),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi),
            ])
            sigma = 0.05
        
        try:
            # Run CMA-ES
            final_params, final_obj = _cma_es(
                init_params,
                sigma=sigma,
                bounds=bounds,
                rng=rng,
                max_iter=300,
                pop_size=12
            )
            
            points = _generate_points(final_params)
            
            # Verify convexity
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
