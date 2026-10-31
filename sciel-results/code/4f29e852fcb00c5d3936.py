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
    Generate 13 points by perturbing angles of a regular 13-gon.
    params: [a1, phi1, a2, phi2]
    - a1, phi1: amplitude and phase for 1st harmonic angle perturbation
    - a2, phi2: amplitude and phase for 2nd harmonic angle perturbation
    
    angle(k) = k*2*pi/13 + a1*cos(k*2*pi/13 + phi1) + a2*cos(2*k*2*pi/13 + phi2)
    Points are on the unit circle: (cos(angle_k), sin(angle_k))
    """
    n = 13
    a1, phi1 = params[0], params[1]
    a2, phi2 = params[2], params[3]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    angles = base_angles + a1 * np.cos(base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    
    # Sort angles to maintain convex ordering
    angles = np.sort(angles)
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _is_convex(points):
    """Check if all points lie on the convex hull (i.e., form a convex polygon)."""
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception:
        return False


def _cma_es(params, sigma, bounds, rng, max_iter=500, pop_size=10):
    """
    Simple CMA-ES implementation for low-dimensional optimization.
    Returns best_params and best_score.
    """
    n_params = len(params)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    best_params = params.copy()
    best_score = _score(_generate_points(params))
    
    for iteration in range(max_iter):
        # Generate population
        population = []
        scores = []
        
        for _ in range(pop_size):
            new_params = params + sigma * rng.standard_normal(n_params)
            # Clip to bounds
            new_params = np.clip(new_params, lower, upper)
            population.append(new_params)
            
            points = _generate_points(new_params)
            if not _is_convex(points):
                scores.append(-1.0)
            else:
                scores.append(_score(points))
        
        # Select best
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = population[best_idx].copy()
        
        # Update mean toward best
        params = 0.7 * params + 0.3 * population[best_idx]
        params = np.clip(params, lower, upper)
        
        # Adaptive sigma
        sigma *= 0.995
        
        # Early termination if sigma is too small
        if sigma < 1e-8:
            break
    
    return best_params, best_score


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses angular perturbation with CMA-ES optimization.
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [a1, phi1, a2, phi2]
    bounds = [
        (-0.3, 0.3),       # a1: 1st harmonic amplitude
        (0, 2 * np.pi),    # phi1: 1st harmonic phase
        (-0.2, 0.2),       # a2: 2nd harmonic amplitude
        (0, 2 * np.pi),    # phi2: 2nd harmonic phase
    ]
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    n_restarts = 6
    
    for trial in range(n_restarts):
        if trial == 0:
            # Start with regular polygon
            init_params = np.array([0.0, 0.0, 0.0, 0.0])
            sigma = 0.05
        else:
            # Random initialization
            init_params = np.array([
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.15, 0.15),
                rng.uniform(0, 2 * np.pi),
            ])
            sigma = rng.uniform(0.03, 0.08)
        
        try:
            final_params, final_score = _cma_es(
                init_params,
                sigma,
                bounds,
                rng,
                max_iter=300,
                pop_size=8
            )
            
            points = _generate_points(final_params)
            
            # Verify convexity
            if not _is_convex(points):
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
