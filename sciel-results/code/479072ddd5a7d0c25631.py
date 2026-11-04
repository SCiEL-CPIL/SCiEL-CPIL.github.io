"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 13

# EVOLVE-BLOCK-START
def _all_triangle_areas(points):
    """Compute all C(n,3) triangle areas efficiently."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                cross = (points[j,0] - points[i,0]) * (points[k,1] - points[i,1]) - \
                        (points[j,1] - points[i,1]) * (points[k,0] - points[i,0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    points = np.asarray(points)
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    return min_area / hull_area

def _soft_min_ratio(params, temperature=1.0):
    """
    Soft-min objective: minimize -soft_min(min_triangle_area / hull_area)
    params: (2*N,) array of [x1, y1, x2, y2, ...]
    Uses smooth approximation to handle non-differentiability.
    """
    points = params.reshape(N, 2)
    
    # Normalize: center at origin, scale so hull area = 1
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    hull_area = _hull_area(points_centered)
    if hull_area <= 1e-12:
        return 1e10
    
    # Scale to unit area
    scale = np.sqrt(1.0 / hull_area)
    points_scaled = points_centered * scale
    
    # Compute all triangle areas
    areas = _all_triangle_areas(points_scaled)
    
    if np.min(areas) <= 1e-12:
        return 1e10
    
    # Soft min: log-sum-exp approximation
    # soft_min(a) ≈ -log(sum(exp(-a_i / T))) * T
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min_val = -log_sum * temperature
    
    return soft_min_val

def _transform_to_fixed(params):
    """
    Transform parameters to fixed representation:
    - Center at origin
    - Scale so hull area = 1
    Returns (2*N,) array.
    """
    points = params.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    hull_area = _hull_area(points_centered)
    if hull_area <= 1e-12:
        return points_centered
    
    scale = np.sqrt(1.0 / hull_area)
    points_scaled = points_centered * scale
    
    return points_scaled.ravel()

def _generate_initial_params(rng, method='polygon'):
    """Generate initial parameter vectors."""
    if method == 'polygon':
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
        radii = 1.0 + rng.normal(0, 0.05, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
    elif method == 'halton':
        # Halton sequence for quasi-random points
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        xs = [halton(i+1, 2) for i in range(N)]
        ys = [halton(i+1, 3) for i in range(N)]
        x = np.array(xs) - 0.5
        y = np.array(ys) - 0.5
        x *= 2.0
        y *= 2.0
    else:  # random
        x = rng.uniform(-1, 1, N)
        y = rng.uniform(-1, 1, N)
    
    return np.column_stack([x, y]).ravel()

def construct():
    """
    Construct 13 points to maximize the minimum triangle area relative to the convex hull area.
    Uses parameterized representation with smooth-min optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Differential evolution for global search
    # Use bounds in a reasonable range
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    try:
        de_result = differential_evolution(
            _soft_min_ratio,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False,
            init='halton',
            workers=1
        )
        
        if de_result.fun < 1e9:
            best_params = de_result.x
            best_points = _transform_to_fixed(best_params)
            best_score = _score(best_points)
    except Exception:
        pass
    
    # Phase 2: Local refinement with L-BFGS-B from multiple starts
    num_starts = 15
    for i in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate diverse starting points
        if i == 0:
            if best_points is not None:
                x0 = best_points.copy()
            else:
                x0 = _generate_initial_params(rng, 'polygon')
        elif i % 3 == 1:
            x0 = _generate_initial_params(rng, 'polygon')
        elif i % 3 == 2:
            x0 = _generate_initial_params(rng, 'halton')
        else:
            x0 = _generate_initial_params(rng, 'random')
        
        # Add small noise to escape local minima
        if i > 0:
            x0 = x0 + rng.normal(0, 0.05, 2*N)
        
        try:
            result = minimize(
                _soft_min_ratio,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 1e9:
                refined_params = result.x
                refined_points = _transform_to_fixed(refined_params)
                score = _score(refined_points)
                
                if score > best_score:
                    best_score = score
                    best_points = refined_points
        except Exception:
            continue
    
    # Phase 3: Final polish with Nelder-Mead on the best solution
    if time.time() < deadline - 5.0 and best_points is not None:
        try:
            def _neg_score(params):
                pts = _transform_to_fixed(params)
                return -_score(pts)
            
            result = minimize(
                _neg_score,
                best_points,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result.fun < -best_score:
                final_params = result.x
                final_points = _transform_to_fixed(final_params)
                final_score = _score(final_points)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points
        except Exception:
            pass
    
    # Ensure we return something valid
    if best_points is None or len(best_points) != 2 * N:
        # Fallback: regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)]).ravel()
    
    return best_points.reshape(N, 2)
# EVOLVE-BLOCK-END
