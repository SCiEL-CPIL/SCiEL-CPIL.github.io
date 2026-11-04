"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _all_tri_indices():
    """Precompute all triple indices."""
    return list(combinations(range(N), 3))

TRI_INDICES = _all_tri_indices()
N_TRIS = len(TRI_INDICES)

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized. Returns array of shape (N_TRIS,)."""
    # points: (N, 2)
    # For each triple (i,j,k): area = 0.5 * |cross(pj-pi, pk-pi)|
    areas = np.empty(N_TRIS)
    for t, (i, j, k) in enumerate(TRI_INDICES):
        pi = points[i]
        pj = points[j]
        pk = points[k]
        ab = pj - pi
        ac = pk - pi
        areas[t] = 0.5 * abs(ab[0] * ac[1] - ab[1] * ac[0])
    return areas

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _is_convex_position(points):
    """Check if all points are on the convex hull."""
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception:
        return False

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 0.0
    areas = _all_triangle_areas(points)
    min_ta = np.min(areas)
    return min_ta / hull_a

def _soft_min(areas, beta):
    """Soft minimum: approximates min with a smooth function."""
    # Use negative log-sum-exp for soft min
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    if max_neg > 1e10:
        return np.exp(max_neg)  # degenerate
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    return -log_sum / beta

def _soft_min_score(points, beta):
    """Soft minimum of triangle areas divided by hull area."""
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 0.0
    areas = _all_triangle_areas(points)
    sm = _soft_min(areas, beta)
    return sm / hull_a

def _neg_soft_min_score(points_flat, beta):
    """Objective for optimization: negative soft min score."""
    points = points_flat.reshape(-1, 2)
    if not _is_convex_position(points):
        return 1000.0
    return -_soft_min_score(points, beta)

def _halton_sequence(n, base):
    """Generate Halton sequence in [0,1)."""
    result = np.zeros(n)
    f = 1.0 / base
    for i in range(n):
        idx = i + 1
        x = 0.0
        while idx > 0:
            idx, rem = divmod(idx, base)
            x += rem * f
            f /= base
        result[i] = x
    return result

def _generate_halton_points(n, base1=2, base2=3, scale=1.0):
    """Generate points using Halton sequences for 2D."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    # Map to a disk-like region
    angles = x * 2 * np.pi
    radii = 0.5 + 0.5 * y
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points * scale

def _regular_polygon(n, scale=1.0):
    """Generate regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return scale * np.column_stack([np.cos(angles), np.sin(angles)])

def _random_convex_points(n, seed=None):
    """Generate random points in convex position."""
    rng = np.random.RandomState(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.7, 1.3, n)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def _simulated_annealing_search(points, beta_start=10, beta_end=50, 
                                  n_iterations=5000, t_start=0.01, t_end=0.001,
                                  seed=42):
    """Simulated annealing to maximize soft min score."""
    rng = np.random.RandomState(seed)
    best_points = points.copy()
    best_score = _soft_min_score(points, beta_start)
    
    current_points = points.copy()
    current_score = best_score
    
    n_iter = n_iterations
    for it in range(n_iter):
        if _time_remaining() < 5:
            break
        
        # Annealing schedule
        progress = it / max(n_iter - 1, 1)
        beta = beta_start + (beta_end - beta_start) * progress
        temperature = t_start * (t_end / t_start) ** progress
        
        # Generate perturbation: move one or two random points
        n_perturb = rng.randint(1, 3)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        
        new_points = current_points.copy()
        for idx in idxs:
            # Random perturbation in polar coordinates around current point
            dx = rng.normal(0, temperature * 0.5)
            dy = rng.normal(0, temperature * 0.5)
            new_points[idx] = current_points[idx] + np.array([dx, dy])
        
        # Check convex position
        if not _is_convex_position(new_points):
            continue
        
        new_score = _soft_min_score(new_points, beta)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-10)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
    
    return best_points, best_score

def _coordinate_descent_polish(points, max_iters=200, step_size=0.01, seed=42):
    """Coordinate descent to polish the configuration."""
    rng = np.random.RandomState(seed)
    current_points = points.copy()
    current_score = _soft_min_score(current_points, 30)
    best_points = current_points.copy()
    best_score = current_score
    
    for it in range(max_iters):
        if _time_remaining() < 3:
            break
        
        improved = False
        # Try moving each point in 8 directions
        for i in range(N):
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                for step in [step_size, step_size * 0.5, step_size * 2]:
                    new_points = current_points.copy()
                    new_points[i] = current_points[i] + step * np.array([np.cos(angle), np.sin(angle)])
                    
                    if not _is_convex_position(new_points):
                        continue
                    
                    new_score = _soft_min_score(new_points, 30)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
        
        if not improved:
            break
    
    return best_points, best_score

def _targeted_repair(points):
    """If points are not in convex position, try to fix by pushing interior points out."""
    if _is_convex_position(points):
        return points
    
    # Find centroid
    centroid = np.mean(points, axis=0)
    
    # For each point, check if it's inside the hull of others
    new_points = points.copy()
    for i in range(N):
        others = np.delete(points, i, axis=0)
        try:
            hull = ConvexHull(others)
            # Check if point i is inside hull
            from scipy.spatial import Delaunay
            tri = Delaunay(others)
            if tri.find_simplex(new_points[i]) >= 0:
                # Push point outward from centroid
                direction = new_points[i] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    new_points[i] = new_points[i] + 0.1 * direction
        except Exception:
            pass
    
    return new_points

def construct():
    """
    Place 13 points to maximize min triangle area / convex hull area.
    
    Strategy:
    1. Multiple diverse initializations (regular polygon, Halton, random convex)
    2. Simulated annealing with soft-min objective
    3. Coordinate descent polish
    4. Return best configuration
    """
    best_points = None
    best_score = -1.0
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular polygon
    initializations.append(_regular_polygon(N))
    
    # 2. Halton sequence points
    initializations.append(_generate_halton_points(N, base1=2, base2=3, scale=1.0))
    initializations.append(_generate_halton_points(N, base1=3, base2=5, scale=1.0))
    
    # 3. Random convex points with different seeds
    for seed in [42, 123, 456, 789]:
        initializations.append(_random_convex_points(N, seed=seed))
    
    # 4. Slightly perturbed regular polygons
    rng = np.random.RandomState(0)
    for _ in range(3):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + 0.05 * rng.randn(N)
        radii = np.maximum(radii, 0.5)
        initializations.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Run simulated annealing from each initialization
    for init_idx, init_points in enumerate(initializations):
        if _time_remaining() < 10:
            break
        
        # Repair if needed
        init_points = _targeted_repair(init_points)
        
        if not _is_convex_position(init_points):
            continue
        
        # Simulated annealing
        sa_points, sa_score = _simulated_annealing_search(
            init_points, 
            beta_start=10, beta_end=50,
            n_iterations=3000,
            t_start=0.02, t_end=0.002,
            seed=100 + init_idx
        )
        
        # Coordinate descent polish
        cd_points, cd_score = _coordinate_descent_polish(
            sa_points,
            max_iters=100,
            step_size=0.01,
            seed=200 + init_idx
        )
        
        # Evaluate actual score (not soft min)
        actual_score = _score(cd_points)
        if actual_score > best_score:
            best_score = actual_score
            best_points = cd_points.copy()
        
        # Also check SA result
        sa_actual = _score(sa_points)
        if sa_actual > best_score:
            best_score = sa_actual
            best_points = sa_points.copy()
    
    # Final polish on best
    if best_points is not None and _time_remaining() > 5:
        final_points, final_score = _coordinate_descent_polish(
            best_points,
            max_iters=50,
            step_size=0.005,
            seed=999
        )
        final_actual = _score(final_points)
        if final_actual > best_score:
            best_score = final_actual
            best_points = final_points
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(N)
    
    # Ensure valid convex position
    if not _is_convex_position(best_points):
        best_points = _regular_polygon(N)
    
    return best_points
# EVOLVE-BLOCK-END
