"""Rewritten program for Heilbronn convex regions: 13 points, maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def _triangle_areas(points):
    """Compute all triangle areas for N points. Returns array of size C(N,3)."""
    n = len(points)
    indices = np.arange(n)
    i, j, k = np.meshgrid(indices, indices, indices, indexing='ij')
    # Only keep i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    a, b, c = points[i], points[j], points[k]
    # Area = 0.5 * |cross(b-a, c-a)|
    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.abs(cross)
    return areas

def _min_triangle_area(points):
    """Return the minimum triangle area among all triples."""
    return _triangle_areas(points).min()

def _convex_hull_area(points):
    """Return the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, 'volume' is the area
    except Exception:
        # Fallback: use shapely-like manual computation or return a large value to penalize
        # Actually, if hull fails, points are collinear; return 0
        return 0.0

def _score(points):
    """Score = min_triangle_area / convex_hull_area. Returns 0 if hull area is ~0."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _init_points(seed):
    """Initialize points on a grid or random, then normalize."""
    rng = np.random.RandomState(seed)
    # Start with points on a grid to ensure convex position initially
    # 13 points: maybe 4x4 grid minus 3? Or just random in unit square.
    # Let's try a regular polygon first, then perturb.
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(0, 0.2, N)
    radii = 1.0 + rng.uniform(-0.1, 0.1, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def _refine_coordinate_descent(points, time_limit, rng):
    """
    Refine points using coordinate descent with small random steps.
    This is a simple local search.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Pick a random point and coordinate
        idx = rng.randint(0, N-1)
        coord = rng.randint(0, 2)
        
        # Try a small perturbation
        delta = rng.uniform(-step_size, step_size)
        new_points = current_points.copy()
        new_points[idx, coord] += delta
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            # Optionally reduce step size if we've been successful
            # step_size *= 0.95
        else:
            # If we fail, maybe shrink step size
            step_size *= 0.95
            if step_size < 1e-4:
                step_size = 0.05 # Reset if too small
        
    return best_points, best_score

def _refine_scipy(points, time_limit):
    """
    Refine points using scipy.optimize.minimize (Powell or Nelder-Mead).
    We flatten the points to a 2D vector.
    """
    start_time = time.time()
    x0 = points.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10
        pts = x.reshape(N, 2)
        return _negative_score(pts)
    
    best_points = points
    best_score = _score(points)
    
    # Try Nelder-Mead first as it doesn't require gradients
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        if time.time() - start_time < time_limit:
            new_points = res.x.reshape(N, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
    except Exception:
        pass
    
    # Try Powell if time remains
    if time.time() - start_time < time_limit - 5:
        try:
            res = minimize(objective, best_points.flatten(), method='Powell',
                           options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6})
            new_points = res.x.reshape(N, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
        except Exception:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_points = None
    best_score = 0.0
    
    # Multi-start strategy
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > deadline - 30:  # Need time for refinement
            break
            
        # Initialize
        points = _init_points(seed=i)
        
        # Quick coordinate descent refinement (fast local search)
        # Give each start a fraction of the time
        time_per_start_cd = 2.0
        points, score = _refine_coordinate_descent(points, time_per_start_cd, np.random.RandomState(i+100))
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
        # If we have time, do a more expensive SciPy refinement on the best candidates
        if time.time() > deadline - 20:
            break
            
    # Final high-precision polish on the best point found so far
    if best_points is not None and time.time() < deadline - 10:
        # Use remaining time for SciPy refinement
        remaining_time = deadline - time.time() - 5
        best_points, best_score = _refine_scipy(best_points, remaining_time)
        
    # Ensure we return a valid array
    if best_points is None:
        # Fallback: return a regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
