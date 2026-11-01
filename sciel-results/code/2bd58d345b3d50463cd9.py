import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    # Vectorized: for each pair (i,j), compute cross product with all other points
    for i in range(n):
        for j in range(i + 1, n):
            # Cross product of (pts[j]-pts[i]) with (pts[k]-pts[i]) for all k
            v1 = pts[j] - pts[i]
            v2 = pts - pts[i]
            cross = v2[:, 0] * v1[1] - v2[:, 1] * v1[0]
            # Exclude i and j themselves (they give 0)
            cross[i] = np.inf
            cross[j] = np.inf
            local_min = np.min(np.abs(cross)) * 0.5
            if local_min < min_area:
                min_area = local_min
    return min_area

def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-12:
            return 0.0
        min_area = _min_triangle_area(pts)
        return min_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Strategy: Use differential evolution to optimize point positions
    # Fix p0 at origin and p1 at (1,0) to remove translation/rotation invariance
    # This reduces variables from 2N to 2*(N-2) = 24
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def negative_score(x):
        pts = get_points(x)
        s = _score(pts)
        return -s
    
    # Bounds for the free points: roughly in [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * (N - 2))
    
    # Run differential evolution with time budget
    de_kwargs = {
        'func': negative_score,
        'bounds': bounds,
        'maxiter': 50,
        'popsize': 15,
        'tol': 1e-6,
        'mutation': (0.5, 1.0),
        'recombination': 0.7,
        'seed': 42,
        'polish': False,
    }
    
    try:
        result = differential_evolution(**de_kwargs)
        if result.success or result.fun < 0:
            best_pts = get_points(result.x)
            best_score = -result.fun
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        if best_pts is not None:
            return best_pts
        return np.array([[0.0, 0.0]] * N)
    
    # Polish the best result with local optimization
    if best_pts is not None:
        x0 = best_pts[2:].flatten()
        
        def neg_score_polish(x):
            pts = get_points(x)
            return -_score(pts)
        
        try:
            res = minimize(neg_score_polish, x0, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < -best_score:
                best_pts = get_points(res.x)
                best_score = -res.fun
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    elapsed = time.time() - start_time
    rng = np.random.RandomState(42)
    while elapsed < deadline - 20:
        # Random initialization near regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.2, 0.2, N)
        radius = 1.0 + rng.uniform(-0.1, 0.1, N)
        pts = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        # Fix p0 and p1
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
        
        # Local polish
        x0 = pts[2:].flatten()
        try:
            res = minimize(neg_score_polish, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            if res.fun < -best_score:
                best_pts = get_points(res.x)
                best_score = -res.fun
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
    
    if best_pts is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_pts
