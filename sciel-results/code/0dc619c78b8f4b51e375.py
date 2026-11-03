"""
Heilbronn n=14: Maximize minimum triangle area / convex hull area.

Strategy:
  1. Coarse-to-fine grid search over the configuration space.
  2. Coordinate-wise local refinement focusing on "bottleneck" triangles.
  3. Use a semi-discrete search: perturb specific points based on the 
     current minimum triangle area.
  4. Explicit deadline guard.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 14
_START_TIME = time.time()
DEADLINE = 145.0  # safety margin before 150 s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

# ---------------------------------------------------------------------------
# Precompute triangle index triples for vectorized area computation
# ---------------------------------------------------------------------------
_idx = np.arange(N)
_II, _JJ, _KK = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_II < _JJ) & (_JJ < _KK)
_T_I = _II[_mask].ravel()
_T_J = _JJ[_mask].ravel()
_T_K = _KK[_mask].ravel()
N_TRI = len(_T_I)

def _all_areas(points):
    """Return array of all C(N,3) triangle areas (vectorized)."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)

def _score(points):
    """min triangle area / convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    areas = _all_areas(points)
    min_a = np.min(areas)
    if min_a < 1e-14:
        return 0.0
    return min_a / hull_area

def _get_min_tri_info(points):
    """Return min area and the indices of the triangle(s) achieving it."""
    areas = _all_areas(points)
    min_a = np.min(areas)
    # Find all triangles with area close to min
    tol = 1e-6
    mask = np.abs(areas - min_a) < tol
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        idxs = [np.argmin(areas)]
    tri_indices = []
    for i in idxs:
        tri_indices.append((_T_I[i], _T_J[i], _T_K[i]))
    return min_a, tri_indices

# ---------------------------------------------------------------------------
# Grid Search / Local Refinement
# ---------------------------------------------------------------------------

def _refine_coordinate_wise(points, max_iter=2000, step_sizes=None):
    """
    Coordinate-wise local refinement.
    Tries to move each point in small steps to improve the min triangle area.
    Focuses on points involved in the current minimum triangles.
    """
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Order points by involvement in min triangles
    _, min_tris = _get_min_tri_info(best_points)
    involved = set()
    for t in min_tris:
        involved.add(t[0])
        involved.add(t[1])
        involved.add(t[2])
    
    # Sort indices: involved first, then others
    indices = list(involved) + [i for i in range(N) if i not in involved]
    
    current_points = best_points.copy()
    current_score = best_score
    
    for step in step_sizes:
        if _time_left() < 5:
            break
        improved = True
        while improved:
            improved = False
            for i in indices:
                if _time_left() < 2:
                    break
                # Try moving point i in 4 directions
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                               (step, step), (-step, -step), (step, -step), (-step, step)]:
                    new_points = current_points.copy()
                    new_points[i, 0] += dx
                    new_points[i, 1] += dy
                    s = _score(new_points)
                    if s > current_score + 1e-10:
                        current_points = new_points
                        current_score = s
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
                        break # Restart loop for this point with new position
                if _time_left() < 2:
                    break
            if _time_left() < 5:
                break
        # After one pass with this step size, update best
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
    return best_points, best_score

def _grid_search_initial():
    """
    Generate initial points via a simple grid-based approach.
    We place points on a grid and select 14 that maximize min area.
    This is a heuristic for a good starting point.
    """
    # Try a few grid resolutions
    best_pts = None
    best_score = 0
    
    # Regular polygon as baseline
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts = base.copy()
    best_score = _score(best_pts)
    
    # Try perturbing the regular polygon
    rng = np.random.RandomState(42)
    for _ in range(50):
        if _time_left() < 10:
            break
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        cand = base + noise
        # Keep in unit circle
        norms = np.linalg.norm(cand, axis=1)
        mask = norms > 1.0
        cand[mask] /= norms[mask, None]
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand.copy()
            
    return best_pts, best_score

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    start = time.time()
    
    # Phase 1: Get a decent initial configuration
    points, score = _grid_search_initial()
    best_points = points.copy()
    best_score = score
    
    # Phase 2: Coordinate-wise refinement
    points, score = _refine_coordinate_wise(points, max_iter=5000)
    if score > best_score:
        best_score = score
        best_points = points.copy()
        
    # Phase 3: Try a few random restarts with refinement
    rng = np.random.RandomState(123)
    for _ in range(5):
        if _time_left() < 20:
            break
        # Random points in unit disk
        r = np.sqrt(rng.rand(N))
        theta = rng.rand(N) * 2 * np.pi
        cand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        # Quick refinement
        cand, s = _refine_coordinate_wise(cand, max_iter=1000)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
            
    # Normalize to unit hull area
    try:
        hull = ConvexHull(best_points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(best_points)
        hull_area = hull.volume

    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        best_points = best_points * scale

    return best_points
