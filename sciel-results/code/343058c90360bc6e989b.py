# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import itertools

_DEADLINE = time.time() + 615.0

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _de_objective(points_flat):
    """Objective for Differential Evolution: minimize -score."""
    pts = points_flat.reshape(-1, 2)
    try:
        s = _score(pts)
        if s <= 0:
            return 1.0 # Penalty for invalid/zero score
        return -s
    except Exception:
        return 1.0

def _polish(points0, max_sweeps=200, step_sizes=None):
    """High-precision coordinate descent polish."""
    if step_sizes is None:
        step_sizes = [0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
    
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 15:
            break
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_local_score = best_score
            
            for step in step_sizes:
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                               (step*0.707,step*0.707),(-step*0.707,step*0.707),
                               (step*0.707,-step*0.707),(-step*0.707,-step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local_score + 1e-13:
                        best_local_score = s
                        best_x, best_y = nx, ny
                        improved = True
            pts[i] = [best_x, best_y]
        
        cur_score = _score(pts)
        if cur_score > best_score + 1e-13:
            best_score = cur_score
            best_pts = pts.copy()
        if not improved:
            break
    
    return best_pts, best_score

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def heilbronn_convex13() -> np.ndarray:
    n = 13
    bounds = [(0.0, 1.0)] * (2 * n)
    
    best_points = None
    best_score = -np.inf

    # Strategy: Differential Evolution (Global Search) followed by Polish
    # DE is robust to non-smoothness and multi-modality.
    
    def _time_callback(xk, fun, convergence):
        if time.time() > _DEADLINE - 60:
            return True # Stop DE to allow time for polishing
        return False

    try:
        result = differential_evolution(
            _de_objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-4, 
            callback=_time_callback,
            polish=False
        )
        
        if result.x is not None:
            de_points = result.x.reshape(-1, 2)
            s = _score(de_points)
            if s > best_score:
                best_score = s
                best_points = de_points.copy()
                
    except Exception:
        pass

    # Fallback if DE fails or finds poor solution
    if best_points is None or best_score < 0.1:
        rng = np.random.default_rng(seed=42)
        for _ in range(10):
            if time.time() > _DEADLINE - 30:
                break
            start = rng.uniform(0.1, 0.9, (n, 2))
            s = _score(start)
            if s > best_score:
                best_score = s
                best_points = start.copy()

    # Final Polish
    if best_points is not None and time.time() < _DEADLINE - 10:
        final_pts, final_s = _polish(best_points, max_sweeps=300)
        if final_s > best_score:
            best_score = final_s
            best_points = final_pts

    if best_points is None:
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))

    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))

    return best_points

# EVOLVE-BLOCK-END
