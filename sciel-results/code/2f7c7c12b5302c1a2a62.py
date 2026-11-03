"""Heilbronn-11 solver: Annealing + coordinate descent with push-apart heuristic."""
import numpy as np
import time
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140.0

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _min_area_and_indices(pts):
    """Return (min_area, indices_of_min_triangle)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    idx = np.argmin(areas)
    return areas[idx], idx

def _project_cart(pts):
    """Project points to be inside the equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def _push_apart(pts, step=0.01):
    """Push points in the smallest triangle apart to increase its area."""
    min_area, idx = _min_area_and_indices(pts)
    if min_area < 1e-15:
        return pts
    
    i, j, k = _TRIPLETS[idx]
    pi = pts[i].copy()
    pj = pts[j].copy()
    pk = pts[k].copy()
    
    # Compute centroids and push each point away from the centroid of the other two
    centroid_jk = (pj + pk) / 2
    centroid_ik = (pi + pk) / 2
    centroid_ij = (pi + pj) / 2
    
    # Determine direction to push each point
    dir_i = pi - centroid_jk
    dir_j = pj - centroid_ik
    dir_k = pk - centroid_ij
    
    # Normalize
    for d in [dir_i, dir_j, dir_k]:
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            d /= norm
    
    # Push
    pts[i] = pi + step * dir_i
    pts[j] = pj + step * dir_j
    pts[k] = pk + step * dir_k
    
    return _project_cart(pts)

def _coordinate_descent(pts, n_iter=200, step_init=0.05):
    """Simple coordinate descent: try small moves in x and y for each point."""
    best = pts.copy()
    best_area = _min_area_and_indices(best)[0]
    step = step_init
    
    for it in range(n_iter):
        improved = False
        for i in range(N):
            for dx in [-step, step]:
                for dy in [-step, step]:
                    trial = best.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    trial = _project_cart(trial)
                    area = _min_area_and_indices(trial)[0]
                    if area > best_area:
                        best = trial
                        best_area = area
                        improved = True
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
    return best, best_area

def _anneal(pts, n_iter=500, t_init=0.05, t_min=1e-5, alpha=0.99):
    """Simulated annealing with push-apart and coordinate descent."""
    best = pts.copy()
    best_area = _min_area_and_indices(best)[0]
    curr = pts.copy()
    curr_area = _min_area_and_indices(curr)[0]
    t = t_init
    
    for it in range(n_iter):
        if t < t_min:
            break
        
        # Random perturbation
        trial = curr.copy()
        i = np.random.randint(N)
        dx = np.random.normal(0, t)
        dy = np.random.normal(0, t)
        trial[i, 0] += dx
        trial[i, 1] += dy
        trial = _project_cart(trial)
        
        # Also try push-apart
        if np.random.rand() < 0.3:
            trial = _push_apart(trial, step=t * 0.5)
        
        trial_area = _min_area_and_indices(trial)[0]
        delta = trial_area - curr_area
        
        if delta > 0 or np.random.rand() < np.exp(delta / max(t, 1e-10)):
            curr = trial
            curr_area = trial_area
            if curr_area > best_area:
                best = curr.copy()
                best_area = curr_area
        
        t *= alpha
    
    return best, best_area

def _get_initial_configurations():
    """Return list of initial configurations to try."""
    configs = []
    
    # Configuration 1: Symmetric with corners, mid-edges, and interior points
    c1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [2/3, S3H/3],
        [0.5, 2*S3H/3],
        [0.5, S3H/3],
        [0.5, S3H/6],
    ], dtype=float)
    configs.append(_project_cart(c1))
    
    # Configuration 2: Slightly different interior arrangement
    c2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [0.4, S3H/4],
        [0.6, S3H/4],
        [0.5, 3*S3H/4],
        [0.35, S3H/2],
        [0.65, S3H/2],
    ], dtype=float)
    configs.append(_project_cart(c2))
    
    # Configuration 3: More spread out
    c3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [0.3, S3H/5],
        [0.7, S3H/5],
        [0.5, 4*S3H/5],
        [0.3, 3*S3H/5],
        [0.7, 3*S3H/5],
    ], dtype=float)
    configs.append(_project_cart(c3))
    
    return configs

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_area_and_indices(pts)[0]
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    configs = _get_initial_configurations()
    
    for cfg in configs:
        _update_best(cfg)
    
    # Phase 1: Annealing from each config
    for cfg in configs:
        if not _elapsed_ok(20):
            break
        
        annealed, area = _anneal(cfg, n_iter=300, t_init=0.04, t_min=1e-4, alpha=0.995)
        _update_best(annealed)
    
    # Phase 2: Coordinate descent polish on best
    if best_pts is not None and _elapsed_ok(15):
        polished, area = _coordinate_descent(best_pts, n_iter=300, step_init=0.03)
        _update_best(polished)
    
    # Phase 3: More annealing from best
    if best_pts is not None and _elapsed_ok(10):
        annealed2, area2 = _anneal(best_pts, n_iter=400, t_init=0.02, t_min=1e-5, alpha=0.998)
        _update_best(annealed2)
    
    # Phase 4: Final coordinate descent
    if best_pts is not None and _elapsed_ok(5):
        polished2, area2 = _coordinate_descent(best_pts, n_iter=200, step_init=0.01)
        _update_best(polished2)
    
    if best_pts is None:
        best_pts = configs[0]
    
    return _project_cart(best_pts)
# EVOLVE-BLOCK-END
