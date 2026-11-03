import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
HALF = 0.5
_TIME_LIMIT = 140  # Leave 10s margin for 150s kill

def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    if np.any(np.abs(c) > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # wall distance
    wall = np.min([
        np.min(c[:, 0]) + HALF,
        HALF - np.max(c[:, 0]),
        np.min(c[:, 1]) + HALF,
        HALF - np.max(c[:, 1]),
    ])
    r = min(wall, min_pair / 2.0)
    return r if r > 0 else 0.0

def _hex_points(s, ox, oy):
    """
    Generate hexagonal lattice points with spacing s and offset (ox, oy).
    Returns points inside the unit square.
    """
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    
    # Generate grid covering [-0.5, 0.5] with margin
    margin = 0.1
    x_min, x_max = -HALF - margin, HALF + margin
    y_min, y_max = -HALF - margin, HALF + margin
    
    nx = int(np.ceil((x_max - x_min) / dx)) + 1
    ny = int(np.ceil((y_max - y_min) / dy)) + 1
    
    points = []
    for j in range(ny):
        y = y_min + j * dy + oy
        x_offset = 0.0 if j % 2 == 0 else dx / 2.0
        for i in range(nx):
            x = x_min + i * dx + x_offset + ox
            points.append([x, y])
    
    arr = np.array(points)
    # Keep only points inside the square
    mask = np.all(np.abs(arr) <= HALF, axis=1)
    return arr[mask]

def _greedy_select(cands, k=N):
    """
    Greedy farthest-point selection of k points.
    Starts with a random point, then repeatedly picks the point farthest from the current set.
    """
    n = cands.shape[0]
    if n <= k:
        return cands.copy()
    
    selected = []
    # Start with a random candidate
    idx = int(np.random.randint(n))
    selected.append(idx)
    min_dist = np.full(n, np.inf)
    
    for _ in range(k - 1):
        # Update min distance to selected set
        last = cands[selected[-1]]
        d = np.linalg.norm(cands - last, axis=1)
        min_dist = np.minimum(min_dist, d)
        # Pick the farthest
        j = int(np.argmax(min_dist))
        selected.append(j)
        
    return cands[selected]

def _targeted_refine(pts, max_iters=500, initial_step=0.005, deadline=None):
    """
    Targeted bottleneck elimination refinement.
    Focuses on the most constraining pair or wall.
    """
    pts = pts.copy()
    best_r = _score(pts)
    step = initial_step
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Identify bottleneck
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_val = d[min_pair_idx]
        
        wall_dists = np.min([
            pts[:, 0] + HALF,
            HALF - pts[:, 0],
            pts[:, 1] + HALF,
            HALF - pts[:, 1],
        ], axis=0)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_val = wall_dists[min_wall_idx]
        
        current_r = min(min_pair_val / 2.0, min_wall_val)
        
        # Determine which constraint is active
        if min_pair_val / 2.0 < min_wall_val:
            # Pair is critical
            i, j = min_pair_idx
            if i == j:
                continue
            vec = pts[i] - pts[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                # Random nudge
                pts[i] += np.random.normal(0, step, 2)
                pts[j] -= np.random.normal(0, step, 2)
            else:
                dir_vec = vec / norm
                # Move i away from j, j away from i
                new_i = pts[i] + dir_vec * step
                new_j = pts[j] - dir_vec * step
                
                # Check bounds
                if np.all(np.abs(new_i) <= HALF) and np.all(np.abs(new_j) <= HALF):
                    pts[i] = new_i
                    pts[j] = new_j
                    new_r = _score(pts)
                    if new_r > best_r:
                        best_r = new_r
                        step = min(step * 1.05, 0.02) # Increase step if improving
                    else:
                        step *= 0.8 # Decrease step if not improving
                else:
                    # If moving out of bounds, try smaller step or different direction
                    step *= 0.9
        else:
            # Wall is critical
            i = min_wall_idx
            # Determine which wall
            if pts[i, 0] + HALF == min_wall_val:
                dir_vec = np.array([1.0, 0.0])
            elif HALF - pts[i, 0] == min_wall_val:
                dir_vec = np.array([-1.0, 0.0])
            elif pts[i, 1] + HALF == min_wall_val:
                dir_vec = np.array([0.0, 1.0])
            else:
                dir_vec = np.array([0.0, -1.0])
                
            new_pt = pts[i] + dir_vec * step
            if np.all(np.abs(new_pt) <= HALF):
                pts[i] = new_pt
                new_r = _score(pts)
                if new_r > best_r:
                    best_r = new_r
                    step = min(step * 1.05, 0.02)
                else:
                    step *= 0.8
            else:
                step *= 0.9
                
        # Occasionally add a small random perturbation to escape local minima
        if it % 50 == 0:
            idx = np.random.randint(N)
            pts[idx] += np.random.normal(0, step * 0.5, 2)
            pts[idx] = np.clip(pts[idx], -HALF, HALF)
            
    return pts, best_r

def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    np.random.seed(42)
    
    # Estimate optimal spacing
    # Area per circle ~ 1/200. Hex packing density ~ 0.9069.
    # s^2 * sqrt(3)/2 ~ 1/200 => s ~ sqrt(2/(200*sqrt(3))) ~ 0.076
    s0 = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)
    
    # Multiple trials with different hex grid offsets and scales
    n_trials = 20
    
    for trial in range(n_trials):
        if time.time() > _TIME_LIMIT - 10:
            break
            
        # Randomize spacing slightly
        s = s0 * (0.95 + 0.1 * np.random.random())
        ox = np.random.uniform(-0.5, 0.5)
        oy = np.random.uniform(-0.5, 0.5)
        
        cands = _hex_points(s, ox, oy)
        if cands.shape[0] < N:
            continue
            
        # Greedy select
        pts = _greedy_select(cands, N)
        
        # Targeted refinement
        deadline = _TIME_LIMIT - 5
        if time.time() < deadline:
            pts, r = _targeted_refine(pts, max_iters=300, initial_step=0.005, deadline=deadline)
        else:
            r = _score(pts)
            
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    # Final polish on the best result if time permits
    if best_pts is not None and time.time() < _TIME_LIMIT - 2:
        final_pts, final_r = _targeted_refine(best_pts, max_iters=200, initial_step=0.002, deadline=_TIME_LIMIT - 1)
        if final_r > best_r:
            best_r = final_r
            best_pts = final_pts.copy()
            
    # Fallback
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
        
    return best_pts
