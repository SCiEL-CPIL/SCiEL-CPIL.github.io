import numpy as np

N = 26

def _valid(arr, eps=1e-9):
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0) or np.any(x < -eps) or np.any(y < -eps) or np.any(x + r > 1 + eps) or np.any(y + r > 1 + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - eps:
                return False
    return True

def _score(arr):
    if not _valid(arr):
        return -np.inf
    return float(np.sum(arr[:, 2]))

def _greedy_place(rng):
    """
    Greedy placement of circles.
    Returns array of shape (N, 3).
    """
    # Candidate grid
    grid_size = 50
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    # Add some random jitter to break symmetry? Or just use grid.
    # Let's use a fine grid.
    candidates_x, candidates_y = np.meshgrid(xs, ys)
    candidates = np.column_stack([candidates_x.ravel(), candidates_y.ravel()])
    
    # Shuffle candidates to avoid bias towards corners? 
    # Actually, for packing, corners are good. 
    # Let's keep deterministic order but maybe sort by distance to center?
    # No, let's just iterate.
    
    circles = [] # list of (x, y, r)
    
    for _ in range(N):
        best_idx = -1
        best_r = 0.0
        for i in range(len(candidates)):
            cx, cy = candidates[i]
            # Check walls
            r_wall = min(cx, 1-cx, cy, 1-cy)
            if r_wall <= 0:
                continue
            
            # Check existing circles
            r_min = r_wall
            for (px, py, pr) in circles:
                d = np.hypot(cx - px, cy - py)
                r_avail = d - pr
                if r_avail < r_min:
                    r_min = r_avail
                    if r_min <= 0:
                        break
            
            if r_min > best_r:
                best_r = r_min
                best_idx = i
        
        if best_idx == -1:
            # Should not happen for N=26 in unit square
            break
            
        cx, cy = candidates[best_idx]
        circles.append((cx, cy, best_r))
        
    arr = np.array(circles)
    return arr

def _optimize(arr, iters=1000):
    """
    Local optimization: try to grow radii and resolve overlaps.
    """
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    n = len(r)
    
    for it in range(iters):
        # Try to grow
        old_r = r.copy()
        r *= 1.001
        
        # Check validity
        valid = True
        for i in range(n):
            if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                valid = False
                break
            for j in range(i + 1, n):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < r[i] + r[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            continue # Success, try to grow more
        
        # If invalid, resolve overlaps by pushing apart
        # Simple relaxation step
        for _ in range(50):
            moved = False
            # Pairwise
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    if d < 1e-12:
                        dx, dy, d = 1e-8, 0, 1e-8
                    need = r[i] + r[j]
                    if d < need:
                        push = (need - d) / 2.0
                        ux, uy = dx/d, dy/d
                        x[i] += push * ux
                        y[i] += push * uy
                        x[j] -= push * ux
                        y[j] -= push * uy
                        moved = True
            # Walls
            for i in range(n):
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                elif x[i] > 1 - r[i]:
                    x[i] = 1 - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                elif y[i] > 1 - r[i]:
                    y[i] = 1 - r[i]
                    moved = True
            
            if not moved:
                break
        
        # Check if we can keep the growth
        if not _valid(np.column_stack([x, y, r]), eps=1e-8):
            r = old_r
        else:
            # We resolved it, keep positions
            pass
            
    arr[:, 0] = x
    arr[:, 1] = y
    arr[:, 2] = r
    return arr

def construct():
    best_arr = None
    best_score = -np.inf
    rng = np.random.default_rng(42)
    
    # Run greedy placement multiple times with different grid shuffles?
    # Greedy is deterministic if grid is fixed.
    # To add variety, we can shuffle the candidate list order.
    
    for restart in range(10):
        # Create candidates
        grid_size = 40 + restart # vary resolution?
        xs = np.linspace(0, 1, grid_size)
        ys = np.linspace(0, 1, grid_size)
        candidates_x, candidates_y = np.meshgrid(xs, ys)
        candidates = np.column_stack([candidates_x.ravel(), candidates_y.ravel()])
        
        # Shuffle candidates to break symmetry bias
        idx = rng.permutation(len(candidates))
        candidates = candidates[idx]
        
        circles = []
        for _ in range(N):
            best_idx = -1
            best_r = 0.0
            for i in range(len(candidates)):
                cx, cy = candidates[i]
                r_wall = min(cx, 1-cx, cy, 1-cy)
                if r_wall <= 0:
                    continue
                
                r_min = r_wall
                for (px, py, pr) in circles:
                    d = np.hypot(cx - px, cy - py)
                    r_avail = d - pr
                    if r_avail < r_min:
                        r_min = r_avail
                        if r_min <= 0:
                            break
                
                if r_min > best_r:
                    best_r = r_min
                    best_idx = i
            
            if best_idx == -1:
                break
                
            cx, cy = candidates[best_idx]
            circles.append((cx, cy, best_r))
            
        if len(circles) < N:
            continue
            
        arr = np.array(circles)
        
        # Optimize
        arr = _optimize(arr, iters=500)
        
        sc = _score(arr)
        if sc > best_score:
            best_score = sc
            best_arr = arr.copy()
            
    if best_arr is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
        
    return best_arr
