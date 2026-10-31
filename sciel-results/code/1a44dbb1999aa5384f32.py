"""Circle packing in unit square: Stochastic Local Search with Overlap Resolution."""
import numpy as np

N = 26


def _valid(arr, eps=1e-9):
    """Check that all circles are inside [0,1]^2 and have disjoint interiors."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0) or np.any(x < -eps) or np.any(y < -eps) or np.any(x + r > 1 + eps) or np.any(y + r > 1 + eps):
        return False
    # pairwise distance check
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - eps:
                return False
    return True


def _score(arr):
    """Sum of radii, only if valid; else -inf."""
    if not _valid(arr):
        return -np.inf
    return float(np.sum(arr[:, 2]))


def _resolve_overlaps(x, y, r, iters=50):
    """
    Deterministically resolve overlaps by pushing circles apart.
    Returns updated x, y, r (radii unchanged).
    """
    n = len(r)
    for _ in range(iters):
        moved = False
        # Pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    if d < 1e-12:
                        # Random push if overlapping exactly
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx, dy = np.cos(angle), np.sin(angle)
                        d = 1.0
                    else:
                        dx /= d
                        dy /= d
                    # Push apart by half the overlap each
                    push = (min_d - d) / 2.0 + 1e-6
                    x[i] += push * dx
                    y[i] += push * dy
                    x[j] -= push * dx
                    y[j] -= push * dy
                    moved = True
        
        # Wall violations
        for i in range(n):
            if x[i] < r[i]:
                x[i] = r[i]
                moved = True
            elif x[i] > 1.0 - r[i]:
                x[i] = 1.0 - r[i]
                moved = True
            if y[i] < r[i]:
                y[i] = r[i]
                moved = True
            elif y[i] > 1.0 - r[i]:
                y[i] = 1.0 - r[i]
                moved = True
        
        if not moved:
            break
            
    return x, y, r


def construct():
    best_arr = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    n_restarts = 50
    max_iterations = 2000
    
    for restart in range(n_restarts):
        # Initialize with a grid-like pattern to start valid
        # Use a simple grid for initialization
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                # Add small random jitter to break symmetry
                cx += rng.uniform(-0.1, 0.1) / side
                cy += rng.uniform(-0.1, 0.1) / side
                cells.append([cx, cy, 0.01])
                idx += 1
        
        arr = np.array(cells[:N])
        x = arr[:, 0].copy()
        y = arr[:, 1].copy()
        r = arr[:, 2].copy()
        
        # Ensure initial validity
        x, y, r = _resolve_overlaps(x, y, r, iters=100)
        arr = np.column_stack([x, y, r])
        
        current_score = _score(arr)
        
        # Stochastic Local Search
        for it in range(max_iterations):
            # Cooling schedule for jitter magnitude
            temp = 0.01 * (1.0 - it / max_iterations) + 0.001
            
            # 1. Try to grow radii
            # Grow all radii by a small amount
            grow_step = 0.0005 * (1.0 - it / max_iterations) + 0.0001
            r_new = r + grow_step
            
            # Check if growing is possible without moving
            # If valid, accept
            if _valid(np.column_stack([x, y, r_new])):
                r = r_new
                current_score += grow_step * N
            else:
                # 2. If not valid, try to move circles to make room
                # Pick a random circle to jitter
                idx = rng.integers(N)
                dx = rng.normal(0, temp)
                dy = rng.normal(0, temp)
                
                x_new = x.copy()
                y_new = y.copy()
                x_new[idx] += dx
                y_new[idx] += dy
                
                # Resolve overlaps after move
                x_res, y_res, _ = _resolve_overlaps(x_new, y_new, r, iters=20)
                
                # Check if we can now grow radii
                # Try to grow again
                r_try = r + grow_step
                if _valid(np.column_stack([x_res, y_res, r_try])):
                    x = x_res
                    y = y_res
                    r = r_try
                    current_score += grow_step * N
                else:
                    # If we couldn't grow, just accept the position change if it's valid
                    # and doesn't decrease score (score is fixed if r doesn't change)
                    # But we want to escape local minima, so accept position change
                    # even if score doesn't increase immediately, hoping it enables future growth
                    if _valid(np.column_stack([x_res, y_res, r])):
                        x = x_res
                        y = y_res
                        # Score remains same (sum of r)
                    
            # Update arr
            arr = np.column_stack([x, y, r])
            
            # Track best
            sc = _score(arr)
            if sc > best_score:
                best_score = sc
                best_arr = arr.copy()
                
            # Early termination if stuck for a long time? 
            # The jittering should handle it.
            
    if best_arr is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
        
    return best_arr
