"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 21

# EVOLVE-BLOCK-START
def _check_valid(cx, cy, r, W, H):
    """Check if circles fit in rectangle [0,W]x[0,H] with disjoint interiors."""
    if W <= 0 or H <= 0:
        return False
    n = len(cx)
    # Wall constraints
    if np.any(cx - r < -1e-9) or np.any(cy - r < -1e-9):
        return False
    if np.any(cx + r > W + 1e-9) or np.any(cy + r > H + 1e-9):
        return False
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False
    return True


def _relax_positions(cx, cy, r, W, H, max_iter=5000, dt=0.002):
    """Force-directed relaxation to fix overlaps and wall violations."""
    n = len(cx)
    cx = cx.astype(float).copy()
    cy = cy.astype(float).copy()
    
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion (strong)
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 50
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 50
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 50
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 50
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d_sq < min_d * min_d:
                    d = np.sqrt(d_sq)
                    if d > 1e-12:
                        overlap = min_d - d
                        fx[i] += (dx / d) * overlap * 20
                        fy[i] += (dy / d) * overlap * 20
                        fx[j] -= (dx / d) * overlap * 20
                        fy[j] -= (dy / d) * overlap * 20
        
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to rectangle
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        
        # Check convergence
        if np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break
    
    return cx, cy


def _expand_radii(cx, cy, r, W, H, max_iter=2000, deadline=None):
    """Gradually expand radii while maintaining validity."""
    n = len(cx)
    cx = cx.astype(float).copy()
    cy = cy.astype(float).copy()
    r = r.astype(float).copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to expand the smallest radius
        min_idx = np.argmin(r)
        old_r = r[min_idx]
        
        # Binary search for max possible radius increase
        lo = 0.0
        hi = 0.01
        best_r = old_r
        
        for _ in range(10):
            mid = (lo + hi) / 2
            r_test = r.copy()
            r_test[min_idx] = old_r + mid
            
            # Check validity
            if _check_valid(cx, cy, r_test, W, H):
                lo = mid
                best_r = old_r + mid
            else:
                hi = mid
        
        if best_r <= old_r + 1e-8:
            # Can't expand this circle, try next smallest
            # If all circles can't expand, stop
            if it > 100:  # Give some iterations to try different circles
                break
            # Shuffle which circle to try
            min_idx = np.random.randint(n)
            old_r = r[min_idx]
            lo = 0.0
            hi = 0.01
            best_r = old_r
            
            for _ in range(10):
                mid = (lo + hi) / 2
                r_test = r.copy()
                r_test[min_idx] = old_r + mid
                if _check_valid(cx, cy, r_test, W, H):
                    lo = mid
                    best_r = old_r + mid
                else:
                    hi = mid
            
            if best_r <= old_r + 1e-8:
                break
        
        r[min_idx] = best_r
        
        # Re-relax positions after radius change
        cx, cy = _relax_positions(cx, cy, r, W, H, max_iter=200, dt=0.001)
    
    return cx, cy, r, float(np.sum(r))


def _grid_init(cols, W, H):
    """Generate initial grid layout."""
    rows = int(np.ceil(N / cols))
    W_cell = W / cols
    H_cell = H / rows
    
    cx = np.array([(i + 0.5) * W_cell for i in range(cols) for _ in range(rows)][:N])
    cy = np.array([(j + 0.5) * H_cell for _ in range(cols) for j in range(rows)][:N])
    r = np.full(N, min(W_cell, H_cell) / 2 * 0.85)
    
    return cx, cy, r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_val = -float('inf')
    best_result = None
    
    # Try various aspect ratios (W + H = 2, so H = 2 - W)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.45, 1.8, 0.55]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Try different grid initializations
        for grid_cols in [3, 4, 5]:
            if time.time() > deadline:
                break
            
            cx, cy, r = _grid_init(grid_cols, W, H)
            
            # Initial relaxation
            cx, cy = _relax_positions(cx, cy, r, W, H, max_iter=2000, dt=0.002)
            
            # Expand radii
            cx, cy, r, val = _expand_radii(cx, cy, r, W, H, max_iter=500, deadline=deadline)
            
            if _check_valid(cx, cy, r, W, H) and val > best_val:
                best_val = val
                best_result = (cx, cy, r, W, H)
    
    # If no valid result found, fall back to seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    cx, cy, r, W, H = best_result
    return np.column_stack([cx, cy, r])
# EVOLVE-BLOCK-END
