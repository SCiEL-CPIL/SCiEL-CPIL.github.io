"""Circle packing in perimeter-4 rectangle: Jiggle Local Search with Aspect Ratio Optimization."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Vectorized distance check for speed
    # Construct upper triangle indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = cx[i_idx] - cx[j_idx]
    dy = cy[i_idx] - cy[j_idx]
    dist_sq = dx * dx + dy * dy
    sum_r = r[i_idx] + r[j_idx]
    if np.any(dist_sq < sum_r ** 2 - tol):
        return False
    return True


def _max_radius_for_circle(i, cx, cy, r, W, H):
    """Calculate the maximum radius circle i can have given its center and others."""
    # Constraint from walls
    r_max = min(cx[i], W - cx[i], cy[i], H - cy[i])
    
    # Constraint from other circles
    for j in range(N):
        if i == j:
            continue
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        # Distance must be >= r[i] + r[j]
        # r[i] <= d - r[j]
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
            
    return max(0.0, r_max)


def _jiggle_optimize(cx, cy, r, W, H, time_budget, rng, seed):
    """
    Jiggle local search.
    State: positions and radii.
    Objective: Maximize sum(r).
    Strategy:
    1. Try to expand each circle to its max possible radius.
    2. If a circle cannot expand (or expands very little), jiggle its position.
    3. Repeat.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    current_sum = np.sum(r)
    best_sum = current_sum
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_r = r.copy()
    
    no_improvement_count = 0
    max_no_improvement = 500
    
    # Initial expansion
    for i in range(N):
        r[i] = _max_radius_for_circle(i, cx, cy, r, W, H)
    current_sum = np.sum(r)
    best_sum = current_sum
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_r = r.copy()
    
    step_size = 0.01
    jiggle_count = 0
    
    while time.time() < time_budget:
        improved = False
        
        # Try to expand all circles
        for i in range(N):
            new_r = _max_radius_for_circle(i, cx, cy, r, W, H)
            if new_r > r[i] + 1e-8:
                r[i] = new_r
                improved = True
                current_sum += new_r - r[i] # Wait, r[i] was updated? No, use old value logic carefully
                # Actually, just recalculate sum at the end of the loop or track delta
                # Let's just track delta
                # delta = new_r - r[i]
                # current_sum += delta
                # r[i] = new_r
                # But we need to be careful. Let's just do:
                pass # The loop above updates r[i] in place. We should calculate sum properly.
        
        # Recalculate sum properly to avoid drift
        current_sum = np.sum(r)
        
        if current_sum > best_sum + 1e-8:
            best_sum = current_sum
            best_cx = cx.copy()
            best_cy = cy.copy()
            best_r = r.copy()
            no_improvement_count = 0
            jiggle_count = 0
        else:
            no_improvement_count += 1
            jiggle_count += 1
            
        if no_improvement_count > max_no_improvement:
            break
            
        # Jiggle a random circle
        i = rng.integers(0, N)
        
        # Determine jiggle direction and magnitude
        # Smaller jiggle if we are close to optimum?
        mag = step_size * (1.0 / (1.0 + jiggle_count * 0.01))
        
        dx = rng.normal(0, mag)
        dy = rng.normal(0, mag)
        
        new_x = cx[i] + dx
        new_y = cy[i] + dy
        
        # Ensure new position is within walls (with current radius)
        # If it goes out, we might need to shrink radius or reject move
        # For simplicity, reject if out of bounds
        if new_x < r[i] or new_x > W - r[i] or new_y < r[i] or new_y > H - r[i]:
            # Try to adjust radius to fit? Or just reject?
            # Rejecting is safer for validity.
            continue
            
        # Check collisions with other circles at new position
        valid = True
        for j in range(N):
            if i == j:
                continue
            ddx = new_x - cx[j]
            ddy = new_y - cy[j]
            if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                valid = False
                break
                
        if not valid:
            continue
            
        # Apply move
        cx[i] = new_x
        cy[i] = new_y
        
        # Try to expand this circle again immediately?
        # The expansion loop at the top of the while handles this.
        
    return best_cx, best_cy, best_r, best_sum


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_value = 0.0
    best_result = None
    best_w = 1.0
    
    rng = np.random.default_rng(42)
    
    # Strategy: Search over aspect ratios
    # Perimeter 4 => W + H = 2.
    # Let W = 1 + alpha, H = 1 - alpha.
    # Alpha in [-0.5, 0.5] roughly.
    # We will try a grid of alphas and then refine.
    
    alphas = np.linspace(-0.4, 0.4, 20)
    
    for alpha in alphas:
        W = 1.0 + alpha
        H = 1.0 - alpha
        if W <= 0.1 or H <= 0.1:
            continue
            
        # Initialize in a grid
        # Number of rows/cols based on aspect ratio
        # Approximate square grid
        n_cols = int(np.round(np.sqrt(N * W / H)))
        if n_cols < 1: n_cols = 1
        if n_cols > N: n_cols = N
        n_rows = int(np.ceil(N / n_cols))
        
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row = i // n_cols
            col = i % n_cols
            # Center of cell
            cx[i] = (col + 0.5) * (W / n_cols)
            cy[i] = (row + 0.5) * (H / n_rows)
            
        # Initial radius: half the min cell dimension, scaled down
        cell_w = W / n_cols
        cell_h = H / n_rows
        init_r = 0.4 * min(cell_w, cell_h)
        r[:] = init_r
        
        # Check validity
        if not _is_valid(cx, cy, r, W, H):
            # Shrink if invalid
            r *= 0.5
            if not _is_valid(cx, cy, r, W, H):
                continue
                
        # Run jiggle optimization
        # Allocate time based on number of alphas
        time_per_alpha = 5.0 # 20 alphas * 5s = 100s. Leaves 15s for final polish.
        t_end = start_time + time_per_alpha
        
        cx_opt, cy_opt, r_opt, val_opt = _jiggle_optimize(cx, cy, r, W, H, t_end, rng, int(alpha*1000))
        
        if val_opt > best_value:
            best_value = val_opt
            best_result = np.column_stack((cx_opt, cy_opt, r_opt))
            best_w = W
            
    # Final Polish: Run a longer jiggle on the best result found
    if best_result is not None and time.time() < deadline - 5:
        cx = best_result[:, 0].copy()
        cy = best_result[:, 1].copy()
        r = best_result[:, 2].copy()
        W = best_w
        H = 2.0 - W
        
        # Use a tighter time budget for the final push
        cx_opt, cy_opt, r_opt, val_opt = _jiggle_optimize(cx, cy, r, W, H, deadline - 2, rng, 999)
        
        if val_opt > best_value:
            best_value = val_opt
            best_result = np.column_stack((cx_opt, cy_opt, r_opt))
            best_w = W
            
    # Fallback
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.4 / side
            cells.append([x, y, r])
        return np.array(cells)
        
    # Final safety check and shrink
    x = best_result[:, 0]
    y = best_result[:, 1]
    r = best_result[:, 2]
    w = best_w
    h = 2.0 - w
    
    if not _is_valid(x, y, r, w, h):
        # Shrink radii until valid
        scale = 0.99
        while not _is_valid(x, y, r * scale, w, h) and scale > 0.5:
            scale -= 0.01
        r = r * scale
        if not _is_valid(x, y, r, w, h):
             # Last resort: return small circles
             r = np.full(N, 0.01)
             x = np.full(N, 0.1)
             y = np.full(N, 0.1)
             # This might still be invalid if they overlap.
             # Let's just return the grid fallback if this happens.
             side = int(np.ceil(np.sqrt(N)))
             cells = []
             for i in range(N):
                 row, col = divmod(i, side)
                 xx = (col + 0.5) / side
                 yy = (row + 0.5) / side
                 rr = 0.4 / side
                 cells.append([xx, yy, rr])
             return np.array(cells)

    return np.column_stack((x, y, r))
