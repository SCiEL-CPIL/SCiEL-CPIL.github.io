"""Circle packing in perimeter-4 rectangle: Rigorous grow-then-slide local search."""
import numpy as np
import time

N = 21


def _check_valid(cx, cy, r, W, H, tol=1e-9):
    """Strict validity check."""
    if W <= 1e-6 or H <= 1e-6:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d_sq = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d_sq < r_sum * r_sum - tol:
                return False
    return True


def _max_radius_for_i(cx, cy, r, W, H, i):
    """
    Compute the maximum possible radius for circle i, given all other circles fixed
    and the rectangle walls fixed.
    Returns (r_max, limiter_type, limiter_index)
    limiter_type: 'left','right','bottom','top', or 'circle'
    limiter_index: j if 'circle', else -1
    """
    x, y = cx[i], cy[i]
    # Wall limits
    r_left = x
    r_right = W - x
    r_bot = y
    r_top = H - y
    r_max = min(r_left, r_right, r_bot, r_top)
    
    if r_left == r_max:
        limiter = ('left', -1)
    elif r_right == r_max:
        limiter = ('right', -1)
    elif r_bot == r_max:
        limiter = ('bottom', -1)
    else:
        limiter = ('top', -1)

    for j in range(N):
        if i == j:
            continue
        dx = x - cx[j]
        dy = y - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
            limiter = ('circle', j)
    
    return max(0.0, r_max), limiter


def _slide_circle(cx, cy, r, W, H, i, step=0.005, time_limit=None, start_time=None):
    """
    Attempt to increase r[i] by moving cx[i], cy[i] towards the wall/neighbor that limits it.
    Returns True if a valid move was found and applied.
    """
    if time_limit is not None and time.time() - start_time > time_limit:
        return False

    r_max, limiter = _max_radius_for_i(cx, cy, r, W, H, i)
    
    # If already at max and no room, try small random nudges to escape local minima
    # But first, try directed slide
    if limiter[0] == 'left':
        # Move right to increase distance from left wall
        dx, dy = step, 0.0
    elif limiter[0] == 'right':
        # Move left
        dx, dy = -step, 0.0
    elif limiter[0] == 'bottom':
        # Move up
        dx, dy = 0.0, step
    elif limiter[0] == 'top':
        # Move down
        dx, dy = 0.0, -step
    else:
        # Limiter is circle j
        j = limiter[1]
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        if d < 1e-12:
            # Random direction
            ang = np.random.uniform(0, 2 * np.pi)
            dx, dy = np.cos(ang) * step, np.sin(ang) * step
        else:
            # Move away from j
            dx = (dx / d) * step
            dy = (dy / d) * step

    new_x = cx[i] + dx
    new_y = cy[i] + dy
    
    # Check if new position is valid with current radii
    old_x, old_y = cx[i], cy[i]
    cx[i], cy[i] = new_x, new_y
    
    if _check_valid(cx, cy, r, W, H):
        # Try to grow radius i now
        new_r_max, _ = _max_radius_for_i(cx, cy, r, W, H, i)
        if new_r_max > r[i] + 1e-10:
            r[i] = new_r_max
            return True
        else:
            # Position change was valid but didn't help radius, keep it? 
            # Usually keep to explore, but to be safe revert if no gain
            cx[i], cy[i] = old_x, old_y
            return False
    else:
        cx[i], cy[i] = old_x, old_y
        return False


def _local_search(cx, cy, r, W, H, time_budget):
    """
    Iterative local search:
    1. Grow all circles to max.
    2. For each circle, try to slide to increase its max radius.
    3. Repeat until no improvement or time out.
    """
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        # Grow phase
        for i in range(N):
            if time.time() - start > time_budget:
                break
            r_max, _ = _max_radius_for_i(cx, cy, r, W, H, i)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        
        # Slide phase
        for i in range(N):
            if time.time() - start > time_budget:
                break
            # Try multiple random steps/directions if directed fails
            for _ in range(5):
                if _slide_circle(cx, cy, r, W, H, i, step=np.random.uniform(0.001, 0.02),
                                 time_limit=time_budget - (time.time() - start),
                                 start_time=start):
                    improved = True
                    break
    return cx, cy, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_value = 0.0
    best_pack = None

    def _update_best(cx, cy, r, W, H):
        nonlocal best_value, best_pack
        if _check_valid(cx, cy, r, W, H):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_pack = (cx.copy(), cy.copy(), r.copy(), W, H)

    # --- Initial Seed: Grid ---
    # Perimeter 4 => 2*(W+H)=4 => W+H=2.
    # Try a few aspect ratios.
    aspect_ratios = [1.0, 1.2, 1.5, 0.8, 1.3, 1.1]
    
    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        # Grid packing
        # 21 circles. 5x5 grid = 25 slots. Use 21.
        # Or 4x6 = 24 slots.
        # Let's try 5x5 grid, centered.
        cols, rows = 5, 5
        cw = W / cols
        ch = H / rows
        
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                cx[count] = (i + 0.5) * cw
                cy[count] = (j + 0.5) * ch
                r[count] = min(cw, ch) * 0.49  # Slightly less than half cell to allow growth
                count += 1
        
        # Ensure valid start
        if _check_valid(cx, cy, r, W, H):
            _update_best(cx, cy, r, W, H)
            
            # Local Search
            time_left = deadline - time.time() - 10
            if time_left > 5:
                cx_ls, cy_ls, r_ls = _local_search(cx.copy(), cy.copy(), r.copy(), W, H, time_left)
                _update_best(cx_ls, cy_ls, r_ls, W, H)

    # --- Perturbation & Restart ---
    if best_pack is not None:
        cx0, cy0, r0, W0, H0 = best_pack
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb positions and radii slightly
            cx = cx0.copy()
            cy = cy0.copy()
            r = r0.copy()
            W = W0
            H = H0
            
            # Random small shifts
            for i in range(N):
                cx[i] += rng.uniform(-0.01, 0.01)
                cy[i] += rng.uniform(-0.01, 0.01)
                r[i] *= rng.uniform(0.95, 1.0)
            
            # Project back into bounds and shrink if overlapping
            # Simple repair: shrink radii until valid
            for _ in range(100):
                if _check_valid(cx, cy, r, W, H):
                    break
                r *= 0.99
            
            if _check_valid(cx, cy, r, W, H):
                _update_best(cx, cy, r, W, H)
                
                time_left = deadline - time.time() - 5
                if time_left > 2:
                    cx_ls, cy_ls, r_ls = _local_search(cx.copy(), cy.copy(), r.copy(), W, H, time_left)
                    _update_best(cx_ls, cy_ls, r_ls, W, H)

    # --- Final Polish ---
    if best_pack is not None:
        cx, cy, r, W, H = best_pack
        time_left = deadline - time.time() - 2
        if time_left > 1:
            cx_ls, cy_ls, r_ls = _local_search(cx.copy(), cy.copy(), r.copy(), W, H, time_left)
            _update_best(cx_ls, cy_ls, r_ls, W, H)

    if best_pack is None:
        # Fallback: simple grid in square
        W, H = 1.0, 1.0
        cols, rows = 5, 5
        cw, ch = W / cols, H / rows
        cx = np.array([(i + 0.5) * cw for i in range(cols) for j in range(rows)][:N])
        cy = np.array([(j + 0.5) * ch for i in range(cols) for j in range(rows)][:N])
        r = np.full(N, min(cw, ch) * 0.49)
        return np.column_stack((cx, cy, r))

    cx, cy, r, W, H = best_pack
    # Safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-10)
        if _check_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break

    return np.column_stack((cx, cy, r))
