"""Circle packing in perimeter-4 rectangle: Simulated Annealing with hard constraints."""
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
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 100:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def _sa_optimize(x, y, r, w, h, start_time, deadline, rng):
    """Simulated Annealing to improve the packing."""
    best_sum = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy(), w)
    
    T = 0.05
    T_min = 0.0001
    alpha = 0.999
    
    x_curr = x.copy()
    y_curr = y.copy()
    r_curr = r.copy()
    w_curr = w
    h_curr = h
    curr_sum = np.sum(r_curr)
    
    while time.time() < deadline - 5 and T > T_min:
        # Random move
        move_type = rng.integers(0, 3)
        
        if move_type == 0:
            # Move a circle
            i = rng.integers(0, N)
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            nx = x_curr[i] + dx
            ny = y_curr[i] + dy
            
            # Check bounds
            if nx < r_curr[i] or nx > w_curr - r_curr[i] or ny < r_curr[i] or ny > h_curr - r_curr[i]:
                T *= alpha
                continue
                
            # Check overlaps
            valid = True
            for j in range(N):
                if i == j:
                    continue
                ddx = nx - x_curr[j]
                ddy = ny - y_curr[j]
                if ddx*ddx + ddy*ddy < (r_curr[i] + r_curr[j])**2 - 1e-9:
                    valid = False
                    break
                    
            if not valid:
                T *= alpha
                continue
            
            # Accept move (position change doesn't change sum directly, but allows future expansion)
            # We accept if it doesn't violate constraints. We don't have an objective change for pure moves.
            # So we just accept it to explore.
            x_curr[i] = nx
            y_curr[i] = ny
            
        elif move_type == 1:
            # Change radius of a circle
            i = rng.integers(0, N)
            dr = rng.normal(0, 0.01)
            new_r = r_curr[i] + dr
            if new_r < 0.01:
                new_r = 0.01
                
            # Check if new radius is valid
            if new_r > x_curr[i] or new_r > w_curr - x_curr[i] or new_r > y_curr[i] or new_r > h_curr - y_curr[i]:
                T *= alpha
                continue
                
            valid = True
            for j in range(N):
                if i == j:
                    continue
                ddx = x_curr[i] - x_curr[j]
                ddy = y_curr[i] - y_curr[j]
                if ddx*ddx + ddy*ddy < (new_r + r_curr[j])**2 - 1e-9:
                    valid = False
                    break
                    
            if not valid:
                T *= alpha
                continue
            
            # Calculate delta in objective (sum of radii)
            delta = new_r - r_curr[i]
            
            if delta > 0:
                # Always accept improvement
                r_curr[i] = new_r
                curr_sum += delta
            else:
                # Accept worse move with probability exp(delta/T)
                if rng.random() < np.exp(delta / T):
                    r_curr[i] = new_r
                    curr_sum += delta
            
        else:
            # Change width
            dw = rng.normal(0, 0.01)
            new_w = w_curr + dw
            if new_w < 0.5 or new_w > 1.5:
                T *= alpha
                continue
            new_h = 2.0 - new_w
            if new_h < 0.5:
                T *= alpha
                continue
                
            # Check if all circles fit in new box
            valid = True
            for i in range(N):
                if x_curr[i] - r_curr[i] < -1e-9 or x_curr[i] + r_curr[i] > new_w + 1e-9:
                    valid = False
                    break
                if y_curr[i] - r_curr[i] < -1e-9 or y_curr[i] + r_curr[i] > new_h + 1e-9:
                    valid = False
                    break
            if not valid:
                T *= alpha
                continue
            
            # Objective doesn't change directly with W, but feasibility might
            # We accept if valid.
            w_curr = new_w
            h_curr = new_h
            
        # Try to expand all circles greedily after each move? 
        # This is expensive but ensures we capture gains.
        # To save time, we only do full greedy expansion occasionally or at the end.
        # For now, we just check if current sum is better than best.
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_state = (x_curr.copy(), y_curr.copy(), r_curr.copy(), w_curr)
            
        T *= alpha
        
    return best_state


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h):
        if _is_valid(x, y, r, w, h):
            return True, float(np.sum(r))
        return False, 0.0

    rng = np.random.default_rng(42)
    
    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # Grid-based initial guesses with various aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.65]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.42
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0))
        
        # Random starts
        for _ in range(15):
            w0 = rng.uniform(0.7, 1.3)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0))
        
        return guesses

    guesses = get_initial_guesses()

    for i, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 10:
            break
        
        h0 = 2.0 - w0
        
        # 1. Greedy expand initial guess
        x_g, y_g, r_g = _greedy_expand(x0, y0, r0, w0, h0, deadline - 5)
        if _is_valid(x_g, y_g, r_g, w0, h0):
            val = float(np.sum(r_g))
            if val > best_value:
                best_value = val
                best_result = (x_g.copy(), y_g.copy(), r_g.copy(), w0)
        
        # 2. Simulated Annealing
        if time.time() < deadline - 5:
            x_best, y_best, r_best, w_best = _sa_optimize(
                x_g, y_g, r_g, w0, h0, start_time, deadline, rng
            )
            if _is_valid(x_best, y_best, r_best, w_best, 2.0 - w_best):
                val = float(np.sum(r_best))
                if val > best_value:
                    best_value = val
                    best_result = (x_best.copy(), y_best.copy(), r_best.copy(), w_best)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w = best_result
    h = 2.0 - w

    # Final Polish: Greedy expansion
    if time.time() < deadline - 2:
        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 1)
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val

    # Format output
    # Final safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
