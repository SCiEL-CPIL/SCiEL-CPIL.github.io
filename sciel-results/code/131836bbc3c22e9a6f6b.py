"""Circle packing in perimeter-4 rectangle: Force-based optimization + surrogate energy."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
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


def _energy_and_grad(cx, cy, r, W, H):
    """Compute energy and gradient for force-based optimization.
    Energy penalizes overlaps and wall violations.
    Returns (energy, grad_x, grad_y) where grad points in direction to reduce energy.
    """
    energy = 0.0
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    # Wall penalties: smooth penalty when circle crosses wall
    wall_eps = 1e-4
    for i in range(N):
        # Left wall
        v = r[i] - cx[i]
        if v > 0:
            energy += v * v
            grad_x[i] += 2 * v
        # Right wall
        v = cx[i] + r[i] - W
        if v > 0:
            energy += v * v
            grad_x[i] -= 2 * v
        # Bottom wall
        v = r[i] - cy[i]
        if v > 0:
            energy += v * v
            grad_y[i] += 2 * v
        # Top wall
        v = cy[i] + r[i] - H
        if v > 0:
            energy += v * v
            grad_y[i] -= 2 * v
    
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            rsum = r[i] + r[j]
            if d2 < rsum * rsum:
                d = np.sqrt(max(d2, 1e-12))
                overlap = rsum - d
                # Penalty: overlap^2
                energy += overlap * overlap
                # Gradient: -2 * overlap * (dx/d, dy/d) for circle i, opposite for j
                grad_x[i] -= 2 * overlap * (dx / d)
                grad_y[i] -= 2 * overlap * (dy / d)
                grad_x[j] += 2 * overlap * (dx / d)
                grad_y[j] += 2 * overlap * (dy / d)
    
    return energy, grad_x, grad_y


def _compute_energy(cx, cy, r, W, H):
    """Compute surrogate energy: penalizes overlaps and wall proximity."""
    energy, _, _ = _energy_and_grad(cx, cy, r, W, H)
    return energy


def _grid_packing(w, h, n_circles):
    """Generate initial positions using a grid layout with uniform radii."""
    positions = []
    # Try different grid configurations
    best_config = None
    best_min_dist = 0
    
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < n_circles:
                continue
            cell_w = w / cols
            cell_h = h / rows
            # Max radius is half the minimum cell dimension
            r_est = min(cell_w, cell_h) / 2 * 0.95
            
            # Check if all circles fit
            valid = True
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                if row >= rows:
                    valid = False
                    break
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                if x - r_est < 0 or x + r_est > w or y - r_est < 0 or y + r_est > h:
                    valid = False
                    break
            
            if valid:
                # Compute minimum distance between any two circles
                min_d = float('inf')
                for i in range(n_circles):
                    row_i = i // cols
                    col_i = i % cols
                    xi = (col_i + 0.5) * cell_w
                    yi = (row_i + 0.5) * cell_h
                    for j in range(i + 1, n_circles):
                        row_j = j // cols
                        col_j = j % cols
                        xj = (col_j + 0.5) * cell_w
                        yj = (row_j + 0.5) * cell_h
                        d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                        min_d = min(min_d, d)
                
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_config = (rows, cols, r_est)
    
    if best_config is None:
        # Fallback: single row
        cols = n_circles
        cell_w = w / cols
        r_est = cell_w / 2 * 0.9
        for i in range(n_circles):
            x = (i + 0.5) * cell_w
            y = h / 2
            positions.append((x, y, r_est))
    else:
        rows, cols, r_est = best_config
        cell_w = w / cols
        cell_h = h / rows
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * cell_w
            y = (row + 0.5) * cell_h
            positions.append((x, y, r_est))
    
    return positions[:n_circles]


def _hexagonal_packing(w, h, n_circles):
    """Generate initial positions using hexagonal close packing with proper geometry."""
    positions = []
    
    # Try different row spacings to find best fit
    best_positions = None
    best_min_clearance = 0
    
    for r_frac in np.linspace(0.03, 0.15, 25):
        r_est = r_frac * min(w, h)
        
        # Hexagonal spacing: horizontal = 2r, vertical = sqrt(3)*r
        h_spacing = 2 * r_est
        v_spacing = np.sqrt(3) * r_est
        
        n_rows = int(h / v_spacing) + 1
        n_cols = int(w / h_spacing) + 1
        
        positions = []
        for row in range(n_rows):
            for col in range(n_cols):
                if len(positions) >= n_circles:
                    break
                # Offset odd rows by r
                offset = r_est if row % 2 == 1 else 0.0
                x = (col + 0.5) * h_spacing + offset
                y = (row + 0.5) * v_spacing
                
                # Check bounds
                if x - r_est >= 0 and x + r_est <= w and y - r_est >= 0 and y + r_est <= h:
                    positions.append((x, y, r_est))
        
        if len(positions) >= n_circles:
            positions = positions[:n_circles]
            # Compute minimum clearance
            min_clearance = float('inf')
            for i in range(len(positions)):
                xi, yi, ri = positions[i]
                # Wall clearance
                min_clearance = min(min_clearance, xi - ri, w - xi - ri, yi - ri, h - yi - ri)
                for j in range(i + 1, len(positions)):
                    xj, yj, rj = positions[j]
                    d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    min_clearance = min(min_clearance, d - ri - rj)
            
            if min_clearance > best_min_clearance:
                best_min_clearance = min_clearance
                best_positions = positions
    
    if best_positions is None:
        # Fallback: grid
        best_positions = []
        for r_frac in np.linspace(0.03, 0.15, 25):
            r_est = r_frac * min(w, h)
            n_cols = int(w / (2 * r_est)) + 1
            n_rows = int(h / (2 * r_est)) + 1
            positions = []
            for row in range(n_rows):
                for col in range(n_cols):
                    if len(positions) >= n_circles:
                        break
                    x = (col + 0.5) * (w / n_cols)
                    y = (row + 0.5) * (h / n_rows)
                    if x - r_est >= 0 and x + r_est <= w and y - r_est >= 0 and y + r_est <= h:
                        positions.append((x, y, r_est))
            if len(positions) >= n_circles:
                best_positions = positions[:n_circles]
                break
    
    if best_positions is None:
        # Last resort: uniform grid
        n_cols = 5
        n_rows = 5
        r_est = min(w / n_cols, h / n_rows) / 2 * 0.9
        for i in range(n_circles):
            row = i // n_cols
            col = i % n_cols
            x = (col + 0.5) * (w / n_cols)
            y = (row + 0.5) * (h / n_rows)
            best_positions.append((x, y, r_est))
    
    return best_positions[:n_circles]


def _force_optimize(x, y, r, w, h, time_budget, max_iter=5000):
    """
    Force-based gradient descent to resolve overlaps and minimize energy.
    Uses adaptive step size with line search.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    if w < 0.2 or h < 0.2:
        return None, None, None

    # Ensure valid start
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    start_time = time.time()
    
    # Adaptive learning rate
    lr = 0.05
    min_lr = 1e-6
    momentum = 0.9
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    prev_energy = _compute_energy(x, y, r, w, h)
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute energy and gradient
        energy, gx, gy = _energy_and_grad(x, y, r, w, h)
        
        # If energy is very low, we're close to valid
        if energy < 1e-10:
            break
        
        # Momentum update
        vx = momentum * vx - lr * gx
        vy = momentum * vy - lr * gy
        
        # Line search: try the step, if energy increases, reduce lr
        x_new = x + vx
        y_new = y + vy
        
        # Clamp to walls
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)
        
        new_energy = _compute_energy(x_new, y_new, r, w, h)
        
        if new_energy < energy:
            x = x_new
            y = y_new
            # Increase lr slightly if progress is good
            if new_energy < energy * 0.9:
                lr = min(lr * 1.05, 0.1)
        else:
            # Reduce lr and retry
            lr = max(lr * 0.5, min_lr)
            vx = momentum * vx - lr * gx
            vy = momentum * vy - lr * gy
            x_new = np.clip(x + vx, r, w - r)
            y_new = np.clip(y + vy, r, h - r)
            new_energy = _compute_energy(x_new, y_new, r, w, h)
            if new_energy < energy:
                x = x_new
                y = y_new
            else:
                # Very small step
                lr = max(lr * 0.5, min_lr)
                x = np.clip(x - lr * gx, r, w - r)
                y = np.clip(y - lr * gy, r, h - r)
        
        # Hard projection: resolve any remaining overlaps by pushing apart
        for _ in range(5):
            max_overlap = 0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    rsum = r[i] + r[j]
                    if d < rsum and d > 1e-10:
                        overlap = rsum - d
                        max_overlap = max(max_overlap, overlap)
                        ux = dx / d
                        uy = dy / d
                        corr = overlap * 0.5
                        x[i] += ux * corr
                        y[i] += uy * corr
                        x[j] -= ux * corr
                        y[j] -= uy * corr
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            if max_overlap < 1e-8:
                break
    
    # Final validity check
    if not _is_valid(x, y, r, w, h):
        # Try harder with more iterations
        for _ in range(200):
            if time.time() - start_time > time_budget:
                break
            energy, gx, gy = _energy_and_grad(x, y, r, w, h)
            if energy < 1e-10:
                break
            x = np.clip(x - lr * gx, r, w - r)
            y = np.clip(y - lr * gy, r, h - r)
            # Hard projection
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    rsum = r[i] + r[j]
                    if d < rsum and d > 1e-10:
                        overlap = rsum - d
                        ux = dx / d
                        uy = dy / d
                        corr = overlap * 0.5
                        x[i] += ux * corr
                        y[i] += uy * corr
                        x[j] -= ux * corr
                        y[j] -= uy * corr
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            if not _is_valid(x, y, r, w, h):
                lr = max(lr * 0.95, min_lr)
            else:
                break
    
    if not _is_valid(x, y, r, w, h):
        return None, None, None

    return x, y, r


def _expand_radii(x, y, r, w, h, time_budget):
    """
    Expand radii by:
    1. Uniform scaling with position re-optimization
    2. Individual circle expansion with local moves
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    start_time = time.time()
    
    # Phase 1: Uniform scaling with re-optimization
    scale_factor = 1.0
    for _ in range(200):
        if time.time() - start_time > time_budget:
            break
        
        new_scale = scale_factor * 1.005
        r_trial = r * new_scale
        
        # Check if valid
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
            scale_factor = new_scale
        else:
            # Try to re-optimize positions with larger radii
            x_new, y_new, r_new = _force_optimize(x, y, r_trial, w, h, 0.5)
            if x_new is not None and _is_valid(x_new, y_new, r_new, w, h):
                x = x_new
                y = y_new
                r = r_new
                scale_factor = new_scale
            else:
                break
    
    # Phase 2: Individual circle expansion
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        
        for i in range(N):
            # Max radius limited by walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall

            # Max radius limited by neighbors
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed

            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-7:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid

                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
    
    # Phase 3: Try moving circles to create more space
    if time.time() - start_time < time_budget:
        for i in range(N):
            # Try small moves in multiple directions
            best_gain = 0
            best_pos = None
            
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                for step in [0.02, 0.01, 0.005]:
                    new_x = x[i] + np.cos(angle) * step
                    new_y = y[i] + np.sin(angle) * step
                    
                    if new_x < r[i] or new_x > w - r[i]:
                        continue
                    if new_y < r[i] or new_y > h - r[i]:
                        continue
                    
                    # Check validity
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                    
                    if valid:
                        # Estimate potential gain
                        r_wall = min(new_x, w - new_x, new_y, h - new_y)
                        r_max_new = r_wall
                        for j in range(N):
                            if i == j:
                                continue
                            dx = new_x - x[j]
                            dy = new_y - y[j]
                            d = np.sqrt(dx * dx + dy * dy)
                            r_allowed = d - r[j]
                            if r_allowed < r_max_new:
                                r_max_new = r_allowed
                        gain = max(0.0, r_max_new - r[i])
                        if gain > best_gain:
                            best_gain = gain
                            best_pos = (new_x, new_y)
            
            if best_pos is not None and best_gain > 1e-5:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
    
    # Final uniform expansion
    for _ in range(500):
        if time.time() - start_time > time_budget:
            break
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None, None, None


def _optimize_aspect(x, y, r, w0, h0, time_budget):
    """
    Optimize aspect ratio (width) by trying different widths with re-optimization.
    """
    best_w = w0
    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    start_time = time.time()
    
    # Coarse search
    for w in np.linspace(0.4, 1.6, 20):
        h = 2.0 - w
        if h < 0.3 or time.time() - start_time > time_budget:
            continue
        
        # Scale positions to fit new rectangle
        scale_x = w / w0
        scale_y = h / h0
        x_new = x * scale_x
        y_new = y * scale_y
        r_new = r * min(scale_x, scale_y)
        
        # Quick validity check
        if _is_valid(x_new, y_new, r_new, w, h):
            val = np.sum(r_new)
            if val > best_val:
                best_val = val
                best_w = w
                best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
    
    # Fine search around best
    if best_w != w0:
        for w in np.linspace(best_w - 0.1, best_w + 0.1, 15):
            if w < 0.3 or w > 1.7:
                continue
            h = 2.0 - w
            if h < 0.3:
                continue
            
            scale_x = w / w0
            scale_y = h / h0
            x_new = x * scale_x
            y_new = y * scale_y
            r_new = r * min(scale_x, scale_y)
            
            if _is_valid(x_new, y_new, r_new, w, h):
                val = np.sum(r_new)
                if val > best_val:
                    best_val = val
                    best_w = w
                    best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
    
    return best_w, best_x, best_y, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    # Generate initial guesses using multiple strategies
    def get_initial_guesses():
        guesses = []
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.25, 0.7, 0.9, 1.1, 0.8, 1.2]
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            
            # Try hexagonal packing
            positions = _hexagonal_packing(w0, h0, N)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))
            
            # Try grid packing
            positions = _grid_packing(w0, h0, N)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))
        
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            x = rng.uniform(0.08, w0 - 0.08, N)
            y = rng.uniform(0.08, h0 - 0.08, N)
            r = rng.uniform(0.06, 0.12, N)
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    # Sort guesses by initial energy (lower is better)
    def guess_energy(g):
        x, y, r, w, h = g
        return _compute_energy(x, y, r, w, h)
    
    guesses.sort(key=guess_energy)

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 25:
            break

        # Step 1: Force optimization to resolve overlaps
        sim_time = min(2.0, (deadline - (time.time() - start_time)) * 0.2)
        x_sim, y_sim, r_sim = _force_optimize(x0, y0, r0, w0, h0, sim_time)
        if x_sim is None:
            continue

        # Step 2: Optimize aspect ratio
        w_opt, x_opt, y_opt, r_opt = _optimize_aspect(x_sim, y_sim, r_sim, w0, h0, 1.0)
        if abs(w_opt - w0) > 1e-6:
            h_opt = 2.0 - w_opt
            x_sim, y_sim, r_sim = x_opt, y_opt, r_opt
            w0, h0 = w_opt, h_opt

        # Step 3: Expand radii
        expand_time = min(6.0, (deadline - (time.time() - start_time)) * 0.5)
        x_exp, y_exp, r_exp = _expand_radii(x_sim, y_sim, r_sim, w0, h0, expand_time)
        if x_exp is not None:
            valid, val = check_validity(x_exp, y_exp, r_exp, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x_exp.copy(), y_exp.copy(), r_exp.copy(), w0, h0)
        else:
            valid, val = check_validity(x_sim, y_sim, r_sim, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x_sim.copy(), y_sim.copy(), r_sim.copy(), w0, h0)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        final_time = deadline - (time.time() - start_time) - 3
        x_final, y_final, r_final = _expand_radii(x_best, y_best, r_best, w_best, h_best, final_time)
        if x_final is not None:
            valid, val = check_validity(x_final, y_final, r_final, w_best, h_best)
            if valid and val > best_value:
                best_value = val
                best_result = (x_final.copy(), y_final.copy(), r_final.copy(), w_best, h_best)

    if best_result is None:
        # Fallback: simple grid in a 1x1 square
        side = 5
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x_pos = (i + 0.5) / side
                y_pos = (j + 0.5) / side
                r_pos = 0.5 / side - 0.005
                cells.append((x_pos, y_pos, r_pos))
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
