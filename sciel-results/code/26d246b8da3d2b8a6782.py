"""Circle packing in perimeter-4 rectangle: Rigid-body dynamics + surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    iu = np.triu_indices(N, k=1)
    dx = cx[iu[0]] - cx[iu[1]]
    dy = cy[iu[0]] - cy[iu[1]]
    d2 = dx * dx + dy * dy
    rsum = r[iu[0]] + r[iu[1]]
    if np.any(d2 < rsum * rsum - tol):
        return False
    return True


def _compute_energy(cx, cy, r, W, H):
    """Compute surrogate energy: penalizes overlaps and wall proximity."""
    # Wall penalties
    energy = 0.0
    energy += np.sum(np.maximum(r - cx, 0.0) ** 2)
    energy += np.sum(np.maximum(cx + r - W, 0.0) ** 2)
    energy += np.sum(np.maximum(r - cy, 0.0) ** 2)
    energy += np.sum(np.maximum(cy + r - H, 0.0) ** 2)
    # Pairwise overlap penalties
    iu = np.triu_indices(N, k=1)
    dx = cx[iu[0]] - cx[iu[1]]
    dy = cy[iu[0]] - cy[iu[1]]
    d2 = dx * dx + dy * dy
    rsum = r[iu[0]] + r[iu[1]]
    d = np.sqrt(np.maximum(d2, 1e-12))
    overlap = np.maximum(rsum - d, 0.0)
    energy += np.sum(overlap * overlap)
    return energy


def _grid_packing(w, h, n_circles):
    """Generate initial positions using a grid layout with uniform radii."""
    best_config = None
    best_score = -1
    
    for rows in range(2, 10):
        for cols in range(2, 10):
            if rows * cols < n_circles:
                continue
            cell_w = w / cols
            cell_h = h / rows
            r_est = min(cell_w, cell_h) / 2 * 0.92
            if r_est < 0.01:
                continue
            # Score: larger radius is better
            score = r_est * min(rows * cols / n_circles, 1.0)
            if score > best_score:
                best_score = score
                best_config = (rows, cols, r_est)
    
    if best_config is None:
        cols = n_circles
        cell_w = w / cols
        r_est = cell_w / 2 * 0.9
        positions = []
        for i in range(n_circles):
            x = (i + 0.5) * cell_w
            y = h / 2
            positions.append((x, y, r_est))
        return positions
    
    rows, cols, r_est = best_config
    cell_w = w / cols
    cell_h = h / rows
    positions = []
    for i in range(n_circles):
        row = i // cols
        col = i % cols
        x = (col + 0.5) * cell_w
        y = (row + 0.5) * cell_h
        positions.append((x, y, r_est))
    return positions


def _hexagonal_packing(w, h, n_circles):
    """Generate initial positions using hexagonal close packing."""
    best_positions = None
    best_r = 0
    
    # Try a range of radii
    for r_trial in np.linspace(0.04, 0.2, 40):
        # Hexagonal packing: vertical spacing = r*sqrt(3), horizontal = 2r
        vert_spacing = r_trial * np.sqrt(3)
        horiz_spacing = 2.0 * r_trial
        
        rows = int((h - 2 * r_trial) / vert_spacing) + 1
        cols = int((w - 2 * r_trial) / horiz_spacing) + 1
        
        if rows < 1 or cols < 1:
            continue
        
        positions = []
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= n_circles:
                    break
                offset = r_trial if (row % 2 == 1) else 0.0
                x = r_trial + col * horiz_spacing + offset
                y = r_trial + row * vert_spacing
                if x + r_trial <= w + 1e-9 and y + r_trial <= h + 1e-9:
                    positions.append((x, y, r_trial))
        
        if len(positions) >= n_circles:
            positions = positions[:n_circles]
            if r_trial > best_r:
                best_r = r_trial
                best_positions = positions
    
    if best_positions is None:
        # Fallback: uniform grid
        return _grid_packing(w, h, n_circles)
    
    return best_positions


def _dynamics_simulate(x, y, r, w, h, time_budget, dt=0.002, max_iter=20000):
    """
    Rigid-body dynamics simulation with impulse-based collision resolution.
    Mass proportional to radius squared. Uses adaptive damping and force scaling.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    if w < 0.15 or h < 0.15:
        return None, None, None

    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = np.maximum(r * r, 1e-6)

    # Ensure valid start
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    start_time = time.time()
    
    wall_stiffness = 8000.0
    pair_stiffness = 5000.0

    iu = np.triu_indices(N, k=1)

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * wall_stiffness
        fx += right_viol * wall_stiffness
        fy -= bottom_viol * wall_stiffness
        fy += top_viol * wall_stiffness

        # Pairwise repulsion - vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * pair_stiffness
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

        # Adaptive damping: stronger when energy is high
        max_overlap = np.max(overlap) if len(overlap) > 0 else 0.0
        damping = 0.90 if max_overlap > 0.01 else 0.95

        # Update velocities
        vx += dt * fx / mass
        vy += dt * fy / mass

        # Damping
        vx *= damping
        vy *= damping

        # Update positions
        x += dt * vx
        y += dt * vy

        # Clamp to walls
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Impulse-based collision resolution (post-step correction)
        for _ in range(12):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

            if np.max(overlap) <= 1e-9:
                break

            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            m_i = mass[iu[0]]
            m_j = mass[iu[1]]
            total_m = m_i + m_j
            correction = overlap * 0.5
            frac_i = m_j / total_m
            frac_j = m_i / total_m

            dx_corr = ux * correction * frac_i
            dy_corr = uy * correction * frac_i
            np.add.at(x, iu[0], dx_corr)
            np.add.at(y, iu[0], dy_corr)
            np.add.at(x, iu[1], -ux * correction * frac_j)
            np.add.at(y, iu[1], -uy * correction * frac_j)

            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

    # Final validity check
    if not _is_valid(x, y, r, w, h):
        return None, None, None

    return x, y, r


def _expand_radii(x, y, r, w, h, time_budget):
    """
    Iteratively expand radii using binary search, moving circles to make space.
    Uses joint optimization of positions and radii.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    start_time = time.time()
    improved = True
    rounds = 0

    while improved and time.time() - start_time < time_budget and rounds < 200:
        improved = False
        rounds += 1

        # Phase 1: Try to expand each circle individually
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
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
                lo = r[i]
                hi = r_max
                for _ in range(30):
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

        # Phase 2: Try to move circles to create space
        if time.time() - start_time < time_budget:
            for i in range(N):
                best_move = None
                best_gain = 0.0
                for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                    for step in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002]:
                        new_x = x[i] + np.cos(angle) * step
                        new_y = y[i] + np.sin(angle) * step
                        if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                            continue
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
                            r_wall = min(new_x, w - new_x, new_y, h - new_y)
                            r_max = r_wall
                            for j in range(N):
                                if i == j:
                                    continue
                                dx = new_x - x[j]
                                dy = new_y - y[j]
                                d = np.sqrt(dx * dx + dy * dy)
                                r_allowed = d - r[j]
                                if r_allowed < r_max:
                                    r_max = r_allowed
                            gain = max(0.0, r_max - r[i])
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (new_x, new_y)
                if best_move is not None and best_gain > 1e-6:
                    x[i] = best_move[0]
                    y[i] = best_move[1]
                    improved = True

        # Phase 3: Pair moves
        if not improved and time.time() - start_time < time_budget:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-6:
                        continue
                    ux = dx / d
                    uy = dy / d
                    for step in [0.012, 0.006, 0.003, 0.001]:
                        new_xi = x[i] + ux * step
                        new_yi = y[i] + uy * step
                        new_xj = x[j] - ux * step
                        new_yj = y[j] - uy * step
                        if new_xi < r[i] or new_xi > w - r[i] or new_yi < r[i] or new_yi > h - r[i]:
                            continue
                        if new_xj < r[j] or new_xj > w - r[j] or new_yj < r[j] or new_yj > h - r[j]:
                            continue
                        valid = True
                        for k in range(N):
                            if k == i or k == j:
                                continue
                            dx_ik = new_xi - x[k]
                            dy_ik = new_yi - y[k]
                            if dx_ik * dx_ik + dy_ik * dy_ik < (r[i] + r[k]) ** 2 - 1e-9:
                                valid = False
                                break
                            dx_jk = new_xj - x[k]
                            dy_jk = new_yj - y[k]
                            if dx_jk * dx_jk + dy_jk * dy_jk < (r[j] + r[k]) ** 2 - 1e-9:
                                valid = False
                                break
                        if valid:
                            x_test = x.copy()
                            y_test = y.copy()
                            x_test[i] = new_xi
                            y_test[i] = new_yi
                            x_test[j] = new_xj
                            y_test[j] = new_yj
                            r_test = r.copy()
                            for _ in range(10):
                                r_trial = r_test * 1.002
                                if _is_valid(x_test, y_test, r_trial, w, h):
                                    r_test = r_trial
                                else:
                                    break
                            if np.sum(r_test) > np.sum(r) + 1e-6:
                                x[i] = new_xi
                                y[i] = new_yi
                                x[j] = new_xj
                                y[j] = new_yj
                                r = r_test
                                improved = True
                                break

    # Final uniform expansion
    for _ in range(3000):
        if time.time() - start_time > time_budget:
            break
        r_trial = r * 1.0002
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Second pass: individual expansion
    for i in range(N):
        if time.time() - start_time > time_budget:
            break
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
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
            lo = r[i]
            hi = r_max
            for _ in range(25):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-8:
                r[i] = lo

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None, None, None


def _optimize_aspect(x, y, r, w0, h0, time_budget):
    """
    Optimize aspect ratio (width) by trying different widths.
    """
    best_w = w0
    best_val = np.sum(r)
    start_time = time.time()
    
    # Coarse search
    for w in np.linspace(0.35, 1.65, 50):
        if time.time() - start_time > time_budget:
            break
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
    
    # Fine search
    for w in np.linspace(max(0.3, best_w - 0.15), min(1.7, best_w + 0.15), 40):
        if time.time() - start_time > time_budget:
            break
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
    
    # Very fine search
    for w in np.linspace(max(0.3, best_w - 0.04), min(1.7, best_w + 0.04), 20):
        if time.time() - start_time > time_budget:
            break
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
    
    return best_w


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
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.25, 0.7, 0.9, 1.1, 0.8, 1.2, 0.95, 1.05, 0.88, 1.12, 0.72, 1.28, 0.65, 1.35, 0.58, 1.42]
        
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
        
        # Random starts with varied radii
        rng = np.random.default_rng(42)
        for trial in range(40):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            x = rng.uniform(0.05, w0 - 0.05, N)
            y = rng.uniform(0.05, h0 - 0.05, N)
            r_min = rng.uniform(0.03, 0.06)
            r_max = rng.uniform(0.08, 0.14)
            r = rng.uniform(r_min, r_max, N)
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    # Sort guesses by initial energy (lower is better)
    def guess_energy(g):
        x, y, r, w, h = g
        return _compute_energy(x, y, r, w, h)
    
    guesses.sort(key=guess_energy)

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Step 1: Dynamics simulation to resolve overlaps
        remaining = deadline - (time.time() - start_time)
        sim_time = min(3.0, remaining * 0.20)
        x_sim, y_sim, r_sim = _dynamics_simulate(x0, y0, r0, w0, h0, sim_time)
        if x_sim is None:
            continue

        # Step 2: Optimize aspect ratio
        remaining = deadline - (time.time() - start_time)
        aspect_time = min(1.5, remaining * 0.10)
        w_opt = _optimize_aspect(x_sim, y_sim, r_sim, w0, h0, aspect_time)
        if abs(w_opt - w0) > 1e-6:
            h_opt = 2.0 - w_opt
            scale_x = w_opt / w0
            scale_y = h_opt / h0
            x_sim = x_sim * scale_x
            y_sim = y_sim * scale_y
            r_sim = r_sim * min(scale_x, scale_y)
            w0 = w_opt
            h0 = h_opt

        # Step 3: Expand radii
        remaining = deadline - (time.time() - start_time)
        expand_time = min(12.0, remaining * 0.50)
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
    if best_result is not None and time.time() < deadline - 3:
        x_best, y_best, r_best, w_best, h_best = best_result
        final_time = max(2.0, deadline - (time.time() - start_time) - 2)
        x_final, y_final, r_final = _expand_radii(x_best, y_best, r_best, w_best, h_best, final_time)
        if x_final is not None:
            valid, val = check_validity(x_final, y_final, r_final, w_best, h_best)
            if valid and val > best_value:
                best_value = val
                best_result = (x_final.copy(), y_final.copy(), r_final.copy(), w_best, h_best)
        
        # Second polish: try aspect optimization again on the improved solution
        if time.time() < deadline - 2:
            x_b, y_b, r_b, w_b, h_b = best_result
            w_opt2 = _optimize_aspect(x_b, y_b, r_b, w_b, h_b, 1.0)
            if abs(w_opt2 - w_b) > 1e-6:
                h_opt2 = 2.0 - w_opt2
                scale_x = w_opt2 / w_b
                scale_y = h_opt2 / h_b
                x_b2 = x_b * scale_x
                y_b2 = y_b * scale_y
                r_b2 = r_b * min(scale_x, scale_y)
                valid2, val2 = check_validity(x_b2, y_b2, r_b2, w_opt2, h_opt2)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = (x_b2.copy(), y_b2.copy(), r_b2.copy(), w_opt2, h_opt2)
        
        # Third polish: expand radii again after aspect optimization
        if time.time() < deadline - 1:
            x_b, y_b, r_b, w_b, h_b = best_result
            polish_time = max(1.0, deadline - (time.time() - start_time) - 0.5)
            x_p, y_p, r_p = _expand_radii(x_b, y_b, r_b, w_b, h_b, polish_time)
            if x_p is not None:
                valid_p, val_p = check_validity(x_p, y_p, r_p, w_b, h_b)
                if valid_p and val_p > best_value:
                    best_value = val_p
                    best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_b, h_b)

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
