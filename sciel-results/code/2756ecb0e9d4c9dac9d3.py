"""Circle packing in perimeter-4 rectangle: Rigid-body dynamics + surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# Precompute indices for vectorized operations
iu = np.triu_indices(N, k=1)


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid using vectorized operations."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    
    dx = cx[iu[0]] - cx[iu[1]]
    dy = cy[iu[0]] - cy[iu[1]]
    d2 = dx * dx + dy * dy
    rsum = r[iu[0]] + r[iu[1]]
    if np.any(d2 < rsum * rsum - tol):
        return False
    return True


def _compute_energy(cx, cy, r, W, H):
    """Compute surrogate energy: penalizes overlaps and wall proximity."""
    energy = 0.0
    # Wall penalties
    energy += np.sum(np.maximum(0, r - cx) ** 2)
    energy += np.sum(np.maximum(0, cx + r - W) ** 2)
    energy += np.sum(np.maximum(0, r - cy) ** 2)
    energy += np.sum(np.maximum(0, cy + r - H) ** 2)
    
    # Pairwise overlap penalties
    dx = cx[iu[0]] - cx[iu[1]]
    dy = cy[iu[0]] - cy[iu[1]]
    d2 = dx * dx + dy * dy
    rsum = r[iu[0]] + r[iu[1]]
    overlaps = np.maximum(rsum - np.sqrt(np.maximum(d2, 1e-12)), 0.0)
    energy += np.sum(overlaps * overlaps)
    return energy


def _generate_initial_packing(w, h, n_circles, seed=42):
    """
    Generate a high-quality initial packing using hexagonal lattice.
    We estimate the best radius that fits n_circles in a hexagonal arrangement
    within w x h, then place them.
    """
    rng = np.random.default_rng(seed)
    
    # Estimate radius based on area packing fraction
    # Hexagonal packing efficiency is ~0.9069
    area_per_circle = (w * h) / n_circles
    r_est = np.sqrt(area_per_circle / (np.pi * 0.9069))
    
    # We need to fit n_circles. Let's try to arrange them in rows.
    # For a hexagonal packing, row height is r * sqrt(3).
    # Let's find a configuration (rows, cols) that fits best.
    
    best_r = 0
    best_positions = None
    
    # Try varying the number of rows
    for rows in range(2, 10):
        # If we have 'rows' rows, how many columns do we need?
        # The number of circles in a hexagonal grid with 'rows' rows depends on the pattern.
        # Let's just try to place circles greedily in a hexagonal pattern.
        
        # Estimate r for this number of rows
        # Height occupied by 'rows' rows is roughly rows * r * sqrt(3)
        # Width occupied depends on cols.
        
        # Let's try a simpler approach:
        # Assume a grid of roughly sqrt(n) x sqrt(n)
        # But hexagonal is better.
        
        # Let's just use the area-based estimate and adjust if it doesn't fit.
        r_test = r_est
        # Adjust r_test to fit in height if rows are fixed
        r_test = min(r_test, h / (rows * np.sqrt(3)))
        
        # Now place circles
        positions = []
        for row in range(rows):
            # Offset for hexagonal
            offset = (row % 2) * r_test
            # Number of circles in this row
            # Available width: w - 2*r_test
            # Spacing: 2*r_test
            max_cols = int((w - 2*r_test + offset) / (2*r_test)) + 1 if r_test > 0 else 0
            
            for col in range(max_cols):
                if len(positions) >= n_circles:
                    break
                x = r_test + offset + col * 2 * r_test
                y = r_test + row * r_test * np.sqrt(3)
                
                if x + r_test <= w and y + r_test <= h:
                    positions.append((x, y, r_test))
        
        if len(positions) >= n_circles:
            # We have enough positions. Check if we can increase r?
            # For now, just take the first n_circles
            if r_test > best_r:
                best_r = r_test
                best_positions = positions[:n_circles]
    
    if best_positions is None:
        # Fallback to grid
        rows = int(np.sqrt(n_circles))
        cols = int(np.ceil(n_circles / rows))
        r_est = min(w / (2 * cols), h / (2 * rows)) * 0.9
        positions = []
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * (w / cols)
            y = (row + 0.5) * (h / rows)
            positions.append((x, y, r_est))
        return np.array([p[0] for p in positions]), np.array([p[1] for p in positions]), np.array([p[2] for p in positions])
    
    return np.array([p[0] for p in best_positions]), np.array([p[1] for p in best_positions]), np.array([p[2] for p in best_positions])


def _dynamics_simulate(x, y, r, w, h, time_budget, dt=0.001, max_iter=20000):
    """
    Rigid-body dynamics simulation with impulse-based collision resolution.
    Mass proportional to radius squared.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    if w < 0.2 or h < 0.2:
        return None, None, None

    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = np.maximum(r * r, 1e-6)

    # Ensure valid start
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    damping = 0.98
    start_time = time.time()
    
    # Adaptive force scaling
    wall_stiffness = 500.0
    pair_stiffness = 500.0

    n_pairs = len(iu[0])

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

        # Pairwise repulsion
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

            # Vectorized force accumulation
            # This is tricky with numpy for sparse updates, but we can use np.add.at
            # Or just loop if N is small (21 is small)
            for idx in range(n_pairs):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

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

        # Impulse-based collision resolution (post-step correction) - multiple passes
        for _ in range(5):
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

            for idx in range(n_pairs):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                m_i = mass[i_idx]
                m_j = mass[j_idx]
                total_m = m_i + m_j
                correction = overlap[idx] * 0.5
                x[i_idx] += ux[idx] * correction * (m_j / total_m)
                y[i_idx] += uy[idx] * correction * (m_j / total_m)
                x[j_idx] -= ux[idx] * correction * (m_i / total_m)
                y[j_idx] -= uy[idx] * correction * (m_i / total_m)

            # Re-clamp after correction
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

    # Final validity check
    if not _is_valid(x, y, r, w, h):
        return None, None, None

    return x, y, r


def _expand_radii(x, y, r, w, h, time_budget):
    """
    Iteratively expand radii.
    Strategy:
    1. Try to scale all radii uniformly. If valid, accept.
    2. If not, try to resolve overlaps by moving circles (dynamics step).
    3. Repeat.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    start_time = time.time()
    
    # We will try to increase a global scale factor
    scale = 1.0
    max_iterations = 500
    
    for it in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
            
        # Try to increase scale
        # Binary search for the maximum scale that keeps packing valid?
        # Or just incremental? Incremental is safer.
        
        # Let's try a small increment
        new_scale = scale * 1.005
        r_new = r * new_scale
        
        # Check if this is valid
        if _is_valid(x, y, r_new, w, h):
            r = r_new
            scale = new_scale
        else:
            # It's invalid. We need to move circles to make space.
            # Run a short dynamics simulation to resolve overlaps
            # Note: dynamics_simulate expects valid input or close to it.
            # If we have overlaps, we need to resolve them.
            
            # We can use the dynamics simulator to push them apart.
            # But dynamics_simulate checks validity at the end.
            # We need a version that just resolves overlaps without strict validity check at the end,
            # or we can just run it and see if it converges.
            
            # Let's try to run a few steps of dynamics to resolve the overlaps created by the expansion.
            # We'll use a very short budget.
            x_res, y_res, r_res = _dynamics_simulate(x, y, r_new, w, h, time_budget=0.1)
            
            if x_res is not None:
                # If dynamics succeeded in making it valid, we can accept the expansion
                x = x_res
                y = y_res
                r = r_res
                scale = new_scale
            else:
                # Dynamics failed to resolve. Try a smaller increment?
                # Or stop?
                # Let's try a smaller increment
                new_scale = scale * 1.001
                r_new = r * new_scale
                if _is_valid(x, y, r_new, w, h):
                    r = r_new
                    scale = new_scale
                else:
                    # Can't expand further
                    break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None, None, None


# Removed _optimize_aspect as we will search aspect ratios in the main loop
# by generating new packings for each aspect ratio.


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if not _is_valid(x, y, r, w, h, tol):
            return False, 0.0
        val = float(np.sum(r))
        return True, val

    # Search over aspect ratios
    # Perimeter is 4, so 2(w+h) = 4 => w+h = 2.
    # w ranges from 0.5 to 1.5 (symmetric around 1.0)
    
    # We will try a set of aspect ratios
    # And for each, generate a good initial packing, resolve it, and expand.
    
    aspect_ratios = np.linspace(0.5, 1.5, 21)
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Generate initial packing
        x0, y0, r0 = _generate_initial_packing(w0, h0, N, seed=int(w0*100))
        
        # Step 1: Dynamics simulation to resolve overlaps and settle
        # We give it a decent amount of time to settle
        sim_time = min(2.0, (deadline - (time.time() - start_time)) * 0.2)
        x_sim, y_sim, r_sim = _dynamics_simulate(x0, y0, r0, w0, h0, sim_time)
        
        if x_sim is None:
            continue

        # Step 2: Expand radii
        # We have time left for expansion
        expand_time = min(5.0, (deadline - (time.time() - start_time)) * 0.5)
        x_exp, y_exp, r_exp = _expand_radii(x_sim, y_sim, r_sim, w0, h0, expand_time)
        
        if x_exp is not None:
            valid, val = check_validity(x_exp, y_exp, r_exp, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x_exp.copy(), y_exp.copy(), r_exp.copy(), w0, h0)
        else:
            # If expansion failed, try the simulation result directly
            valid, val = check_validity(x_sim, y_sim, r_sim, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x_sim.copy(), y_sim.copy(), r_sim.copy(), w0, h0)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 5:
        x_best, y_best, r_best, w_best, h_best = best_result
        final_time = deadline - (time.time() - start_time) - 2
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
