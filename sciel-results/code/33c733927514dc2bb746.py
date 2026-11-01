"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_sum = 0.0
    best_result = None
    best_w = 1.0
    best_h = 1.0

    def check_validity(positions, radii, w, h):
        """Check if packing is valid and return sum of radii if so."""
        if w <= 0.05 or h <= 0.05:
            return -1.0
        if np.any(radii <= 0):
            return -1.0
        if np.any(positions[:, 0] < radii - 1e-9):
            return -1.0
        if np.any(positions[:, 0] > w - radii + 1e-9):
            return -1.0
        if np.any(positions[:, 1] < radii - 1e-9):
            return -1.0
        if np.any(positions[:, 1] > h - radii + 1e-9):
            return -1.0

        dx = positions[:, 0][:, None] - positions[:, 0][None, :]
        dy = positions[:, 1][:, None] - positions[:, 1][None, :]
        d_sq = dx * dx + dy * dy
        r_sum = radii[:, None] + radii[None, :]
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        if np.any(d_pairs < r_sum[iu] - 1e-9):
            return -1.0

        return np.sum(radii)

    def smooth_objective(z, penalty_weight=1000.0):
        """
        Smooth vectorized objective with penalty terms.
        z: [x0, y0, r0, x1, y1, r1, ..., x_{N-1}, y_{N-1}, r_{N-1}, w]
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        # Soft constraints via smooth max
        if w < 0.2 or h < 0.2:
            return 1e8

        # Clip radii to valid range
        r_clipped = np.clip(r, 0.001, min(w, h) / 2.0 - 0.001)

        obj = -np.sum(r_clipped)

        # Pairwise overlap penalty (smooth)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_clipped[:, None] + r_clipped[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-15))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap ** 2)

        # Wall constraint penalties (vectorized)
        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)

        wall_penalty = penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                         np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))

        # Penalize if any radius is too small
        small_r_penalty = penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)

        return obj + overlap_penalty + wall_penalty + small_r_penalty

    def force_relax(positions, radii, w, h, max_iter=500):
        """
        Force-directed relaxation to resolve overlaps.
        Returns relaxed positions and radii.
        """
        positions = positions.copy()
        radii = radii.copy()
        
        # Initial scaling to ensure no overlaps if possible
        # We want to find a configuration where circles don't overlap.
        # If they do, we shrink radii or move them.
        # Simple approach: iterate, compute forces, update positions.
        
        dt = 0.01
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
                
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            
            # Circle-Circle repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-9:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - dist
                        force_mag = overlap * 10.0
                        fx = (dx / dist) * force_mag
                        fy = (dy / dist) * force_mag
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
                    elif dist <= 1e-9:
                        # Random jitter
                        forces_x[i] += np.random.randn() * 0.1
                        forces_y[i] += np.random.randn() * 0.1
                        forces_x[j] -= np.random.randn() * 0.1
                        forces_y[j] -= np.random.randn() * 0.1

            # Wall repulsion
            for i in range(N):
                if positions[i, 0] < radii[i]:
                    forces_x[i] += (radii[i] - positions[i, 0]) * 10.0
                if positions[i, 0] > w - radii[i]:
                    forces_x[i] -= (positions[i, 0] - (w - radii[i])) * 10.0
                if positions[i, 1] < radii[i]:
                    forces_y[i] += (radii[i] - positions[i, 1]) * 10.0
                if positions[i, 1] > h - radii[i]:
                    forces_y[i] -= (positions[i, 1] - (h - radii[i])) * 10.0

            # Update positions
            positions[:, 0] += forces_x * dt
            positions[:, 1] += forces_y * dt
            
            # Clip to bounds (hard constraint for stability)
            positions[:, 0] = np.clip(positions[:, 0], radii, w - radii)
            positions[:, 1] = np.clip(positions[:, 1], radii, h - radii)
            
            # If no overlaps, we can stop early or slightly expand radii?
            # For now, just resolve overlaps.
            # Check if converged
            max_force = np.max(np.sqrt(forces_x**2 + forces_y**2))
            if max_force < 1e-4:
                break
                
        return positions, radii

    def get_grid_initialization(rows, cols, w, h):
        """
        Grid-based cellular decomposition.
        """
        positions = np.zeros((N, 2))
        radii = np.zeros(N)

        cell_w = w / cols
        cell_h = h / rows

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                max_r = min(cell_w, cell_h) / 2.0 * 0.90

                remaining_circles = N - idx
                remaining_cells = (rows - r) * cols - c

                if remaining_circles > remaining_cells:
                    # This cell needs 2 circles
                    r1 = max_r * 0.55
                    r2 = max_r * 0.35
                    offset_x = (r1 + r2) * 0.6
                    
                    if cell_w > cell_h:
                        positions[idx, 0] = cx - offset_x
                        positions[idx, 1] = cy
                        radii[idx] = max(r1, 0.01)
                        idx += 1
                        if idx < N:
                            positions[idx, 0] = cx + offset_x
                            positions[idx, 1] = cy
                            radii[idx] = max(r2, 0.01)
                            idx += 1
                    else:
                        positions[idx, 0] = cx
                        positions[idx, 1] = cy - (r1 + r2) * 0.3
                        radii[idx] = max(r1, 0.01)
                        idx += 1
                        if idx < N:
                            positions[idx, 0] = cx
                            positions[idx, 1] = cy + (r1 + r2) * 0.3
                            radii[idx] = max(r2, 0.01)
                            idx += 1
                else:
                    positions[idx, 0] = cx
                    positions[idx, 1] = cy
                    radii[idx] = max(max_r, 0.01)
                    idx += 1

        while idx < N:
            positions[idx, 0] = w / 2.0
            positions[idx, 1] = h / 2.0
            radii[idx] = 0.01
            idx += 1

        return positions, radii

    def get_initial_guesses():
        """Generate diverse initial guesses."""
        guesses = []
        grid_configs = [
            (3, 7), (7, 3),
            (4, 6), (6, 4),
            (5, 5),
            (3, 8), (8, 3),
            (4, 5), (5, 4),
            (2, 11), (11, 2),
            (3, 5), (5, 3),
        ]
        aspect_ratios = [
            1.0, 1.2, 0.8, 1.5, 0.5, 1.3, 0.7, 1.4, 0.6, 1.1, 0.9,
            1.6, 0.4, 1.7, 0.3, 1.8, 0.25
        ]

        for w_frac in aspect_ratios:
            w = w_frac
            h = 2.0 - w
            if w < 0.2 or h < 0.2:
                continue

            for rows, cols in grid_configs:
                positions, radii = get_grid_initialization(rows, cols, w, h)
                guesses.append((w, h, positions, radii))
                
        # Add some random starts
        rng = np.random.default_rng(42)
        for _ in range(5):
            w = rng.uniform(0.6, 1.4)
            h = 2.0 - w
            positions = np.zeros((N, 2))
            radii = np.zeros(N)
            for i in range(N):
                r = rng.uniform(0.05, 0.1)
                x = rng.uniform(r, w - r)
                y = rng.uniform(r, h - r)
                positions[i] = [x, y]
                radii[i] = r
            guesses.append((w, h, positions, radii))

        return guesses

    def polish_result(w, h, positions, radii, max_iter=500, ftol=1e-12):
        """Polish a valid packing using SLSQP with smooth objective."""
        z0 = np.concatenate([positions.flatten(), [w]])

        bounds = []
        for i in range(N):
            bounds.append((0.001, w - 0.001))
            bounds.append((0.001, h - 0.001))
            bounds.append((0.005, min(w, h) / 2.0 - 0.001))
        bounds.append((0.25, 1.75))

        try:
            result = minimize(
                lambda z: smooth_objective(z),
                z0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            if result.fun < 0:
                z_opt = result.x
                x_opt = z_opt[0::3][:N]
                y_opt = z_opt[1::3][:N]
                r_opt = z_opt[2::3][:N]
                w_opt = z_opt[-1]
                h_opt = 2.0 - w_opt

                positions_opt = np.column_stack([x_opt, y_opt])

                score = check_validity(positions_opt, r_opt, w_opt, h_opt)
                if score > 0:
                    return positions_opt, r_opt, w_opt, h_opt, score
        except Exception:
            pass

        return None

    def expansion_relaxation_polish(w, h, positions, radii):
        """
        Try to expand radii while maintaining validity.
        """
        positions = positions.copy()
        radii = radii.copy()
        
        # Sort circles by radius descending? No, just iterate.
        # We want to maximize sum(r).
        # Try increasing each radius slightly and adjusting position.
        
        for iteration in range(100):
            if time.time() > deadline - 5.0:
                break

            improved = False
            # Iterate in random order to avoid bias
            order = np.random.permutation(N)
            for i in order:
                old_r = radii[i]
                # Try a small increase
                new_r = old_r * 1.01 + 0.001
                if new_r > min(w, h) / 2.0 - 0.001:
                    continue

                # Try to find a position for i with radius new_r
                # Start from current position
                best_pos = None
                best_overlap = np.inf
                
                # Local search around current position
                for dx in np.linspace(-0.05, 0.05, 5):
                    for dy in np.linspace(-0.05, 0.05, 5):
                        new_x = positions[i, 0] + dx
                        new_y = positions[i, 1] + dy

                        # Check walls
                        if new_x < new_r or new_x > w - new_r:
                            continue
                        if new_y < new_r or new_y > h - new_r:
                            continue

                        # Check overlaps
                        valid = True
                        min_dist_violation = 0
                        for j in range(N):
                            if i == j:
                                continue
                            dx2 = new_x - positions[j, 0]
                            dy2 = new_y - positions[j, 1]
                            dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                            req_dist = new_r + radii[j]
                            if dist < req_dist:
                                valid = False
                                min_dist_violation = max(min_dist_violation, req_dist - dist)
                                break

                        if valid:
                            best_pos = (new_x, new_y)
                            break
                    if best_pos is not None:
                        break

                if best_pos is not None:
                    positions[i, 0] = best_pos[0]
                    positions[i, 1] = best_pos[1]
                    radii[i] = new_r
                    improved = True

            if not improved:
                break

        score = check_validity(positions, radii, w, h)
        if score > 0:
            return positions, radii, w, h, score
        return None

    # Main search loop
    guesses = get_initial_guesses()
    
    # Sort by initial sum of radii (descending)
    guesses.sort(key=lambda g: -np.sum(g[3]))

    for w, h, positions, radii in guesses:
        if time.time() > deadline - 20.0:
            break

        # 1. Force relax to get a valid starting point
        pos_relaxed, rad_relaxed = force_relax(positions, radii, w, h)
        
        # Check if relaxed is valid
        score_relaxed = check_validity(pos_relaxed, rad_relaxed, w, h)
        if score_relaxed > 0:
            if score_relaxed > best_sum:
                best_sum = score_relaxed
                best_result = np.column_stack([pos_relaxed, rad_relaxed])
                best_w = w
                best_h = h

            # 2. SLSQP Polish
            if time.time() < deadline - 10.0:
                result = polish_result(w, h, pos_relaxed, rad_relaxed)
                if result is not None:
                    pos_opt, r_opt, w_opt, h_opt, score = result
                    if score > best_sum:
                        best_sum = score
                        best_result = np.column_stack([pos_opt, r_opt])
                        best_w = w_opt
                        best_h = h_opt

                        # 3. Expansion Polish on the best SLSQP result
                        if time.time() < deadline - 5.0:
                            exp_result = expansion_relaxation_polish(w_opt, h_opt, pos_opt, r_opt)
                            if exp_result is not None:
                                pos_exp, r_exp, w_exp, h_exp, score_exp = exp_result
                                if score_exp > best_sum:
                                    best_sum = score_exp
                                    best_result = np.column_stack([pos_exp, r_exp])
                                    best_w = w_exp
                                    best_h = h_exp

    # If we haven't found anything good, return a simple grid solution
    if best_result is None:
        w, h = 1.0, 1.0
        positions, radii = get_grid_initialization(5, 5, w, h)
        # Force relax it
        positions, radii = force_relax(positions, radii, w, h)
        best_result = np.column_stack([positions, radii])
        best_w = w
        best_h = h

    return best_result
# EVOLVE-BLOCK-END
