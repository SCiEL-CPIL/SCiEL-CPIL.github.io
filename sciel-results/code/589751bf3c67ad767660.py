"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    best_sum = 0.0
    best_result = None

    def check_validity(positions, radii, w, h):
        """Check if packing is valid and return sum of radii if so."""
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
        if np.any(np.sqrt(d_sq[iu]) < r_sum[iu] - 1e-9):
            return -1.0

        return np.sum(radii)

    def smooth_objective(z, penalty_weight=500.0):
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
        r_clipped = np.clip(r, 0.001, min(w, h) / 2 - 0.001)

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
        # x violations
        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        # y violations
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)

        wall_penalty = penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                         np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))

        # Penalize if any radius is too small
        small_r_penalty = penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)

        return obj + overlap_penalty + wall_penalty + small_r_penalty

    def get_grid_initialization(rows, cols, w, h):
        """
        Grid-based cellular decomposition: partition rectangle into rows x cols cells,
        assign circles to cells, and set initial radii based on cell inradius.
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

                # Determine how many circles go in this cell
                remaining_circles = N - idx
                remaining_cells = (rows - r) * cols - c

                if remaining_circles > remaining_cells:
                    # This cell needs 2 circles
                    r1 = max_r * 0.55
                    r2 = max_r * 0.35
                    offset_x = (r1 + r2) * 0.6
                    offset_y = 0.0

                    # Check if horizontal or vertical fits better
                    if cell_w > cell_h:
                        # Horizontal arrangement
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
                        # Vertical arrangement
                        positions[idx, 0] = cx
                        positions[idx, 1] = cy - offset_y - (r1 + r2) * 0.3
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

        # Fill any remaining with tiny circles at center
        while idx < N:
            positions[idx, 0] = w / 2.0
            positions[idx, 1] = h / 2.0
            radii[idx] = 0.01
            idx += 1

        return positions, radii

    def get_initial_guesses():
        """Generate diverse initial guesses using grid-based cellular decomposition."""
        guesses = []

        # Grid configurations that can accommodate 21 circles
        # (rows, cols) where rows*cols >= 21 or we can pack 2 per cell
        grid_configs = [
            (3, 7), (7, 3),   # 21 cells exactly
            (4, 6), (6, 4),   # 24 cells
            (5, 5),           # 25 cells
            (3, 8), (8, 3),   # 24 cells
            (4, 5), (5, 4),   # 20 cells, need some 2-per-cell
            (2, 11), (11, 2), # 22 cells
            (3, 5), (5, 3),   # 15 cells, need 2-3 per cell
        ]

        # Aspect ratios: w + h = 2.0
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

                # Check if this is a valid starting point
                score = check_validity(positions, radii, w, h)
                if score > 0:
                    guesses.append((w, h, positions, radii))

        return guesses

    def polish_result(w, h, positions, radii, max_iter=500, ftol=1e-12):
        """Polish a valid packing using SLSQP with smooth objective."""
        z0 = np.concatenate([positions.flatten(), [w]])

        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.001, w - 0.001))  # x_i
            bounds.append((0.001, h - 0.001))  # y_i
            bounds.append((0.005, min(w, h) / 2.0 - 0.001))  # r_i
        bounds.append((0.25, 1.75))  # w

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

                # Validate
                score = check_validity(positions_opt, r_opt, w_opt, h_opt)
                if score > 0:
                    return positions_opt, r_opt, w_opt, h_opt, score
        except Exception:
            pass

        return None

    def high_precision_polish(w, h, positions, radii, max_iter=1000, ftol=1e-14):
        """Final high-precision polish pass with tighter tolerance."""
        z0 = np.concatenate([positions.flatten(), [w]])

        bounds = []
        for i in range(N):
            bounds.append((max(0.001, positions[i, 0] - 0.05), min(w - 0.001, positions[i, 0] + 0.05)))
            bounds.append((max(0.001, positions[i, 1] - 0.05), min(h - 0.001, positions[i, 1] + 0.05)))
            bounds.append((max(0.005, radii[i] * 0.8), min(min(w, h) / 2.0 - 0.001, radii[i] * 1.2)))
        bounds.append((max(0.25, w - 0.1), min(1.75, w + 0.1)))

        try:
            result = minimize(
                lambda z: smooth_objective(z, penalty_weight=2000.0),
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

    def _force_relax(positions, radii, w, h, max_iters=200, tol=1e-10):
        """Force-based relaxation: push overlapping circles apart."""
        positions = positions.copy()
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            forces = np.zeros_like(positions)
            max_disp = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        if dist < 1e-12:
                            dx, dy = 1e-6, 1e-6
                            dist = np.sqrt(dx * dx + dy * dy)
                        overlap = min_dist - dist
                        fx = dx / dist * overlap
                        fy = dy / dist * overlap
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            # Wall forces
            for i in range(N):
                if positions[i, 0] < radii[i]:
                    forces[i, 0] += (radii[i] - positions[i, 0])
                if positions[i, 0] > w - radii[i]:
                    forces[i, 0] -= (positions[i, 0] - (w - radii[i]))
                if positions[i, 1] < radii[i]:
                    forces[i, 1] += (radii[i] - positions[i, 1])
                if positions[i, 1] > h - radii[i]:
                    forces[i, 1] -= (positions[i, 1] - (h - radii[i]))
            # Apply with damping
            step = 0.5
            for i in range(N):
                positions[i, 0] += step * forces[i, 0]
                positions[i, 1] += step * forces[i, 1]
                # Clamp to walls
                positions[i, 0] = max(radii[i], min(w - radii[i], positions[i, 0]))
                positions[i, 1] = max(radii[i], min(h - radii[i], positions[i, 1]))
                max_disp = max(max_disp, abs(forces[i, 0]), abs(forces[i, 1]))
            if max_disp < tol:
                break
        return positions

    def expansion_relaxation_polish(w, h, positions, radii):
        """
        Try to expand radii while maintaining validity.
        Uses force-based relaxation + iterative radius increase.
        """
        positions = positions.copy()
        radii = radii.copy()
        w_current = w
        h_current = h

        for iteration in range(120):
            if time.time() > deadline - 2.0:
                break

            improved = False
            for i in range(N):
                old_r = radii[i]
                for dr in [0.003, 0.001, 0.0005, 0.0002]:
                    new_r = old_r + dr
                    if new_r > min(w_current, h_current) / 2.0 - 0.001:
                        continue

                    # Try current position and small perturbations
                    best_pos = None
                    for dx in [0, 0.003, -0.003, 0.006, -0.006, 0.01, -0.01]:
                        for dy in [0, 0.003, -0.003, 0.006, -0.006, 0.01, -0.01]:
                            new_x = positions[i, 0] + dx
                            new_y = positions[i, 1] + dy

                            # Check walls
                            if new_x < new_r or new_x > w_current - new_r:
                                continue
                            if new_y < new_r or new_y > h_current - new_r:
                                continue

                            # Check overlaps with other circles
                            valid = True
                            for j in range(N):
                                if i == j:
                                    continue
                                dx2 = new_x - positions[j, 0]
                                dy2 = new_y - positions[j, 1]
                                dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                                if dist < new_r + radii[j] - 1e-9:
                                    valid = False
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
                        break

            if not improved:
                # Force-based relaxation to resolve near-overlaps, then retry
                positions = _force_relax(positions, radii, w_current, h_current, max_iters=50)
                # Check validity after relaxation
                if check_validity(positions, radii, w_current, h_current) <= 0:
                    break
                # One more attempt after relaxation
                for i in range(N):
                    old_r = radii[i]
                    new_r = old_r + 0.001
                    if new_r > min(w_current, h_current) / 2.0 - 0.001:
                        continue
                    best_pos = None
                    for dx in [0, 0.003, -0.003, 0.006, -0.006]:
                        for dy in [0, 0.003, -0.003, 0.006, -0.006]:
                            new_x = positions[i, 0] + dx
                            new_y = positions[i, 1] + dy
                            if new_x < new_r or new_x > w_current - new_r:
                                continue
                            if new_y < new_r or new_y > h_current - new_r:
                                continue
                            valid = True
                            for j in range(N):
                                if i == j:
                                    continue
                                dx2 = new_x - positions[j, 0]
                                dy2 = new_y - positions[j, 1]
                                dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                                if dist < new_r + radii[j] - 1e-9:
                                    valid = False
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
                        break
                if not improved:
                    break

        score = check_validity(positions, radii, w_current, h_current)
        if score > 0:
            return positions, radii, w_current, h_current, score

        return None

    # Main search loop
    guesses = get_initial_guesses()

    # Sort by initial sum of radii (descending)
    guesses.sort(key=lambda g: -np.sum(g[3]))

    for w, h, positions, radii in guesses:
        if time.time() > deadline - 10.0:
            break

        # Try polishing
        result = polish_result(w, h, positions, radii)
        if result is not None:
            pos_opt, r_opt, w_opt, h_opt, score = result
            if score > best_sum:
                best_sum = score
                best_result = np.column_stack([pos_opt, r_opt])

                # Try high-precision polish on the best result
                if time.time() < deadline - 15.0:
                    hp_result = high_precision_polish(w_opt, h_opt, pos_opt, r_opt)
                    if hp_result is not None:
                        pos_hp, r_hp, w_hp, h_hp, score_hp = hp_result
                        if score_hp > best_sum:
                            best_sum = score_hp
                            best_result = np.column_stack([pos_hp, r_hp])

        # Try expansion relaxation
        if time.time() < deadline - 5.0:
            # First apply force relaxation to the initial guess
            pos_relaxed = _force_relax(positions, radii, w, h, max_iters=100)
            if check_validity(pos_relaxed, radii, w, h) > 0:
                result2 = expansion_relaxation_polish(w, h, pos_relaxed, radii)
            else:
                result2 = expansion_relaxation_polish(w, h, positions, radii)
            if result2 is not None:
                pos2, r2, w2, h2, score2 = result2
                if score2 > best_sum:
                    best_sum = score2
                    best_result = np.column_stack([pos2, r2])

    # If we haven't found anything good, return a simple grid solution
    if best_result is None:
        w, h = 1.0, 1.0
        positions, radii = get_grid_initialization(5, 5, w, h)
        best_result = np.column_stack([positions, radii])

    return best_result
# EVOLVE-BLOCK-END
