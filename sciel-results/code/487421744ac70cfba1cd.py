"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(positions):
    """Compute the exact minimum radius."""
    n = len(positions)
    if n == 0:
        return 0.0

    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    if min_boundary <= 0:
        return 0.0

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)

    return min(min_boundary, min_pair / 2.0)


def _generate_hexagonal_grid(n_circles, spacing=None):
    """Generate a hexagonal lattice of points inside the unit square."""
    if spacing is None:
        spacing = 0.072

    points = []
    max_extent = 0.5
    row_spacing = spacing * np.sqrt(3) / 2.0
    n_rows = int(max_extent / row_spacing) + 2

    for row in range(-n_rows, n_rows + 1):
        offset = (row % 2) * 0.5 * spacing
        y = row * row_spacing
        if abs(y) > max_extent:
            continue
        x_max = int(max_extent / spacing) + 1
        for col in range(-x_max, x_max + 1):
            x = col * spacing + offset
            if abs(x) <= max_extent and abs(y) <= max_extent:
                points.append([x, y])

    points = np.array(points)

    if len(points) < n_circles:
        # Retry with a tighter spacing to get enough candidate points.
        spacing = 0.065
        points = []
        row_spacing = spacing * np.sqrt(3) / 2.0
        n_rows = int(max_extent / row_spacing) + 2
        for row in range(-n_rows, n_rows + 1):
            offset = (row % 2) * 0.5 * spacing
            y = row * row_spacing
            if abs(y) > max_extent:
                continue
            x_max = int(max_extent / spacing) + 1
            for col in range(-x_max, x_max + 1):
                x = col * spacing + offset
                if abs(x) <= max_extent and abs(y) <= max_extent:
                    points.append([x, y])
        points = np.array(points)

    # Pick the n_circles closest to the centre to keep the packing symmetric.
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx].copy()

    return points


def _force_relax(positions, max_iters=2000, dt=0.002, k_repel=0.5, k_boundary=5.0):
    """Repulsive force-based relaxation pushing circles toward a dense local packing."""
    positions = positions.copy()
    n = len(positions)

    for it in range(max_iters):
        forces = np.zeros_like(positions)

        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)

        threshold = 0.15
        mask = dist < threshold
        mask &= ~np.eye(n, dtype=bool)

        for i in range(n):
            close_j = np.where(mask[i])[0]
            for j in close_j:
                d = dist[i, j]
                if d < 1e-10:
                    continue
                force_mag = k_repel / (d * d)
                direction = diff[i, j] / d
                forces[i] += force_mag * direction

        for i in range(n):
            x, y = positions[i]
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y
            if d_left < 0.1:
                forces[i, 0] += k_boundary / (d_left * d_left)
            if d_right < 0.1:
                forces[i, 0] -= k_boundary / (d_right * d_right)
            if d_bottom < 0.1:
                forces[i, 1] += k_boundary / (d_bottom * d_bottom)
            if d_top < 0.1:
                forces[i, 1] -= k_boundary / (d_top * d_top)

        positions += dt * forces
        positions = np.clip(positions, -0.49, 0.49)

        if it > 100:
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < 1e-6:
                break

    return positions


def _soft_min_radius(positions, t):
    """Smooth surrogate of the minimum radius via log-sum-exp soft-min."""
    n = len(positions)

    boundary_dists = np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    )

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    pair_radii = np.sqrt(dist_sq) / 2.0

    iu = np.triu_indices(n, k=1)
    pair_radii_flat = pair_radii[iu]

    all_radii = np.concatenate([boundary_dists, pair_radii_flat])
    all_radii = np.clip(all_radii, 1e-12, None)

    x = -all_radii / t
    max_x = np.max(x)
    if not np.isfinite(max_x):
        return 0.0

    log_sum = np.log(np.sum(np.exp(x - max_x))) + max_x
    soft_min_val = -t * log_sum

    return soft_min_val


def _optimize_soft_min(positions, t_start, t_end, n_steps, max_iter, deadline):
    """Annealed L-BFGS-B on the negative soft-min radius."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    current_pos = positions.copy()

    for i in range(n_steps):
        if time.time() > deadline:
            break
        t_val = t_start * (t_end / t_start) ** (i / (n_steps - 1)) if n_steps > 1 else t_start

        def _objective(pos_flat, t_val=t_val):
            pos = pos_flat.reshape(-1, 2)
            return -_soft_min_radius(pos, t_val)

        try:
            res = minimize(
                _objective,
                current_pos.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': max_iter * 3,
                         'ftol': 1e-8, 'gtol': 1e-5}
            )
            current_pos = res.x.reshape(-1, 2)
        except Exception:
            pass

    return current_pos


def _pairwise_push_optimize(positions, max_iters=3000, deadline=None):
    """Iteratively find the closest pair and push them apart."""
    positions = positions.copy()
    n = len(positions)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)

        min_idx = np.argmin(dist_sq)
        i, j = np.unravel_index(min_idx, (n, n))
        if i >= j:
            i, j = j, i

        d = np.sqrt(dist_sq[i, j])
        if d < 1e-10:
            positions[i, 0] += 0.001 * np.random.randn()
            positions[i, 1] += 0.001 * np.random.randn()
            positions[j, 0] -= 0.001 * np.random.randn()
            positions[j, 1] -= 0.001 * np.random.randn()
        else:
            direction = (positions[i] - positions[j]) / d
            push = 0.002
            positions[i] += direction * push
            positions[j] -= direction * push

        for k in range(n):
            x, y = positions[k]
            if x > 0.49:
                positions[k, 0] = 0.49
            elif x < -0.49:
                positions[k, 0] = -0.49
            if y > 0.49:
                positions[k, 1] = 0.49
            elif y < -0.49:
                positions[k, 1] = -0.49

    return positions


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

    best_score = 0.0
    best_positions = None

    def _consider(pos):
        nonlocal best_score, best_positions
        s = _score(pos)
        if s > best_score:
            best_score = s
            best_positions = pos.copy()
        return s

    # Phase 1: Hexagonal lattice initialisation.
    hex_pos = _generate_hexagonal_grid(N)
    _consider(hex_pos)

    # Phase 2: Force-based relaxation to reach a dense local packing.
    if time.time() < deadline - 60:
        try:
            relaxed_pos = _force_relax(best_positions, max_iters=1500, dt=0.002,
                                       k_repel=0.5, k_boundary=5.0)
            _consider(relaxed_pos)
        except Exception:
            pass

    # Phase 3: Annealed soft-min L-BFGS-B optimisation (coarse -> fine).
    if time.time() < deadline - 40:
        try:
            opt_pos = _optimize_soft_min(best_positions, t_start=5.0, t_end=0.05,
                                         n_steps=8, max_iter=40, deadline=deadline)
            _consider(opt_pos)
        except Exception:
            pass

    # Phase 4: Pairwise push refinement of the closest pairs.
    if time.time() < deadline - 20:
        try:
            pushed_pos = _pairwise_push_optimize(best_positions, max_iters=3000,
                                                 deadline=deadline)
            _consider(pushed_pos)
        except Exception:
            pass

    # Phase 5: Final high-precision polish with a very low temperature.
    if time.time() < deadline - 5:
        try:
            polish_pos = _optimize_soft_min(best_positions, t_start=0.1, t_end=0.01,
                                            n_steps=5, max_iter=30, deadline=deadline)
            _consider(polish_pos)
        except Exception:
            pass

    # Ensure a valid output.
    if best_positions is None:
        best_positions = _generate_hexagonal_grid(N)

    best_positions = np.clip(best_positions, -0.5, 0.5)
    return best_positions

# EVOLVE-BLOCK-END
