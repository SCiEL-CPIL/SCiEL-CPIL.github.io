"""Circle packing in unit square n=32: physics-based optimization with diverse starts."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32
DEADLINE_SECONDS = 115  # safety margin before 120s limit


def _pairwise_distances(pos):
    """Return matrix of pairwise distances between N circles."""
    # pos shape (N, 2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _energy(pos, radii):
    """
    Objective: maximize sum of radii subject to constraints.
    We minimize -sum(radii) + penalty for violations.
    pos: (N, 2), radii: (N,)
    """
    # Boundary constraints: circle must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    boundary_viol = 0.0
    for i in range(N):
        x, y = pos[i]
        r = radii[i]
        boundary_viol += max(0, r - x) ** 2
        boundary_viol += max(0, r - (1 - x)) ** 2
        boundary_viol += max(0, r - y) ** 2
        boundary_viol += max(0, r - (1 - y)) ** 2

    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    overlap_viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2)
            min_d = radii[i] + radii[j]
            if d < min_d:
                overlap_viol += (min_d - d) ** 2

    penalty = 1000.0 * (boundary_viol + overlap_viol)
    return -np.sum(radii) + penalty


def _unconstrained_objective(vars_vec):
    """
    vars_vec: (3N,) = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    We use a soft-potential approach: minimize repulsive energy.
    """
    pos = vars_vec[:2 * N].reshape(N, 2)
    radii = vars_vec[2 * N:]

    # Clamp radii to be positive
    radii = np.maximum(radii, 1e-6)

    # Soft repulsive potential: sum over pairs of (1/d^6) when d < r_i + r_j + margin
    # Plus boundary repulsion
    total_energy = 0.0

    # Boundary repulsion: push circles away from edges
    eps = 0.01
    for i in range(N):
        x, y = pos[i]
        r = radii[i]
        # Distance to each wall
        d_left = x - r
        d_right = (1 - x) - r
        d_bottom = y - r
        d_top = (1 - y) - r
        for d in [d_left, d_right, d_bottom, d_top]:
            if d < eps:
                total_energy += 100.0 / (d + 1e-6) ** 2

    # Pairwise repulsion
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy) + 1e-6
            r_sum = radii[i] + radii[j]
            if d < r_sum + 0.1:
                # Strong repulsion when overlapping or close
                total_energy += 10.0 / d ** 6

    # We want to MAXIMIZE sum of radii, so minimize negative sum
    # But the repulsion keeps them apart
    # Combine: minimize -sum(radii) + repulsion
    return -np.sum(radii) + total_energy


def _generate_grid_init(rows, cols, noise_scale=0.0):
    """Generate initial positions on a grid with optional noise."""
    positions = []
    radii = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= N:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            if noise_scale > 0:
                x += np.random.uniform(-noise_scale, noise_scale)
                y += np.random.uniform(-noise_scale, noise_scale)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            positions.append([x, y])
            # Initial radius estimate based on grid spacing
            r = min(1.0 / cols, 1.0 / rows) * 0.45
            radii.append(r)

    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5])
        radii.append(0.05)

    positions = np.array(positions[:N])
    radii = np.array(radii[:N])
    return positions, radii


def _generate_random_init():
    """Generate random initial positions."""
    positions = np.random.uniform(0.1, 0.9, size=(N, 2))
    radii = np.full(N, 0.1)
    return positions, radii


def _optimize_from_start(pos, radii, deadline):
    """
    Optimize from a given starting configuration.
    Returns (best_sum_radii, best_pos, best_radii) or None if timeout.
    """
    start_time = time.time()

    # Combine into single vector
    x0 = np.concatenate([pos.flatten(), radii])

    best_energy = _unconstrained_objective(x0)
    best_x = x0.copy()

    # Try multiple optimization methods
    methods = ['L-BFGS-B', 'Nelder-Mead']

    for method in methods:
        if time.time() - start_time > deadline - 10:
            break

        try:
            result = minimize(
                _unconstrained_objective,
                x0,
                method=method,
                options={'maxiter': 2000, 'maxfev': 5000, 'disp': False}
            )

            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_x = result.x.copy()

        except Exception:
            pass

        x0 = best_x.copy()  # Use best so far as next start

    # Extract best configuration
    best_pos = best_x[:2 * N].reshape(N, 2)
    best_radii = np.maximum(best_x[2 * N:], 1e-6)

    # Validate: check if all circles are inside and non-overlapping
    # If invalid, shrink radii to make valid
    valid = True
    for i in range(N):
        x, y = best_pos[i]
        r = best_radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            break
        for j in range(i + 1, N):
            d = np.sqrt((best_pos[i, 0] - best_pos[j, 0]) ** 2 + (best_pos[i, 1] - best_pos[j, 1]) ** 2)
            if d < best_radii[i] + best_radii[j] - 1e-9:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Shrink all radii until valid
        scale = 1.0
        for _ in range(50):
            best_radii = best_radii * scale
            valid = True
            for i in range(N):
                x, y = best_pos[i]
                r = best_radii[i]
                if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                    valid = False
                    break
                for j in range(i + 1, N):
                    d = np.sqrt((best_pos[i, 0] - best_pos[j, 0]) ** 2 + (best_pos[i, 1] - best_pos[j, 1]) ** 2)
                    if d < best_radii[i] + best_radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            scale *= 0.95

    sum_radii = np.sum(best_radii)
    return sum_radii, best_pos, best_radii


def construct():
    """
    Pack 32 circles in unit square maximizing sum of radii.
    Returns array of shape (32, 3): rows (x, y, radius).
    """
    global DEADLINE_SECONDS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    np.random.seed(42)

    best_sum = 0.0
    best_pos = None
    best_radii = None

    # Define diverse initializations
    initializations = []

    # Regular grids
    for rows, cols in [(6, 6), (5, 7), (7, 5), (4, 8), (8, 4), (8, 4)]:
        pos, rad = _generate_grid_init(rows, cols, noise_scale=0.0)
        initializations.append((pos, rad))

    # Grids with noise
    for rows, cols in [(6, 6), (5, 7), (7, 5)]:
        for noise in [0.05, 0.1, 0.15]:
            pos, rad = _generate_grid_init(rows, cols, noise_scale=noise)
            initializations.append((pos, rad))

    # Random inits
    for _ in range(5):
        pos, rad = _generate_random_init()
        initializations.append((pos, rad))

    # Sort initializations: try better-looking ones first (larger initial sum)
    initializations.sort(key=lambda pr: -np.sum(pr[1]))

    for pos, rad in initializations:
        if time.time() > deadline - 15:
            break

        try:
            sum_r, pos_opt, rad_opt = _optimize_from_start(pos, rad, deadline - 5)
            if sum_r > best_sum:
                best_sum = sum_r
                best_pos = pos_opt
                best_radii = rad_opt
        except Exception:
            continue

    # If no valid result, fall back to grid
    if best_pos is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final validation and cleanup
    result = np.column_stack([best_pos, best_radii])

    # Ensure all values are within bounds
    result[:, 0] = np.clip(result[:, 0], 0, 1)
    result[:, 1] = np.clip(result[:, 1], 0, 1)
    result[:, 2] = np.maximum(result[:, 2], 1e-6)

    # Final validity check: shrink if needed
    for _ in range(20):
        valid = True
        for i in range(N):
            x, y, r = result[i]
            if x - r < -1e-8 or x + r > 1 + 1e-8 or y - r < -1e-8 or y + r > 1 + 1e-8:
                valid = False
                break
            for j in range(i + 1, N):
                d = np.sqrt((result[i, 0] - result[j, 0]) ** 2 + (result[i, 1] - result[j, 1]) ** 2)
                if d < result[i, 2] + result[j, 2] - 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        result[:, 2] *= 0.95

    return result
