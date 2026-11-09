"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START


def _max_radius_for_circle(x, y, centers, radii, idx_to_skip):
    """Compute the maximum radius for a circle at (x,y) given other circles."""
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    for j in range(len(centers)):
        if j == idx_to_skip:
            continue
        dx = x - centers[j, 0]
        dy = y - centers[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        r = dist - radii[j]
        if r < max_r:
            max_r = r
    return max(0.0, max_r)


def _greedy_expansion(centers, radii):
    """Grow each circle to its local maximum radius, iterating until convergence."""
    for _iteration in range(100):
        changed = False
        for i in range(N):
            new_r = _max_radius_for_circle(centers[i, 0], centers[i, 1], centers, radii, i)
            if new_r > radii[i] + 1e-12:
                radii[i] = new_r
                changed = True
        if not changed:
            break
    return radii


def _validate(centers, radii):
    """Check all circles are inside [0,1]^2 and disjoint."""
    for i in range(N):
        if centers[i, 0] - radii[i] < -1e-9 or centers[i, 0] + radii[i] > 1 + 1e-9:
            return False
        if centers[i, 1] - radii[i] < -1e-9 or centers[i, 1] + radii[i] > 1 + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True


def _score(centers, radii):
    """Sum of radii."""
    return np.sum(radii)


def _perturb_and_expand(centers0, radii0, sigma, rng):
    """Perturb positions, then greedily expand. Return new centers, radii, score."""
    centers = centers0.copy()
    radii = radii0.copy()
    # Perturb positions
    centers += rng.normal(0, sigma, size=centers.shape)
    # Clamp to valid range
    for i in range(N):
        centers[i, 0] = np.clip(centers[i, 0], 0.01, 0.99)
        centers[i, 1] = np.clip(centers[i, 1], 0.01, 0.99)
    # Reset radii to minimum possible
    for i in range(N):
        radii[i] = _max_radius_for_circle(centers[i, 0], centers[i, 1], centers, radii, i)
    # Greedy expansion
    radii = _greedy_expansion(centers, radii)
    if not _validate(centers, radii):
        return None, None, -np.inf
    return centers, radii, _score(centers, radii)


def _make_initial_grid(n_cols, n_rows, offset_x=0.5, offset_y=0.5):
    """Create an initial grid layout for N circles."""
    # Place N circles in a grid of n_cols x n_rows
    centers = np.zeros((N, 2))
    placed = 0
    for j in range(n_rows):
        for i in range(n_cols):
            if placed >= N:
                break
            x = (i + offset_x) / n_cols
            y = (j + offset_y) / n_rows
            centers[placed, 0] = x
            centers[placed, 1] = y
            placed += 1
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    rng = np.random.default_rng(42)

    # Try multiple grid configurations
    best_centers = None
    best_radii = None
    best_score = -np.inf

    grid_configs = [
        (5, 5),   # 25 cells, use 26? no, need exactly N. Let me handle differently.
    ]

    # Actually, let's use various grid sizes that can hold 26 circles
    # A 5x6 grid = 30 cells, use 26
    # A 6x5 grid = 30 cells
    # A 4x7 grid = 28 cells
    # A 7x4 grid = 28 cells
    # A 5x5 = 25, not enough. Need at least 26.
    # Let's use grids with >= 26 cells and pick 26 of them.

    grid_configs = [
        (6, 5),
        (5, 6),
        (7, 4),
        (4, 7),
        (6, 6),  # 36 cells, pick 26
    ]

    for n_cols, n_rows in grid_configs:
        if time.time() > deadline:
            break
        centers0 = _make_initial_grid(n_cols, n_rows)
        # Ensure we have exactly N circles
        if len(centers0) < N:
            continue
        centers0 = centers0[:N]

        # Initial greedy expansion
        radii0 = np.zeros(N)
        for i in range(N):
            radii0[i] = _max_radius_for_circle(centers0[i, 0], centers0[i, 1], centers0, radii0, i)
        radii0 = _greedy_expansion(centers0, radii0)

        if _validate(centers0, radii0):
            s = _score(centers0, radii0)
            if s > best_score:
                best_score = s
                best_centers = centers0.copy()
                best_radii = radii0.copy()

    # Local search: perturb and expand
    if best_centers is not None:
        current_centers = best_centers.copy()
        current_radii = best_radii.copy()
        current_score = best_score

        sigma = 0.05
        for step in range(2000):
            if time.time() > deadline:
                break

            new_centers, new_radii, new_score = _perturb_and_expand(
                current_centers, current_radii, sigma, rng
            )

            if new_score is not None and new_score > current_score:
                current_centers = new_centers
                current_radii = new_radii
                current_score = new_score
                # Anneal: decrease sigma
                sigma *= 0.99
                if sigma < 0.001:
                    sigma = 0.001
            else:
                # Increase sigma occasionally
                if step % 50 == 0:
                    sigma = min(0.1, sigma * 1.1)

    # Final validation and return
    if best_centers is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Use the best found
    if current_score > best_score:
        best_centers = current_centers
        best_radii = current_radii

    # Clamp radii to ensure validity with tiny margin
    for i in range(N):
        best_radii[i] = min(best_radii[i], best_centers[i, 0], 1.0 - best_centers[i, 0],
                            best_centers[i, 1], 1.0 - best_centers[i, 1])

    result = np.zeros((N, 3))
    result[:, 0] = best_centers[:, 0]
    result[:, 1] = best_centers[:, 1]
    result[:, 2] = best_radii

    return result


# EVOLVE-BLOCK-END
