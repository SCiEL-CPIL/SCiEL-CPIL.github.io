"""Improved packing: hexagonal lattice init + targeted bottleneck elimination + smooth polish."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

_DEADLINE = 150.0  # wall-clock seconds; keep a safety margin


def _compute_radius(centres: np.ndarray) -> float:
    """Compute the largest common radius for given centres in [-0.5, 0.5]^2."""
    if centres.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Wall constraint: min distance from each centre to the four walls
    wall_dists = np.minimum(
        np.minimum(centres[:, 0] + 0.5, -centres[:, 0] + 0.5),
        np.minimum(centres[:, 1] + 0.5, -centres[:, 1] + 0.5)
    )
    min_wall = float(np.min(wall_dists))
    # Pairwise constraint
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0)  # all pairs within 1.0 (all of them for 200 pts)
    if len(pairs) == 0:
        min_pair = float('inf')
    else:
        min_pair = float('inf')
        for i, j in pairs:
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair:
                min_pair = d
    return min(min_wall, min_pair / 2.0)


def _init_hexagonal(n: int) -> np.ndarray:
    """Generate a hexagonal-lattice-based initial packing for n circles in unit square."""
    # For N=200, try various hex lattice spacings
    best_centres = None
    best_r = 0.0

    # Hex lattice: rows offset by half spacing
    # Try different numbers of rows/cols
    for n_rows in range(12, 20):
        n_cols = int(np.ceil(n / n_rows))
        if n_rows * n_cols < n:
            continue
        # Spacing to fit in unit square
        # Horizontal spacing: 1.0 / (n_cols - 1) if n_cols > 1
        # Vertical spacing: sqrt(3)/2 * horizontal for hex
        # But we need to fit n_rows rows
        # Let's compute: total width = (n_cols - 1) * dx, total height = (n_rows - 1) * dy
        # where dy = sqrt(3)/2 * dx for hexagonal
        # Fit: (n_cols-1)*dx <= 1 and (n_rows-1)*dy <= 1
        # => dx <= 1/(n_cols-1) and dx <= 2/(sqrt(3)*(n_rows-1))
        dx_max = min(1.0 / max(n_cols - 1, 1), 2.0 / (np.sqrt(3) * max(n_rows - 1, 1)))
        dx = dx_max
        dy = np.sqrt(3) / 2.0 * dx

        # Generate lattice points
        pts = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = (col - (n_cols - 1) / 2.0) * dx
                y = (row - (n_rows - 1) / 2.0) * dy
                # Offset odd rows
                if row % 2 == 1:
                    x += dx / 2.0
                pts.append([x, y])
        pts = np.array(pts[:n])
        if len(pts) < n:
            continue

        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_centres = pts.copy()

    if best_centres is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:n]

    return best_centres


def _targeted_bottleneck_elimination(centres: np.ndarray, max_iter: int = 50000) -> np.ndarray:
    """
    Focus computation on the critical constraint: find the closest pair or wall violation,
    and move the involved points to increase the minimum radius.
    """
    centres = centres.copy()
    n = len(centres)

    for iteration in range(max_iter):
        # Find current bottleneck
        tree = cKDTree(centres)
        # Find closest pair
        pairs = tree.query_pairs(1.0)
        if len(pairs) == 0:
            min_dist = float('inf')
            min_pair = None
        else:
            min_dist = float('inf')
            min_pair = None
            for i, j in pairs:
                d = np.linalg.norm(centres[i] - centres[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)

        # Find wall violation
        wall_dists = np.minimum(
            np.minimum(centres[:, 0] + 0.5, -centres[:, 0] + 0.5),
            np.minimum(centres[:, 1] + 0.5, -centres[:, 1] + 0.5)
        )
        min_wall_idx = int(np.argmin(wall_dists))
        min_wall = float(wall_dists[min_wall_idx])

        # Determine which is the bottleneck
        # The radius is min(min_wall, min_dist/2)
        # If min_wall < min_dist/2, wall is bottleneck
        if min_wall < min_dist / 2.0:
            # Move the point away from the wall
            i = min_wall_idx
            # Determine which wall is closest
            x, y = centres[i]
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y

            # Move away from closest wall
            step = 0.001
            if d_left == min(d_left, d_right, d_bottom, d_top):
                centres[i, 0] += step
            elif d_right == min(d_left, d_right, d_bottom, d_top):
                centres[i, 0] -= step
            elif d_bottom == min(d_left, d_right, d_bottom, d_top):
                centres[i, 1] += step
            else:
                centres[i, 1] -= step

            # Clamp to stay in square
            centres[i] = np.clip(centres[i], -0.499, 0.499)

        elif min_pair is not None:
            # Move the two closest points apart
            i, j = min_pair
            diff = centres[j] - centres[i]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                # Coincident, random small push
                centres[i] += np.random.randn(2) * 0.001
                centres[j] -= np.random.randn(2) * 0.001
            else:
                direction = diff / norm
                step = 0.001
                centres[i] -= direction * step
                centres[j] += direction * step

            # Clamp
            centres[i] = np.clip(centres[i], -0.499, 0.499)
            centres[j] = np.clip(centres[j], -0.499, 0.499)

        else:
            break

    return centres


def _smooth_polish(centres: np.ndarray, max_iter: int = 20000, lr: float = 0.005) -> np.ndarray:
    """
    Smooth gradient-based polish using a soft-min surrogate for the min constraint.
    Maximizes the soft-min of all pairwise distances/2 and wall distances.
    """
    centres = centres.copy()
    n = len(centres)
    beta = 50.0  # sharpness of soft-min

    for iteration in range(max_iter):
        # Compute all pairwise distances
        # For efficiency with n=200, this is O(n^2) = 40000 pairs, fine
        tree = cKDTree(centres)
        pairs = tree.query_pairs(1.0)

        # Collect all constraints: pairwise distances/2 and wall distances
        # Use a vectorized approach for the soft-min
        # Soft-min: -log(sum(exp(-x_i/beta)))/beta approximates min(x_i)

        # Wall constraints
        wall_dists = np.minimum(
            np.minimum(centres[:, 0] + 0.5, -centres[:, 0] + 0.5),
            np.minimum(centres[:, 1] + 0.5, -centres[:, 1] + 0.5)
        )

        # Pairwise distances / 2
        pair_dists = []
        for i, j in pairs:
            d = np.linalg.norm(centres[i] - centres[j]) / 2.0
            pair_dists.append(d)
        pair_dists = np.array(pair_dists) if len(pair_dists) > 0 else np.array([1.0])

        # Combine all constraints
        all_constr = np.concatenate([wall_dists, pair_dists])

        # Soft-min and its gradient
        # soft_min = -beta * log(mean(exp(-x/beta)))
        # d(soft_min)/dx_i = -exp(-x_i/beta) / sum(exp(-x_j/beta))

        # For numerical stability
        min_c = np.min(all_constr)
        exps = np.exp(-(all_constr - min_c) / beta)
        sum_exps = np.sum(exps)

        # Gradient of soft-min with respect to each constraint value
        # d(soft_min)/d(constr_i) = -exps[i] / sum_exps
        grad_constr = -exps / sum_exps

        # Now propagate gradient to centres
        grad = np.zeros_like(centres)

        # Wall gradient
        # For each point, wall_dist = min(x+0.5, 0.5-x, y+0.5, 0.5-y)
        # The gradient of the min is the gradient of the active constraint
        for i in range(n):
            x, y = centres[i]
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y

            # Weight from soft-min
            w = grad_constr[i]  # d(soft_min)/d(wall_dist_i)

            # Which wall is active? Use the one that gives the min
            # For smoothness, use softmax weighting over the four walls
            wall_vals = np.array([d_left, d_right, d_bottom, d_top])
            wall_exps = np.exp(-(wall_vals - np.min(wall_vals)) / beta)
            wall_weights = wall_exps / np.sum(wall_exps)

            # Gradient contribution
            # d(wall_dist)/d(x) for left wall = 1, right wall = -1, etc.
            grad[i, 0] += w * (wall_weights[0] * 1.0 + wall_weights[1] * (-1.0))
            grad[i, 1] += w * (wall_weights[2] * 1.0 + wall_weights[3] * (-1.0))

        # Pairwise gradient
        for idx, (i, j) in enumerate(pairs):
            w = grad_constr[n + idx]  # d(soft_min)/d(pair_dist_ij)
            diff = centres[j] - centres[i]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                continue
            # d(pair_dist/2)/d(centres_i) = -diff/(2*norm)
            # d(pair_dist/2)/d(centres_j) = diff/(2*norm)
            grad[i] += w * (-diff / (2.0 * norm))
            grad[j] += w * (diff / (2.0 * norm))

        # Update
        centres += lr * grad

        # Clamp to square
        centres = np.clip(centres, -0.499, 0.499)

    return centres


def construct():
    start_time = time.time()

    # Phase 1: Hexagonal lattice initialization
    centres = _init_hexagonal(N)

    # Phase 2: Targeted bottleneck elimination
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed < 60.0:
        centres = _targeted_bottleneck_elimination(centres, max_iter=30000)

    # Phase 3: Smooth polish
    elapsed = time.time() - start_time
    if elapsed < 120.0:
        remaining = max(1.0, 120.0 - elapsed)
        # Estimate iterations based on remaining time
        # Each iteration takes roughly 1-2ms for n=200
        est_iters = int(remaining * 500)  # conservative
        centres = _smooth_polish(centres, max_iter=est_iters, lr=0.003)

    # Final check
    elapsed = time.time() - start_time
    if elapsed > 145.0:
        # Safety: return what we have
        pass

    # Ensure validity
    centres = np.clip(centres, -0.499, 0.499)

    return centres
# EVOLVE-BLOCK-END
