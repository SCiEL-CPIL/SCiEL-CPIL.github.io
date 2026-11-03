"""Structural change: hexagonal quasi-crystal + deterministic corner-aware relaxation."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0  # safety margin below 150s kill


def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _hexagonal_lattice(n_points):
    """Generate a hexagonal lattice of approximately n_points within [-0.5,0.5]^2."""
    # Hexagonal lattice: spacing s, rows offset
    # Try to fit as many as possible, then trim to N
    s = 0.06  # initial spacing guess; will be refined
    # Generate points on hex grid
    dx = s
    dy = s * np.sqrt(3) / 2
    points = []
    # Cover the square with margin
    margin = 0.01
    x_min, x_max = -0.5 + margin, 0.5 - margin
    y_min, y_max = -0.5 + margin, 0.5 - margin

    row = 0
    y = y_min
    while y <= y_max:
        x_offset = (dx / 2) if (row % 2 == 1) else 0.0
        x = x_min + x_offset
        while x <= x_max:
            points.append((x, y))
            x += dx
        y += dy
        row += 1

    points = np.array(points)
    if len(points) < n_points:
        # If we don't have enough, reduce spacing
        s *= 0.95
        return _hexagonal_lattice(n_points)

    # Trim to n_points: choose the subset that minimizes boundary distance
    # Simple approach: sort by distance from center, keep closest n_points
    center_dist = np.sum(points ** 2, axis=1)
    idx = np.argsort(center_dist)[:n_points]
    return points[idx]


def _relax(centers, max_iters=5000):
    """Deterministic corner-aware relaxation: push apart overlapping pairs, pull back from walls."""
    centers = centers.copy()
    r_best = _compute_radius(centers)

    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break

        # Compute all pairwise distances
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)

        # Find worst pair (minimum distance)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        min_d = dists[i, j]

        # Find worst boundary violation
        boundary_dist = np.minimum(
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1])
        )
        min_boundary_idx = np.argmin(boundary_dist)
        min_boundary = boundary_dist[min_boundary_idx]

        # Determine the limiting factor
        limiting_pair = min_d / 2.0
        limiting_boundary = min_boundary

        if limiting_pair < limiting_boundary:
            # Push apart the worst pair
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                vec = np.array([0.01, 0.0])
                norm = np.linalg.norm(vec)
            unit = vec / norm
            # Push each by half the deficit
            target_d = max(min_d + 0.001, 0.005)
            push = (target_d - min_d) / 2.0
            centers[i] += unit * push
            centers[j] -= unit * push
        else:
            # Pull the worst boundary point inward
            c = centers[min_boundary_idx]
            # Determine which wall is closest
            dist_left = c[0] + 0.5
            dist_right = 0.5 - c[0]
            dist_bottom = c[1] + 0.5
            dist_top = 0.5 - c[1]
            min_wall_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            if min_wall_dist == dist_left:
                c[0] += 0.001
            elif min_wall_dist == dist_right:
                c[0] -= 0.001
            elif min_wall_dist == dist_bottom:
                c[1] += 0.001
            else:
                c[1] -= 0.001
            centers[min_boundary_idx] = c

        # Compute current radius
        r_cur = _compute_radius(centers)
        if r_cur > r_best:
            r_best = r_cur

    return centers, r_best


def construct():
    start_time = time.time()

    # Generate hexagonal lattice
    centers = _hexagonal_lattice(N)

    # Initial relaxation
    centers, r_best = _relax(centers, max_iters=3000)

    # Try a second pass with different initial spacing to escape local minima
    if time.time() < _DEADLINE - 10:
        # Perturb the lattice slightly and re-relax
        np.random.seed(42)
        perturbed = centers + np.random.normal(0, 0.005, size=centers.shape)
        # Clip to square
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
        centers2, r2 = _relax(perturbed, max_iters=2000)
        if r2 > r_best:
            centers = centers2
            r_best = r2

    # Final validation and return
    r_final = _compute_radius(centers)
    if r_final < 0:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]

    return centers
# EVOLVE-BLOCK-END
