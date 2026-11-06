"""Packing 200 equal circles in a unit square to maximize radius.

Combines:
  - Multi-initialization (grid, random, honeycomb) from the seed
  - Vectorized repulsion with a *correct* force model (target 2*r separation)
  - Bottleneck-driven local refinement using pdist + cKDTree for O(N log N)
    nearest-pair detection
  - High-precision polish
"""
import numpy as np
import time
import random
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

N = 200
START_TIME = time.time()
TIME_LIMIT = 140.0
SAFETY_MARGIN = 10.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    if len(centers) == 0:
        return 0.0
    # Distance to each wall for each center, then min over all
    dw = np.minimum(
        np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
        np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
    )
    wall_term = float(np.min(dw))
    n = len(centers)
    if n < 2:
        return wall_term
    d = pdist(centers)
    min_pair = float(np.min(d))
    pair_term = min_pair / 2.0
    return min(wall_term, pair_term)


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1


def _initial_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    return centers


def _initial_random():
    return np.random.uniform(-0.5, 0.5, size=(N, 2))


def _initial_honeycomb():
    """Hexagonal packing initial layout, scaled to fit the unit square."""
    rows = 15
    cols = 14
    centers = []
    dx = 1.0 / (cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= N:
                break
            x = j * dx - 0.5 + dx / 2
            y = i * dy - 0.5 + dy / 2
            if i % 2 == 1:
                x += dx / 2
            x = x - (cols * dx) / 2 + 0.5
            y = y - (rows * dy) / 2 + 0.5
            x = x - 0.5
            y = y - 0.5
            centers.append([x, y])
    return np.array(centers[:N])


def _optimize_repulsion(centers, time_limit=20.0):
    """Global vectorized repulsion optimization.

    Force model: each pair repels with magnitude 1/d^p (targeting 2*r
    separation), walls push inward. This drives the minimum separation
    up rather than stopping at the first contact.
    """
    start = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()

    learning_rate = 0.002
    decay = 0.9995
    repulsion_power = 4
    wall_strength = 0.8

    for _ in range(200000):
        if time.time() - start > time_limit:
            break

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1e-10)

        repulsion = 1.0 / (dist ** repulsion_power)
        np.fill_diagonal(repulsion, 0.0)

        force_mag = repulsion * learning_rate
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)

        # Wall repulsion
        dist_left = centers[:, 0] + 0.5
        dist_right = 0.5 - centers[:, 0]
        dist_bottom = centers[:, 1] + 0.5
        dist_top = 0.5 - centers[:, 1]

        forces[:, 0] += wall_strength * learning_rate / (dist_left ** 2 + 1e-10)
        forces[:, 0] -= wall_strength * learning_rate / (dist_right ** 2 + 1e-10)
        forces[:, 1] += wall_strength * learning_rate / (dist_bottom ** 2 + 1e-10)
        forces[:, 1] -= wall_strength * learning_rate / (dist_top ** 2 + 1e-10)

        centers += forces
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)

        current_r = _compute_radius(centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()

        learning_rate *= decay

    return best_centers, best_r


def _bottleneck_local_opt(centers, time_limit=25.0):
    """Bottleneck-driven local optimization.

    Repeatedly identifies the tightest constraint (closest pair or closest
    wall) and pushes those centers apart. Uses cKDTree for fast nearest-
    neighbour lookup so each iteration is O(N log N) rather than O(N^2).
    """
    start = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()

    step_size = 0.004
    no_improve = 0

    for _ in range(200000):
        if time.time() - start > time_limit:
            break

        # Wall distances
        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_idx = int(np.argmin(dw))
        min_wall_dist = dw[min_wall_idx]

        # Closest pair via cKDTree
        tree = cKDTree(centers)
        pairs = tree.query_pairs(2.0 * best_r + 0.02)
        if len(pairs) == 0:
            # Fallback: find nearest pair
            d = pdist(centers)
            min_pair_idx = int(np.argmin(d))
            min_pair_dist = d[min_pair_idx]
            i, j = _pdist_index_to_pair(min_pair_idx, N)
        else:
            # Find the actual minimum distance pair among close pairs
            best_pair_dist = float('inf')
            i, j = 0, 1
            for (a, b) in pairs:
                dd = np.linalg.norm(centers[a] - centers[b])
                if dd < best_pair_dist:
                    best_pair_dist = dd
                    i, j = a, b
            min_pair_dist = best_pair_dist

        if min_pair_dist / 2.0 <= min_wall_dist:
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            centers[i] += step_size * direction
            centers[j] -= step_size * direction
        else:
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                centers[idx, 0] += step_size
            elif min_d == dr:
                centers[idx, 0] -= step_size
            elif min_d == db:
                centers[idx, 1] += step_size
            else:
                centers[idx, 1] -= step_size

        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)

        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 80:
                # Random kick to escape local minimum
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.004
                centers[idx] = np.clip(centers[idx], -0.5 + 1e-10, 0.5 - 1e-10)
                no_improve = 0
                step_size = max(0.0004, step_size * 0.9)
            elif no_improve > 40:
                step_size = max(0.0004, step_size * 0.95)

    return best_centers, best_r


def _polish(centers, time_limit=8.0):
    """High-precision polish with very small bottleneck steps."""
    start = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    step_size = 0.0001

    for _ in range(500000):
        if time.time() - start > time_limit:
            break

        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_idx = int(np.argmin(dw))
        min_wall_dist = dw[min_wall_idx]

        d = pdist(centers)
        min_pair_idx = int(np.argmin(d))
        min_pair_dist = d[min_pair_idx]

        if min_pair_dist / 2.0 <= min_wall_dist:
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                continue
            direction = vec / norm
            centers[i] += step_size * direction
            centers[j] -= step_size * direction
        else:
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                centers[idx, 0] += step_size
            elif min_d == dr:
                centers[idx, 0] -= step_size
            elif min_d == db:
                centers[idx, 1] += step_size
            else:
                centers[idx, 1] -= step_size

        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)

        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()

    return best_centers, best_r


def construct():
    """Construct optimal circle packing for 200 circles in a unit square."""
    random.seed(42)
    np.random.seed(42)

    best_radius = 0.0
    best_centers = None

    initializations = [
        ("grid", _initial_grid()),
        ("random", _initial_random()),
        ("honeycomb", _initial_honeycomb()),
    ]

    for name, centers in initializations:
        if _remaining_time() < 30.0:
            break

        # Global repulsion optimization
        opt_centers, opt_radius = _optimize_repulsion(centers.copy(), time_limit=18.0)

        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers.copy()

        # Bottleneck local refinement
        if _remaining_time() > 25.0:
            local_centers, local_radius = _bottleneck_local_opt(
                best_centers.copy(), time_limit=22.0
            )
            if local_radius > best_radius:
                best_radius = local_radius
                best_centers = local_centers.copy()

        # High-precision polish
        if _remaining_time() > 10.0:
            polish_centers, polish_radius = _polish(
                best_centers.copy(), time_limit=6.0
            )
            if polish_radius > best_radius:
                best_radius = polish_radius
                best_centers = polish_centers.copy()

    # Final safety: ensure validity
    if best_centers is None:
        best_centers = _initial_grid()
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    best_centers = np.asarray(best_centers, dtype=np.float64)

    # Validate shape and finiteness
    assert best_centers.shape == (N, 2)
    assert np.all(np.isfinite(best_centers))

    return best_centers
