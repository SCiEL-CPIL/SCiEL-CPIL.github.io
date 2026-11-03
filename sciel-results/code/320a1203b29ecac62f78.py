"""Optimized sphere packing in a unit cube."""
import numpy as np
import time

N = 100
DEADLINE = 150.0  # seconds wall-clock limit
SAFETY_MARGIN = 5.0  # seconds to return before deadline


def _score(centers):
    """Compute the radius: min of half min pairwise distance and min distance to wall."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=-1)
    min_wall = np.min(wall_dists)

    return min(min_pair, min_wall)


def _construct():
    start_time = time.time()

    # Initial grid placement
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()

    best_centers = centers.copy()
    best_score = _score(centers)

    # Repulsion-based local optimization with adaptive step size
    rng = np.random.RandomState(42)
    step_size = 0.01
    patience = 0
    max_patience = 500

    while time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        # Compute forces from repulsion
        forces = np.zeros_like(centers)
        for i in range(N):
            diff = centers[i] - centers
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, 1e-10)
            # Repulsion force: stronger when closer
            repulsion = 1.0 / (dists ** 2)
            forces[i] = np.sum(diff * repulsion[:, np.newaxis], axis=0)

        # Wall repulsion
        for i in range(N):
            for k in range(3):
                wall_dist = 0.5 - abs(centers[i, k])
                if wall_dist < 0.1:
                    force_mag = 1.0 / (wall_dist + 1e-6)
                    forces[i, k] += np.sign(centers[i, k]) * force_mag

        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=-1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces = forces / max_force

        # Move centers
        new_centers = centers + step_size * forces

        # Clamp to cube
        new_centers = np.clip(new_centers, -0.5, 0.5)

        # Check validity
        if not np.all(np.isfinite(new_centers)):
            break

        new_score = _score(new_centers)

        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers
            patience = 0
        else:
            centers = new_centers  # Still move, but track patience
            patience += 1

        # Adaptive step size
        if patience > max_patience:
            step_size *= 0.5
            patience = 0
            if step_size < 1e-6:
                break
        elif patience % 100 == 0 and patience < max_patience:
            step_size *= 0.9

    return best_centers


def construct():
    """Construct centers of 100 equal spheres in unit cube [-1/2, 1/2]^3."""
    centers = _construct()
    return centers
