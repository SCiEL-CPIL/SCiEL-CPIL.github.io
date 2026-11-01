"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the largest valid radius for the given centers."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Wall clearance
    wall_r = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_r = np.min(wall_r)
    if min_wall_r < -1e-12:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    pair_r = min_dist / 2.0
    return min(min_wall_r, pair_r)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill

    # Initial positions: use a perturbed grid to break symmetry
    np.random.seed(42)
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # Create a grid of m^3 points, take first N
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random perturbation to break symmetry
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    # Clip to cube
    centers = np.clip(centers, -0.49, 0.49)

    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)

    # Force-directed relaxation with annealing
    # Target radius: we want to maximize the minimum clearance
    # Use a soft potential: for pairs closer than 2*r_target, repel
    # For walls, if center is closer than r_target to wall, repel inward

    # Start with a reasonable target and increase it
    # The best known is ~0.1114
    r_target = 0.05  # Start low, will increase

    # Velocity Verlet with damping
    velocities = np.zeros_like(centers)
    dt = 0.001
    damping = 0.95

    # Annealing schedule: gradually increase r_target and decrease temperature
    n_iterations = 50000

    for it in range(n_iterations):
        if time.time() > deadline - 5.0:
            break

        # Compute forces
        forces = np.zeros_like(centers)

        # Pairwise repulsion: for each pair, if dist < 2*r_target, push apart
        # Use a smooth potential: F = k * (2*r_target - d) * unit_vec for d < 2*r_target
        # To be efficient, use numpy broadcasting but this is O(N^2)
        # For N=100, this is 10000 pairs, manageable
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)

        # Only compute forces for pairs that are too close
        threshold = 2.0 * r_target
        mask = dists < threshold
        # Avoid division by zero
        safe_dists = np.where(mask, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        # Force magnitude: proportional to overlap
        overlap = np.where(mask, threshold - dists, 0.0)
        force_mag = 100.0 * overlap  # Spring constant
        forces += np.sum(unit_vecs * force_mag[:, :, np.newaxis], axis=1)

        # Wall forces: if center is too close to wall, push inward
        # For each coordinate, if |c| + r_target > 0.5, push toward center
        for k in range(3):
            dist_to_wall = 0.5 - np.abs(centers[:, k])
            mask_wall = dist_to_wall < r_target
            if np.any(mask_wall):
                # Force direction: toward center (sign of -c)
                direction = -np.sign(centers[mask_wall, k])
                # Magnitude proportional to violation
                violation = r_target - dist_to_wall[mask_wall]
                forces[mask_wall, k] += 100.0 * violation * direction

        # Velocity Verlet update
        # v = (v + F/m * dt) * damping
        velocities = (velocities + forces * dt) * damping
        centers = centers + velocities * dt

        # Clip to cube (hard constraint)
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)

        # Every 1000 iterations, check radius and possibly increase target
        if it % 1000 == 0 and it > 0:
            current_r = _compute_radius(centers)
            if current_r > best_radius:
                best_radius = current_r
                best_centers = centers.copy()

            # Gradually increase target radius if we're making progress
            # Use a heuristic: if current_r is close to r_target, increase
            if current_r > 0.9 * r_target:
                r_target = min(r_target * 1.02, 0.12)

            # Reduce damping over time to allow settling
            damping = max(0.8, damping - 0.0001)

    # Final polish: use a local optimization to fine-tune
    # Simple gradient ascent on the radius using finite differences
    # But this is expensive. Instead, do a few more iterations with smaller dt
    dt = 0.0001
    damping = 0.9
    for it in range(10000):
        if time.time() > deadline - 1.0:
            break
        forces = np.zeros_like(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        threshold = 2.0 * r_target
        mask = dists < threshold
        safe_dists = np.where(mask, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        overlap = np.where(mask, threshold - dists, 0.0)
        force_mag = 50.0 * overlap
        forces += np.sum(unit_vecs * force_mag[:, :, np.newaxis], axis=1)
        for k in range(3):
            dist_to_wall = 0.5 - np.abs(centers[:, k])
            mask_wall = dist_to_wall < r_target
            if np.any(mask_wall):
                direction = -np.sign(centers[mask_wall, k])
                violation = r_target - dist_to_wall[mask_wall]
                forces[mask_wall, k] += 50.0 * violation * direction
        velocities = (velocities + forces * dt) * damping
        centers = centers + velocities * dt
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)

        if it % 500 == 0:
            current_r = _compute_radius(centers)
            if current_r > best_radius:
                best_radius = current_r
                best_centers = centers.copy()

    # Ensure validity
    final_r = _compute_radius(best_centers)
    if final_r <= 0:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
