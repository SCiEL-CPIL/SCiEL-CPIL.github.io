"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius[mask], kind="stable")[:n]
            return points[idx]
        spacing *= 0.98


def _compute_score(points):
    """Compute the packing score: min of pairwise dist/2 and boundary dist."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)


def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 150s kill

    # Start with hexagonal lattice
    centers = _hexagonal_seed(N)
    
    # Precompute distance matrix for fast updates
    def calc_dist_matrix(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=-1)

    dist_mat = calc_dist_matrix(centers)
    np.fill_diagonal(dist_mat, np.inf)

    def get_min_radius(dist_m, c):
        min_pair = np.min(dist_m) / 2.0
        radii = np.linalg.norm(c, axis=1)
        min_bound = np.min(1.0 - radii)
        return min(min_pair, min_bound)

    current_r = get_min_radius(dist_mat, centers)
    best_centers = centers.copy()
    best_r = current_r

    # Store boundary distances for fast boundary check
    boundary_dists = 1.0 - np.linalg.norm(centers, axis=1)
    min_boundary_other = np.min(boundary_dists)

    step_size = 0.05
    no_improve_count = 0

    while time.time() - start_time < timeout:
        # Randomly pick a circle to move
        i = np.random.randint(N)

        # Try several random moves
        for _ in range(20):
            delta = np.random.normal(0, step_size, size=2)
            new_c = centers[i] + delta
            r_new_c = np.linalg.norm(new_c)
            if r_new_c > 1.0:
                continue

            # Save old row for rollback
            old_row = dist_mat[i].copy()

            # Update distance matrix row/col i
            new_dists = np.linalg.norm(centers - new_c, axis=1)
            new_dists[i] = np.inf
            dist_mat[i] = new_dists
            dist_mat[:, i] = new_dists

            # Compute candidate radius
            min_pair = np.min(dist_mat) / 2.0
            # Boundary: replace i-th element
            boundary_dists[i] = 1.0 - r_new_c
            min_bound = np.min(boundary_dists)

            candidate_r = min(min_pair, min_bound)

            if candidate_r > current_r + 1e-12:
                # Accept the move
                centers[i] = new_c
                current_r = candidate_r
                no_improve_count = 0

                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()

                # Slightly increase step size on success
                step_size *= 1.01
            else:
                # Revert
                dist_mat[i] = old_row
                dist_mat[:, i] = old_row
                boundary_dists[i] = 1.0 - np.linalg.norm(centers[i])
                no_improve_count += 1

        # Periodically reduce step size
        if no_improve_count > 50:
            step_size *= 0.9
            no_improve_count = 0
            if step_size < 0.001:
                step_size = 0.05  # Reset to encourage exploration

    # Final validation: ensure all points are inside the unit disk
    radii = np.linalg.norm(best_centers, axis=1)
    if np.any(radii >= 1.0):
        best_centers = best_centers * (1.0 - 1e-8) / radii

    return best_centers
# EVOLVE-BLOCK-END
