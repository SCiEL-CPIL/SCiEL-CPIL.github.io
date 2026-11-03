import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(n_rows, n_cols, d_scale=1.0):
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d / 2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d / 2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def square_grid_init(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def get_bottleneck(centers):
        """Return (bottleneck_value, bottleneck_type, indices)."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        min_wall_val = wall_dists[min_wall_idx]

        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        min_pair_val = dist_matrix[min_pair_idx] / 2.0

        if min_wall_val <= min_pair_val:
            return min_wall_val, 'wall', (min_wall_idx[0], min_wall_idx[1])
        else:
            return min_pair_val, 'pair', (min_pair_idx[0], min_pair_idx[1])

    def try_improve_point(centers, idx, direction, magnitude, tree):
        """Try moving point idx in given direction by magnitude; return new radius if improved."""
        new_pos = centers[idx].copy()
        new_pos += direction * magnitude
        new_pos = np.clip(new_pos, -0.5, 0.5)

        if np.allclose(new_pos, centers[idx]):
            return None, None

        old_pos = centers[idx].copy()
        centers[idx] = new_pos

        # Quick check: compute new bottleneck
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)

        # For pair distances, we only need to check distances involving idx
        # since only idx moved
        others = np.delete(centers, idx, axis=0)
        if len(others) > 0:
            d_min, _ = tree.query(new_pos)
            min_pair = d_min / 2.0
        else:
            min_pair = 1.0

        new_r = min(min_wall, min_pair)

        # Also need to check if this creates a new global bottleneck elsewhere
        # For efficiency, just check if new_r is better than old bottleneck
        return new_r, new_pos

    def local_search(centers, time_limit=5.0):
        """Direct bottleneck-elimination local search."""
        centers = centers.copy()
        start = time.time()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        no_improve_count = 0
        max_no_improve = 50

        while time.time() < start + time_limit:
            if no_improve_count > max_no_improve:
                break

            # Get current bottleneck
            bottleneck_val, bottleneck_type, indices = get_bottleneck(centers)

            if bottleneck_val >= current_r - 1e-12:
                # Bottleneck matches computed radius, try to eliminate it
                if bottleneck_type == 'wall':
                    idx, coord = indices
                    # Move point away from wall
                    sign = np.sign(centers[idx, coord])
                    if sign == 0:
                        sign = 1.0
                    direction = np.zeros(2)
                    direction[coord] = sign
                    moved = False
                    for mag in [0.001, 0.002, 0.005, 0.01, 0.02]:
                        new_pos = centers[idx].copy()
                        new_pos[coord] += sign * mag
                        if np.all(np.abs(new_pos) <= 0.5):
                            old_pos = centers[idx].copy()
                            centers[idx] = new_pos
                            new_r = compute_radius(centers)
                            if new_r > current_r + 1e-12:
                                current_r = new_r
                                no_improve_count = 0
                                if current_r > best_r:
                                    best_r = current_r
                                    best_centers = centers.copy()
                                moved = True
                                break
                            else:
                                centers[idx] = old_pos
                    if not moved:
                        no_improve_count += 1
                else:
                    i, j = indices
                    # Move the two points apart
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-12:
                        # Coincident, randomize
                        centers[i] += np.random.randn(2) * 0.01
                        centers[j] += np.random.randn(2) * 0.01
                        centers = np.clip(centers, -0.5, 0.5)
                        new_r = compute_radius(centers)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            no_improve_count = 0
                            if current_r > best_r:
                                best_r = current_r
                                best_centers = centers.copy()
                        else:
                            no_improve_count += 1
                        continue

                    unit_diff = diff / dist
                    moved = False
                    for mag in [0.001, 0.002, 0.005, 0.01, 0.02]:
                        old_i = centers[i].copy()
                        old_j = centers[j].copy()
                        centers[i] = old_i + unit_diff * mag
                        centers[j] = old_j - unit_diff * mag
                        centers = np.clip(centers, -0.5, 0.5)
                        new_r = compute_radius(centers)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            no_improve_count = 0
                            if current_r > best_r:
                                best_r = current_r
                                best_centers = centers.copy()
                            moved = True
                            break
                        else:
                            centers[i] = old_i
                            centers[j] = old_j
                    if not moved:
                        no_improve_count += 1
            else:
                # Bottleneck is larger than computed radius? Shouldn't happen, but try random moves
                no_improve_count += 1

        return best_centers, best_r

    def sa_local_search(centers, time_limit=10.0, T_init=0.01, T_min=0.0001, alpha=0.995):
        """Simulated annealing local search on the exact objective."""
        centers = centers.copy()
        start = time.time()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        T = T_init

        while time.time() < start + time_limit and T > T_min:
            # Pick a random point to move
            idx = np.random.randint(N)
            old_pos = centers[idx].copy()

            # Random perturbation
            step_size = 0.005 + 0.02 * (T / T_init)
            new_pos = old_pos + np.random.randn(2) * step_size
            new_pos = np.clip(new_pos, -0.5, 0.5)

            centers[idx] = new_pos
            new_r = compute_radius(centers)
            delta = new_r - current_r

            if delta > 0 or np.random.random() < math.exp(delta / T):
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                centers[idx] = old_pos

            T *= alpha

        return best_centers, best_r

    # Main logic
    best_centers = None
    best_radius = 0.0

    # Generate diverse initial configurations
    inits = []

    # Hexagonal grids with various parameters
    for n_rows in range(13, 17):
        if time.time() > deadline - 80:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05, 1.1]:
            pts = hexagonal_init(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)

    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    inits.append(grid)

    # Slightly perturbed grids
    for _ in range(2):
        if time.time() > deadline - 80:
            break
        perturbed = grid + np.random.randn(N, 2) * 0.01
        perturbed = np.clip(perturbed, -0.49, 0.49)
        inits.append(perturbed)

    # Random
    for _ in range(2):
        if time.time() > deadline - 80:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)

    # For each init, run local search + SA
    for init in inits:
        if time.time() > deadline - 20:
            break

        # Quick eval
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()

        # Phase 1: Direct bottleneck-elimination local search
        if time.time() < deadline - 15:
            ls_centers, ls_r = local_search(init, time_limit=5.0)
            if ls_r > best_radius:
                best_radius = ls_r
                best_centers = ls_centers.copy()

        # Phase 2: Simulated annealing
        if time.time() < deadline - 10:
            sa_centers, sa_r = sa_local_search(best_centers, time_limit=8.0)
            if sa_r > best_radius:
                best_radius = sa_r
                best_centers = sa_centers.copy()

        # Phase 3: Another round of local search after SA
        if time.time() < deadline - 5:
            ls2_centers, ls2_r = local_search(best_centers, time_limit=4.0)
            if ls2_r > best_radius:
                best_radius = ls2_r
                best_centers = ls2_centers.copy()

    # Final polish if time remains
    if time.time() < deadline - 2 and best_centers is not None:
        final_ls, final_r = local_search(best_centers, time_limit=2.0)
        if final_r > best_radius:
            best_radius = final_r
            best_centers = final_ls.copy()

    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
