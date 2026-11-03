import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    def compute_radius(positions):
        """Compute the actual radius from positions."""
        from scipy.spatial.distance import cdist
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)

        xs = positions[:, 0]
        ys = positions[:, 1]
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        min_boundary_dist = np.min(dist_b)

        return min(min_pair_dist / 2, min_boundary_dist)

    def smooth_overlap_energy(positions, r, beta=50.0):
        """
        Smooth surrogate for the overlap energy using soft-min.
        """
        from scipy.spatial.distance import cdist
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, 2*r + 1)

        xs = positions[:, 0]
        ys = positions[:, 1]

        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)

        # Pairwise: smooth max(0, 2r - d)^2
        pair_diff = 2*r - dists
        pair_diff_clipped = np.clip(pair_diff, -20/beta, 20/beta)
        sig = 1.0 / (1.0 + np.exp(-beta * pair_diff_clipped))
        pair_energy = np.sum((pair_diff**2) * sig)

        # Boundary: smooth max(0, r - dist_b)^2
        bound_diff = r - dist_b
        bound_diff_clipped = np.clip(bound_diff, -20/beta, 20/beta)
        sig_b = 1.0 / (1.0 + np.exp(-beta * bound_diff_clipped))
        boundary_energy = np.sum((bound_diff**2) * sig_b)

        return pair_energy + boundary_energy

    def hard_overlap_energy(positions, r):
        """Original hard overlap energy for feasibility checking."""
        from scipy.spatial.distance import cdist
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, 2*r + 1)

        xs = positions[:, 0]
        ys = positions[:, 1]

        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)

        pair_overlap = np.maximum(0, 2*r - dists)
        pair_energy = np.sum(pair_overlap**2)

        boundary_overlap = np.maximum(0, r - dist_b)
        boundary_energy = np.sum(boundary_overlap**2)

        return pair_energy + boundary_energy

    def hexagonal_init(n, r_target=0.036):
        """Create a hexagonal grid centered in the square."""
        s = 2 * r_target * 1.001
        points = []
        for i in range(-15, 15):
            for j in range(-15, 15):
                x = j * s + (i % 2) * s / 2
                y = i * s * np.sqrt(3) / 2
                points.append([x, y])

        points = np.array(points)
        mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
               (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        points = points[mask]

        if len(points) > n:
            dists_to_center = np.sum(points**2, axis=1)
            dist_x = np.minimum(0.5 - points[:, 0], points[:, 0] + 0.5)
            dist_y = np.minimum(0.5 - points[:, 1], points[:, 1] + 0.5)
            dist_b = np.minimum(dist_x, dist_y)
            score = dists_to_center - 0.1 * dist_b
            idx = np.argsort(score)[:n]
            points = points[idx]
        elif len(points) < n:
            needed = n - len(points)
            extra = np.random.uniform(-0.4, 0.4, size=(needed, 2))
            points = np.vstack([points, extra])

        return points

    def square_grid_init(n):
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]

    def graph_polish(positions, r_target, max_iter=50, tol=1e-10):
        """
        Graph-based polish: iteratively move points to equalize
        distances to their nearest neighbors and boundaries.
        Uses a Delaunay-based proximity graph.
        """
        if elapsed() > deadline - 10:
            return positions

        pos = positions.copy()
        bounds_lo = -0.4999
        bounds_hi = 0.4999

        for it in range(max_iter):
            if elapsed() > deadline - 10:
                break

            # Compute current radius
            cur_r = compute_radius(pos)
            if cur_r >= r_target:
                break

            # Compute pairwise distances
            from scipy.spatial.distance import cdist
            dists = cdist(pos, pos)
            np.fill_diagonal(dists, np.inf)

            # For each point, find its nearest neighbor distance
            min_pair = np.min(dists, axis=1)

            # Boundary distances
            xs = pos[:, 0]
            ys = pos[:, 1]
            dist_x = np.minimum(0.5 - xs, xs + 0.5)
            dist_y = np.minimum(0.5 - ys, ys + 0.5)
            dist_b = np.minimum(dist_x, dist_y)

            # Target: each point should have min(min_pair, dist_b) >= r_target
            # For points that are too close to boundary, push inward
            # For points that are too close to neighbors, push apart

            # Compute displacement for each point
            disp = np.zeros_like(pos)

            # Boundary correction
            for i in range(N):
                if dist_b[i] < r_target:
                    # Push toward center
                    cx, cy = pos[i]
                    # Determine which boundary is closest
                    if 0.5 - cx < r_target:  # right boundary
                        disp[i, 0] -= (r_target - (0.5 - cx)) * 0.5
                    if cx + 0.5 < r_target:  # left boundary
                        disp[i, 0] += (r_target - (cx + 0.5)) * 0.5
                    if 0.5 - cy < r_target:  # top boundary
                        disp[i, 1] -= (r_target - (0.5 - cy)) * 0.5
                    if cy + 0.5 < r_target:  # bottom boundary
                        disp[i, 1] += (r_target - (cy + 0.5)) * 0.5

            # Neighbor correction: for each pair too close, push apart
            # Use only nearest neighbors for efficiency
            for i in range(N):
                if min_pair[i] < 2 * r_target:
                    # Find nearest neighbor
                    j = np.argmin(dists[i])
                    d = dists[i, j]
                    if d < 1e-12:
                        continue
                    # Push i away from j
                    direction = (pos[i] - pos[j]) / d
                    overlap = 2 * r_target - d
                    disp[i] += direction * overlap * 0.25
                    disp[j] -= direction * overlap * 0.25

            # Apply displacement with clipping
            pos += disp
            pos = np.clip(pos, bounds_lo, bounds_hi)

            # Check convergence
            new_r = compute_radius(pos)
            if abs(new_r - cur_r) < tol:
                break

        return pos

    def polish(positions, r_target, max_iter=200, tol=1e-10):
        """High-precision polish pass with tight tolerances."""
        if elapsed() > deadline - 10:
            return positions

        bounds = [(-0.5, -0.5), (0.5, 0.5)] * N

        try:
            result = minimize(
                lambda x: smooth_overlap_energy(x, r_target, beta=100.0),
                positions,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < 1e-8:
                return result.x
        except Exception:
            pass

        try:
            result = minimize(
                lambda x: hard_overlap_energy(x, r_target),
                positions,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < 1e-8:
                return result.x
        except Exception:
            pass

        return positions

    def delaunay_graph_optimize(positions, r_target, max_iter=100):
        """
        Use Delaunay triangulation to identify critical edges and
        optimize to equalize edge lengths.
        """
        if elapsed() > deadline - 10:
            return positions

        pos = positions.copy()

        for it in range(max_iter):
            if elapsed() > deadline - 10:
                break

            # Build Delaunay triangulation
            try:
                tri = Delaunay(pos)
            except Exception:
                break

            # Get all edges from the triangulation
            edges = np.unique(tri.edges, axis=0)

            # Compute edge lengths
            edge_lengths = np.sqrt(np.sum((pos[edges[:, 0]] - pos[edges[:, 1]])**2, axis=1))

            # Find the shortest edge (most critical)
            min_edge_idx = np.argmin(edge_lengths)
            min_edge_len = edge_lengths[min_edge_idx]

            if min_edge_len >= 2 * r_target:
                break

            # Get the two points of the shortest edge
            i, j = edges[min_edge_idx]

            # Compute displacement to increase this edge length
            direction = (pos[i] - pos[j])
            d = np.linalg.norm(direction)
            if d < 1e-12:
                direction = np.array([1.0, 0.0])
                d = 1.0
            else:
                direction = direction / d

            # Also consider boundary constraints
            # Push i and j apart along the edge direction
            overlap = 2 * r_target - min_edge_len
            step = overlap * 0.3

            # Check if moving i would violate boundary
            new_i = pos[i] + direction * step
            new_j = pos[j] - direction * step

            # Clip to boundary
            new_i = np.clip(new_i, -0.4999, 0.4999)
            new_j = np.clip(new_j, -0.4999, 0.4999)

            # If clipping reduced the movement, adjust
            actual_step_i = np.linalg.norm(new_i - pos[i])
            actual_step_j = np.linalg.norm(new_j - pos[j])

            pos[i] = new_i
            pos[j] = new_j

        return pos

    best_radius = -np.inf
    best_positions = None

    # Multiple initializations with different random seeds
    np.random.seed(42)
    inits = [
        hexagonal_init(N, 0.036),
        hexagonal_init(N, 0.035),
        square_grid_init(N),
        np.random.uniform(-0.45, 0.45, size=(N, 2))
    ]

    for init_pos in inits:
        if elapsed() > deadline - 20:
            break

        current_pos = init_pos.copy()

        # Start with a conservative radius and inflate
        r = 0.025
        step = 0.002

        while r < 0.042:
            if elapsed() > deadline - 15:
                break

            # Optimize with smooth energy
            try:
                result = minimize(
                    lambda x: smooth_overlap_energy(x, r, beta=50.0),
                    current_pos,
                    method='L-BFGS-B',
                    bounds=[(-0.5, -0.5), (0.5, 0.5)] * N,
                    options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
                )
                current_pos = result.x
            except Exception:
                break

            if elapsed() > deadline - 10:
                break

            hard_energy = hard_overlap_energy(current_pos, r)

            if hard_energy < 1e-6:
                # Feasible! Try to increase r
                r += step
            else:
                # Infeasible. Binary search for the max feasible r
                r_low = r - step
                r_high = r

                for _ in range(12):
                    if elapsed() > deadline - 10:
                        break
                    r_mid = (r_low + r_high) / 2

                    try:
                        res_mid = minimize(
                            lambda x: smooth_overlap_energy(x, r_mid, beta=50.0),
                            current_pos,
                            method='L-BFGS-B',
                            bounds=[(-0.5, -0.5), (0.5, 0.5)] * N,
                            options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30}
                        )
                        current_pos = res_mid.x
                        energy_mid = hard_overlap_energy(current_pos, r_mid)
                        if energy_mid < 1e-6:
                            r_low = r_mid
                        else:
                            r_high = r_mid
                    except Exception:
                        break

                if r_low > best_radius:
                    best_radius = r_low
                    best_positions = current_pos.copy()

                break

        if r >= 0.042:
            try:
                hard_energy = hard_overlap_energy(current_pos, r)
                if hard_energy < 1e-6 and r > best_radius:
                    best_radius = r
                    best_positions = current_pos.copy()
            except Exception:
                pass

    # Fallback if no valid position found
    if best_positions is None:
        best_positions = square_grid_init(N)

    # Final high-precision polish
    if best_positions is not None and elapsed() < deadline - 15:
        actual_r = compute_radius(best_positions)

        # Try graph-based polish
        if elapsed() < deadline - 15:
            graph_pos = graph_polish(best_positions, actual_r + 0.0001, max_iter=50)
            graph_r = compute_radius(graph_pos)
            if graph_r > best_radius:
                best_radius = graph_r
                best_positions = graph_pos

        # Try Delaunay-based optimization
        if elapsed() < deadline - 15:
            del_pos = delaunay_graph_optimize(best_positions, actual_r + 0.0001, max_iter=100)
            del_r = compute_radius(del_pos)
            if del_r > best_radius:
                best_radius = del_r
                best_positions = del_pos

        # Standard polish at slightly higher radius
        for r_try in [actual_r + 0.0001, actual_r + 0.00005, actual_r]:
            if elapsed() > deadline - 10:
                break
            polished = polish(best_positions, r_try, max_iter=300, tol=1e-12)
            r_polished = compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_positions = polished

    # Final validation and clamping
    best_positions = np.clip(best_positions, -0.49999, 0.49999)

    return best_positions
