# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 620  # Leave 10s margin from 630s limit

    # Precompute indices for all 286 triangles
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    i_idx, j_idx, k_idx = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]

    def calculate_score(points):
        # points: (13, 2)
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0
        except Exception:
            return 0.0

        # Vectorized triangle area calculation
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]

        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)

        A_min = np.min(areas)

        if A_min <= 1e-10:
            return 0.0

        return A_min / A_hull

    def neg_score(points):
        s = calculate_score(points)
        return -s

    def delaunay_guided_relaxation(points, n_iter=500, cooling_rate=0.999):
        """
        Physics-based relaxation using Delaunay triangulation structure.
        Points are moved to maximize minimum triangle area by balancing
        repulsive forces (from small triangles) and attractive forces (toward
        centroid of adjacent triangles).
        """
        points = points.copy()
        T = 1.0  # Temperature for simulated annealing
        rng_local = np.random.default_rng(123)

        for iteration in range(n_iter):
            if time.time() - start_time > time_limit - 10:
                break

            # Compute all triangle areas
            pi = points[i_idx]
            pj = points[j_idx]
            pk = points[k_idx]
            cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
            areas = 0.5 * np.abs(cross)
            A_min = np.min(areas)

            # Get Delaunay triangulation
            try:
                tri = scipy.spatial.Delaunay(points)
                simplices = tri.simplices  # (m, 3) array of triangle indices
            except Exception:
                # Fallback: use all triangles
                simplices = tri_indices

            # Compute forces for each point
            forces = np.zeros_like(points)

            # For each triangle in the Delaunay triangulation
            for simplex in simplices:
                i, j, k = simplex
                area = 0.5 * np.abs(
                    (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
                    (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                )

                # Force is proportional to how much smaller this triangle is compared to A_min
                # We want to increase small triangles, so push vertices apart
                if area < 2 * A_min:
                    # Push vertices away from each other
                    centroid = np.mean(points[[i, j, k]], axis=0)
                    for idx in [i, j, k]:
                        diff = points[idx] - centroid
                        dist = np.linalg.norm(diff)
                        if dist > 1e-10:
                            # Repulsive force away from centroid
                            strength = (2 * A_min - area) / A_min
                            forces[idx] += strength * diff / dist

            # Also add global repulsion to prevent clustering
            for i in range(n):
                for j in range(i + 1, n):
                    diff = points[i] - points[j]
                    dist = np.linalg.norm(diff)
                    if dist < 0.3:  # Only for close points
                        strength = (0.3 - dist) / 0.3
                        forces[i] += strength * diff / dist
                        forces[j] -= strength * diff / dist

            # Add small random perturbation (simulated annealing)
            forces += T * rng_local.normal(0, 0.01, (n, 2))

            # Update points
            step_size = 0.01
            new_points = points + step_size * forces

            # Check if new configuration is better
            old_score = calculate_score(points)
            new_score = calculate_score(new_points)

            if new_score >= old_score:
                points = new_points
            else:
                # Accept with probability exp(-(new_score - old_score) / T)
                delta = old_score - new_score
                if delta > 0:
                    accept_prob = np.exp(-delta / (T + 1e-10))
                    if rng_local.random() < accept_prob:
                        points = new_points

            # Cool down
            T *= cooling_rate

        return points

    best_score = 0.0
    best_points = None

    # Generate initial guesses
    initial_guesses = []

    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_guesses.append(reg_poly)

    rng = np.random.default_rng(42)

    # 2. Jittered Regular Polygon (20 variations)
    for _ in range(20):
        jitter_scale = 0.1 + 0.3 * rng.random()
        jitter = rng.normal(0, jitter_scale, (n, 2))
        guess = reg_poly + jitter
        initial_guesses.append(guess)

    # 3. Random points (20 variations)
    for _ in range(20):
        guess = rng.random((n, 2))
        initial_guesses.append(guess)

    # 4. "Square" based (4x4 grid subset)
    grid_x = np.linspace(0, 1, 4)
    grid_y = np.linspace(0, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    for _ in range(10):
        idx = rng.choice(16, 13, replace=False)
        guess = grid_points[idx]
        initial_guesses.append(guess)

    # 5. Near-optimal structured pattern: 3 concentric rings
    center = np.array([[0.0, 0.0]])
    sq_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    sq_r = 0.5
    sq_pts = np.column_stack([sq_r * np.cos(sq_angles), sq_r * np.sin(sq_angles)])
    oct_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    oct_r = 1.0
    oct_pts = np.column_stack([oct_r * np.cos(oct_angles), oct_r * np.sin(oct_angles)])
    ring_guess = np.vstack([center, sq_pts, oct_pts])
    initial_guesses.append(ring_guess)

    # Jittered ring patterns
    for _ in range(10):
        jitter = rng.normal(0, 0.1, (n, 2))
        initial_guesses.append(ring_guess + jitter)

    # Optimization Loop with two phases:
    # Phase 1: Delaunay-guided relaxation
    # Phase 2: Nelder-Mead refinement

    for guess in initial_guesses:
        if time.time() - start_time > time_limit - 30:
            break

        # Normalize
        guess_centered = guess - np.mean(guess, axis=0)
        max_dist = np.max(np.linalg.norm(guess_centered, axis=1))
        if max_dist > 1e-10:
            guess_normalized = guess_centered / max_dist
        else:
            guess_normalized = guess_centered

        # Phase 1: Delaunay-guided relaxation
        relaxed_points = delaunay_guided_relaxation(guess_normalized, n_iter=200)

        # Evaluate relaxed points
        relaxed_score = calculate_score(relaxed_points)
        if relaxed_score > best_score:
            best_score = relaxed_score
            best_points = relaxed_points.copy()

        # Phase 2: Nelder-Mead refinement from relaxed points
        if time.time() - start_time > time_limit - 15:
            continue

        remaining = time_limit - (time.time() - start_time)
        maxiter = min(3000, max(300, int(remaining * 30)))

        try:
            result = scipy.optimize.minimize(
                neg_score,
                relaxed_points,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )

            current_score = -result.fun
            if current_score > best_score:
                best_score = current_score
                best_points = result.x.copy()
        except Exception:
            pass

    # If time remains, refine the best point with Powell
    if best_points is not None and time.time() - start_time < time_limit - 40:
        try:
            result_powell = scipy.optimize.minimize(
                neg_score,
                best_points,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if -result_powell.fun > best_score:
                best_score = -result_powell.fun
                best_points = result_powell.x.copy()
        except Exception:
            pass

    # Final polish: coordinate descent on the most constrained triangles
    if best_points is not None and time.time() - start_time < time_limit - 5:
        try:
            current_points = best_points.copy()
            for _ in range(30):
                if time.time() - start_time > time_limit - 2:
                    break
                # Find the 5 smallest triangles
                pi = current_points[i_idx]
                pj = current_points[j_idx]
                pk = current_points[k_idx]
                cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
                areas = 0.5 * np.abs(cross)

                # Get indices of points in the 5 smallest triangles
                small_tri_idx = np.argsort(areas)[:5]
                involved_points = set()
                for ti in small_tri_idx:
                    involved_points.add(int(i_idx[ti]))
                    involved_points.add(int(j_idx[ti]))
                    involved_points.add(int(k_idx[ti]))

                # Try small perturbations of involved points
                improved = False
                for pt_idx in list(involved_points)[:5]:
                    for dx in [-0.01, 0.01]:
                        for dy in [-0.01, 0.01]:
                            test_points = current_points.copy()
                            test_points[pt_idx, 0] += dx
                            test_points[pt_idx, 1] += dy
                            test_score = calculate_score(test_points)
                            if test_score > best_score:
                                best_score = test_score
                                current_points = test_points
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
            best_points = current_points
        except Exception:
            pass

    if best_points is None:
        best_points = initial_guesses[0]

    return best_points

# EVOLVE-BLOCK-END
