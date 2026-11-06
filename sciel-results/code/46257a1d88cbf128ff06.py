"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. Multiple diverse initializations (Fibonacci, random, grid, perturbed).
    2. Fast vectorized Lloyd's relaxation on a moderate test grid.
    3. Precise convex-hull covering radius evaluation.
    4. Worst-facet targeting refinement with geodesic bisectors and
       soft-max weighting for a smooth surrogate over the top facets.
    5. Final high-precision polish pass with tight tolerance.
    6. Explicit deadline guard to return best result before timeout.
    """
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        n = np.linalg.norm(pts, axis=1, keepdims=True)
        n = np.where(n < 1e-14, 1.0, n)
        return pts / n

    def compute_covering_radius_precise(points):
        """
        Compute the covering radius using the convex hull approach.
        For each triangular facet, compute the circumcenter and check
        if it lies on the sphere (i.e., the facet's circumcircle passes
        through the origin's antipode or the facet is "empty").
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return compute_covering_radius_sampled(points, 200000)

        max_circumradius = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]

            e1 = v2 - v1
            e2 = v3 - v1

            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])

            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue

            circumcenter = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(circumcenter)
            if norm_cc < 1e-10:
                continue
            circumcenter_norm = circumcenter / norm_cc

            # Check that the circumcenter is on the "far side" of the facet
            # (i.e., the cap centered at circumcenter_norm is empty)
            dot_val = np.clip(np.dot(circumcenter_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)

            if angular_radius > max_circumradius:
                max_circumradius = angular_radius

        return np.degrees(max_circumradius)

    def compute_covering_radius_sampled(points, num_test_points=200000):
        """
        Estimate covering radius by sampling many points on the sphere.
        """
        test_points = fibonacci_sphere(num_test_points)
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        min_dot = np.min(dot_products, axis=1)
        angles = np.arccos(min_dot)
        max_angle = np.max(angles)
        return np.degrees(max_angle)

    def lloyd_iteration(points, test_points, step_size=1.0):
        """One iteration of Lloyd's algorithm (vectorized)."""
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)

        n_points = len(points)
        sums = np.zeros((n_points, 3))
        np.add.at(sums, nearest, test_points)

        counts = np.bincount(nearest, minlength=n_points)

        # Avoid division by zero
        counts_safe = np.where(counts == 0, 1, counts)

        centroids = sums / counts_safe[:, np.newaxis]

        # Normalize centroids
        new_points = normalize(centroids)

        # Handle empty clusters (keep old point)
        empty_mask = counts == 0
        if np.any(empty_mask):
            new_points[empty_mask] = points[empty_mask]

        # Interpolate for stability
        result = (1 - step_size) * points + step_size * new_points
        return normalize(result)

    def get_worst_facets(points, top_k=10):
        """
        Return the top_k worst facets (largest circumradius) with their
        circumcenters and vertex indices.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return []

        facet_data = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]

            e1 = v2 - v1
            e2 = v3 - v1

            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])

            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue

            cc = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(cc)
            if norm_cc < 1e-10:
                continue
            cc_norm = cc / norm_cc

            dot_val = np.clip(np.dot(cc_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)

            facet_data.append((angular_radius, tuple(simplex), cc_norm))

        if not facet_data:
            return []

        facet_data.sort(key=lambda x: x[0], reverse=True)
        return facet_data[:top_k]

    def refine_worst_facets(points, num_steps=50, lr=0.02):
        """
        Targeted local search using geodesic bisectors: move the three vertices
        of the worst facet along their geodesic bisectors to minimize that
        facet's circumradius. Uses a smooth surrogate via soft-max weighting
        over the top facets and an adaptive learning-rate schedule.
        """
        pts = points.copy()
        for step in range(num_steps):
            if time.time() > deadline:
                break

            # Get top facets with a larger pool for soft-max weighting
            top_k_pool = min(12, len(pts) * 3)
            worst = get_worst_facets(pts, top_k=top_k_pool)
            if not worst:
                break

            # Use soft-max weighting over the top facets for smoothness
            radii = np.array([r for r, _, _ in worst])
            # Temperature for soft-max: use spread of radii
            temp = max(0.01, (radii.max() - radii.min()) * 0.5)
            exp_vals = np.exp((radii - radii.max()) / temp)
            weights = exp_vals / exp_vals.sum()

            gradient = np.zeros_like(pts)
            for w, (radius, simplex, cc_norm) in zip(weights, worst):
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]

                # Compute edges
                e1 = v2 - v1
                e2 = v3 - v1

                A_mat = np.array([
                    [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                    [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
                ])
                b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])

                try:
                    params = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    continue

                cc = v1 + params[0] * e1 + params[1] * e2
                norm_cc = np.linalg.norm(cc)
                if norm_cc < 1e-10:
                    continue
                cc_unit = cc / norm_cc

                # For each vertex, compute the geodesic bisector direction
                # that reduces the circumradius.
                for idx in simplex:
                    current = pts[idx]
                    # Direction from vertex to circumcenter (tangent to sphere)
                    to_cc = cc_unit - current
                    radial_component = np.dot(to_cc, current) * current
                    tangent_dir = to_cc - radial_component
                    norm_td = np.linalg.norm(tangent_dir)
                    if norm_td > 1e-10:
                        # Weight by soft-max weight and inverse distance for locality
                        gradient[idx] += w * tangent_dir / norm_td

            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                gradient = gradient / grad_norm
                # Adaptive learning rate with cosine decay
                progress = step / max(1, num_steps)
                effective_lr = lr * (0.5 + 0.5 * np.cos(np.pi * progress))
                pts = pts + effective_lr * gradient
                pts = normalize(pts)

        return pts

    def refine_high_precision(points, num_steps=80, lr=0.003):
        """
        High-precision polish pass: very small steps, focus on the single
        worst facet with tight tolerance to squeeze out the last bits.
        """
        pts = points.copy()
        for step in range(num_steps):
            if time.time() > deadline - 2.0:
                break

            worst = get_worst_facets(pts, top_k=3)
            if not worst:
                break

            # Only target the single worst facet for precision
            radius, simplex, cc_norm = worst[0]

            gradient = np.zeros_like(pts)
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]

            e1 = v2 - v1
            e2 = v3 - v1

            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])

            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                break

            cc = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(cc)
            if norm_cc < 1e-10:
                break
            cc_unit = cc / norm_cc

            for idx in simplex:
                current = pts[idx]
                to_cc = cc_unit - current
                radial_component = np.dot(to_cc, current) * current
                tangent_dir = to_cc - radial_component
                norm_td = np.linalg.norm(tangent_dir)
                if norm_td > 1e-10:
                    gradient[idx] += tangent_dir / norm_td

            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                gradient = gradient / grad_norm
                pts = pts + lr * gradient
                pts = normalize(pts)

        return pts

    def separate_close_points(points, min_angle=0.05):
        """Separate points that are too close together using vectorized operations."""
        pts = points.copy()
        cos_min = np.cos(min_angle)
        n = len(pts)
        for iteration in range(3):
            dots = pts @ pts.T
            np.fill_diagonal(dots, -1.0)
            for i in range(n):
                close = np.where(dots[i] > cos_min)[0]
                close = close[close > i]
                for j in close:
                    diff = pts[i] - pts[j]
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff > 1e-10:
                        diff_norm = diff / norm_diff
                        push = 0.01 * (1 - dots[i, j]) / (1 - cos_min)
                        pts[i] = pts[i] + push * diff_norm
                        pts[j] = pts[j] - push * diff_norm
                        pts[i] = pts[i] / np.linalg.norm(pts[i])
                        pts[j] = pts[j] / np.linalg.norm(pts[j])
        return pts

    # Pre-generate test points for Lloyd's
    num_test = 60000
    test_points = fibonacci_sphere(num_test)
    rng = np.random.RandomState(42)

    # Multiple initializations
    best_points = None
    best_score = np.inf

    initializations = []

    # 1. Fibonacci
    initializations.append(fibonacci_sphere(N))

    # 2. Perturbed Fibonacci (small)
    fib = fibonacci_sphere(N)
    initializations.append(normalize(fib + rng.randn(N, 3) * 0.03))

    # 3. Perturbed Fibonacci (medium)
    initializations.append(normalize(fib + rng.randn(N, 3) * 0.08))

    # 4. Random initializations
    for _ in range(5):
        random_pts = rng.randn(N, 3)
        random_pts = normalize(random_pts)
        initializations.append(random_pts)

    # 5. Grid-like (lat/lon)
    latitudes = np.linspace(-np.pi/2, np.pi/2, 10)
    longitudes = np.linspace(0, 2*np.pi, 5, endpoint=False)
    grid_pts = []
    for lat in latitudes:
        for lon in longitudes:
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:N])
    grid_pts = normalize(grid_pts)
    initializations.append(grid_pts)

    # 6. Another perturbed Fibonacci variant
    initializations.append(normalize(fib + rng.randn(N, 3) * 0.15))

    # Optimize each initialization with time budget
    time_per_init = (deadline - start_time - 20.0) / len(initializations)

    for init_idx, init_pts in enumerate(initializations):
        if time.time() >= deadline - 15.0:
            break

        points = init_pts.copy()
        init_deadline = start_time + (init_idx + 1) * time_per_init + 5.0
        init_deadline = min(init_deadline, deadline - 12.0)

        # Phase 1: Lloyd's algorithm (fast convergence)
        step_size = 0.7
        max_lloyd_iters = 250
        iters = 0
        while time.time() < min(init_deadline - 15.0, start_time + time_per_init * 0.5) and iters < max_lloyd_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.96
            if step_size < 0.02:
                step_size = 0.02
            iters += 1

        # Phase 2: Fine Lloyd refinement
        step_size = 0.1
        max_fine_iters = 150
        iters = 0
        while time.time() < min(init_deadline - 8.0, start_time + time_per_init * 0.8) and iters < max_fine_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.94
            if step_size < 0.005:
                break
            iters += 1

        # Phase 3: Separate close points
        points = separate_close_points(points, min_angle=0.04)

        # Phase 4: Worst-facet refinement (targeted local search with geodesic bisectors)
        if time.time() < init_deadline - 5.0:
            points = refine_worst_facets(points, num_steps=50, lr=0.015)
            points = separate_close_points(points, min_angle=0.03)

            # Second pass with smaller steps
            if time.time() < init_deadline - 2.0:
                points = refine_worst_facets(points, num_steps=35, lr=0.008)
                points = separate_close_points(points, min_angle=0.03)

        # Evaluate using fast sampled method first
        fast_score = compute_covering_radius_sampled(points, num_test_points=30000)
        if fast_score < best_score + 0.3:
            score = compute_covering_radius_precise(points)
            if score < best_score:
                best_score = score
                best_points = points.copy()

    # Final refinement with remaining time
    if best_points is not None and time.time() < deadline - 15.0:
        points = best_points.copy()

        # Fine Lloyd polish
        step_size = 0.05
        max_final_iters = 80
        iters = 0
        while time.time() < deadline - 20.0 and iters < max_final_iters:
            points = lloyd_iteration(points, test_points, step_size=step_size)
            step_size *= 0.95
            iters += 1

            if iters % 8 == 0:
                current_score = compute_covering_radius_precise(points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = points.copy()

        # Worst-facet refinement (multiple passes with decreasing lr)
        if time.time() < deadline - 10.0:
            points = best_points.copy()
            points = refine_worst_facets(points, num_steps=50, lr=0.012)
            points = separate_close_points(points, min_angle=0.02)
            current_score = compute_covering_radius_precise(points)
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()

        # Pass 2: smaller steps for fine tuning
        if time.time() < deadline - 7.0:
            points = best_points.copy()
            points = refine_worst_facets(points, num_steps=40, lr=0.005)
            points = separate_close_points(points, min_angle=0.02)
            current_score = compute_covering_radius_precise(points)
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()

        # Pass 3: high-precision polish (single worst facet, tiny steps)
        if time.time() < deadline - 4.0:
            points = best_points.copy()
            points = refine_high_precision(points, num_steps=60, lr=0.002)
            current_score = compute_covering_radius_precise(points)
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()

        # Pass 4: another very fine polish
        if time.time() < deadline - 2.0:
            points = best_points.copy()
            points = refine_high_precision(points, num_steps=40, lr=0.001)
            current_score = compute_covering_radius_precise(points)
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()

    # Final normalization and safety checks
    if best_points is None:
        best_points = fibonacci_sphere(N)

    best_points = normalize(best_points)

    # Ensure no coincident points (final safety check)
    best_points = separate_close_points(best_points, min_angle=0.02)
    best_points = normalize(best_points)

    return best_points
# EVOLVE-BLOCK-END
