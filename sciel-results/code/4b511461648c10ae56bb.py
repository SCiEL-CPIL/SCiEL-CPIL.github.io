import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    def normalize(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return points / norms

    def compute_facet_data(points):
        """
        Compute convex hull and facet circumcenters/radii.
        Returns: hull, centers (array), radii (array)
        """
        points = normalize(points)
        try:
            hull = ConvexHull(points)
        except Exception:
            return None, None, None

        n_facets = len(hull.simplices)
        centers = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        for i, simplex in enumerate(hull.simplices):
            a = points[simplex[0]]
            b = points[simplex[1]]
            c = points[simplex[2]]

            ab = np.dot(a, b)
            ac = np.dot(a, c)
            bc = np.dot(b, c)

            Gram = np.array([
                [1.0, ab, ac],
                [ab, 1.0, bc],
                [ac, bc, 1.0]
            ])

            try:
                m = np.linalg.solve(Gram, np.ones(3))
                c_dir = m[0] * a + m[1] * b + m[2] * c
                norm_c_dir = np.linalg.norm(c_dir)
                if norm_c_dir < 1e-10:
                    continue
                centers[i] = c_dir / norm_c_dir
                radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
            except np.linalg.LinAlgError:
                continue

        return hull, centers, radii

    def compute_covering_radius(points):
        """
        Compute the actual covering radius (max facet circumradius).
        """
        hull, centers, radii = compute_facet_data(points)
        if hull is None or len(radii) == 0:
            return np.pi
        return np.max(radii)

    def soft_max_radius_and_grad(points, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points.
        Uses a more numerically stable approach.
        """
        hull, centers, radii = compute_facet_data(points)
        if hull is None or len(radii) == 0:
            return np.pi, np.zeros_like(points)

        max_r = np.max(radii)
        if max_r < 1e-15:
            return max_r, np.zeros_like(points)

        # Use log-sum-exp trick for numerical stability
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)

        grads = np.zeros_like(points)
        n_facets = len(hull.simplices)

        for i in range(n_facets):
            w = weights[i]
            if w < 1e-15:
                continue
            r = radii[i]
            if r < 1e-12:
                continue

            C = centers[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue

            # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r) projected to tangent space
            simplex = hull.simplices[i]
            for k in range(3):
                p_idx = simplex[k]
                p = points[p_idx]
                # Project gradient to tangent space at p
                grad_contrib = -C / sin_r
                radial = np.dot(grad_contrib, p) * p
                tangential = grad_contrib - radial
                grads[p_idx] += w * tangential

        return soft_radius, grads

    def pressure_minimization(pts, time_budget, alpha_start=0.02):
        """
        Pressure-minimization dynamic: move vertices along the negative gradient
        of the convex hull's volume, effectively treating the problem as finding
        the equilibrium shape of a soap film. This is a structural change from
        the current program's approach.
        """
        pts = pts.copy()
        alpha = alpha_start
        start_t = time.time()

        def hull_volume(p):
            """Compute the volume of the convex hull."""
            try:
                hull = ConvexHull(p)
                return hull.volume
            except Exception:
                return 0.0

        def volume_gradient(p):
            """
            Compute the gradient of the convex hull volume w.r.t. the points.
            For a polytope, the gradient of the volume w.r.t. a vertex is
            (1/3) * the area of the facet opposite to that vertex, in the
            direction of the facet's outward normal.
            """
            try:
                hull = ConvexHull(p)
            except Exception:
                return np.zeros_like(p)

            n_facets = len(hull.simplices)
            grads = np.zeros_like(p)

            for i, simplex in enumerate(hull.simplices):
                a = p[simplex[0]]
                b = p[simplex[1]]
                c = p[simplex[2]]

                # Facet normal (outward)
                v1 = a - b
                v2 = b - c
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n

                # Ensure normal points outward (away from centroid)
                centroid = np.mean(p, axis=0)
                facet_centroid = np.mean([a, b, c], axis=0)
                if np.dot(n, facet_centroid - centroid) < 0:
                    n = -n

                # Facet area
                area = 0.5 * norm_n

                # For each vertex of the facet, the contribution to the volume
                # gradient is (1/3) * area * n
                for k in range(3):
                    idx = simplex[k]
                    grads[idx] += (1.0 / 3.0) * area * n

            return grads

        current_vol = hull_volume(pts)

        while time.time() - start_t < time_budget:
            grads = volume_gradient(pts)

            # Project onto tangent plane
            dot_gp = np.sum(grads * pts, axis=1)
            grads_tan = grads - dot_gp[:, np.newaxis] * pts

            # Move in the direction that decreases volume (negative gradient)
            step = alpha
            improved = False
            for _ in range(8):
                new_pts = pts - step * grads_tan
                new_pts = normalize(new_pts)
                new_vol = hull_volume(new_pts)
                if new_vol < current_vol:
                    pts = new_pts
                    current_vol = new_vol
                    improved = True
                    break
                step *= 0.5

            if not improved:
                alpha *= 0.85
            else:
                alpha = min(alpha * 1.02, 0.05)

            alpha = max(alpha, 1e-6)

        return pts

    def repulsion_relax(pts, time_budget, alpha_start=0.05):
        """
        Repulsion relaxation: minimize sum 1/||p_i - p_j||^2 via gradient descent.
        Helps spread points more uniformly before covering optimization.
        """
        pts = pts.copy()
        alpha = alpha_start
        start_t = time.time()

        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)

        current_energy = repulsion_energy(pts)

        while time.time() - start_t < time_budget:
            diff = pts[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)

            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)

            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

            # Project onto tangent plane
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts

            # Adaptive step with backtracking
            step = alpha
            improved = False
            for _ in range(8):
                new_pts = pts + step * forces_tan
                new_pts = normalize(new_pts)
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5

            if not improved:
                alpha *= 0.85
            else:
                alpha = min(alpha * 1.02, 0.08)

            alpha = max(alpha, 1e-6)

        return pts

    def facet_local_search(pts, n_iters=80, time_budget=10.0):
        """
        Facet-based local search: directly minimize the maximum circumradius
        by moving vertices of the largest facets toward their circumcenters.
        """
        pts = pts.copy()
        start_t = time.time()

        for it in range(n_iters):
            if time.time() - start_t > time_budget:
                break

            hull, centers, radii = compute_facet_data(pts)
            if hull is None:
                break

            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break

            # Find facets that are close to the maximum radius (top 20%)
            threshold = max_rad * 0.8
            large_facets = np.where(radii > threshold)[0]

            if len(large_facets) == 0:
                break

            # Compute weighted updates: move vertices toward circumcenters
            updates = np.zeros_like(pts)
            counts = np.zeros(N)

            for f in large_facets:
                facet_pts = hull.simplices[f]
                cc = centers[f]
                rad = radii[f]
                if rad < 1e-12:
                    continue
                weight = ((rad - threshold) / (max_rad - threshold + 1e-15)) ** 2
                weight = max(weight, 0.1)
                for idx in facet_pts:
                    updates[idx] += weight * (cc - pts[idx])
                    counts[idx] += weight

            # Apply updates where counts > 0
            mask = counts > 1e-10
            if np.any(mask):
                mean_rad = np.mean(radii)
                gap = max_rad - mean_rad
                step_size = 0.08 * (gap / (mean_rad + 1e-15))
                step_size = min(step_size, 0.15)
                step_size = max(step_size, 0.01)

                best_pts_trial = pts.copy()
                best_rad_trial = max_rad
                for trial_step in [step_size, step_size * 0.5, step_size * 0.25, step_size * 0.125]:
                    trial_pts = pts.copy()
                    trial_pts[mask] += trial_step * updates[mask] / counts[mask, np.newaxis]
                    trial_pts = normalize(trial_pts)
                    trial_rad = compute_covering_radius(trial_pts)
                    if trial_rad < best_rad_trial:
                        best_rad_trial = trial_rad
                        best_pts_trial = trial_pts.copy()

                if best_rad_trial < max_rad:
                    pts = best_pts_trial
                else:
                    break

            pts = normalize(pts)

        return pts

    def icosahedral_init():
        """
        Generate an icosahedral geodesic dome with 50 vertices.
        """
        t = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)

        ico_faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])

        points = [ico[i] for i in range(12)]

        edge_mids = set()
        for face in ico_faces:
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                if (a, b) not in edge_mids:
                    edge_mids.add((a, b))
                    mid = (ico[a] + ico[b]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    points.append(mid)

        face_centers = []
        for face in ico_faces:
            center = np.mean(ico[face], axis=0)
            center = center / np.linalg.norm(center)
            face_centers.append(center)

        all_candidates = np.array(points + face_centers)

        selected = [all_candidates[0]]
        for _ in range(N - 1):
            min_dists = []
            for cand in all_candidates:
                min_dist = min(np.linalg.norm(cand - s) for s in selected)
                min_dists.append(min_dist)
            min_dists = np.array(min_dists)
            for s in selected:
                idx = np.where(np.all(all_candidates == s, axis=1))[0]
                if len(idx) > 0:
                    min_dists[idx] = -1
            best_idx = np.argmax(min_dists)
            selected.append(all_candidates[best_idx])

        return np.array(selected)

    def greedy_init(seed_val):
        np.random.seed(seed_val)
        pts = [np.random.randn(3)]
        norms = np.linalg.norm(pts[0])
        pts[0] = pts[0] / norms
        for _ in range(N - 1):
            candidates = np.random.randn(2000, 3)
            norms = np.linalg.norm(candidates, axis=1, keepdims=True)
            candidates = candidates / norms
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)

    def lbfgs_stage(points, time_budget):
        """
        Use L-BFGS-B to minimize the soft-max covering radius with gradients.
        """
        start_time = time.time()
        best_pts = points.copy()
        best_radius = compute_covering_radius(points)

        for beta in [80.0, 150.0, 300.0]:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 3.0:
                break

            def objective_and_grad(x, _beta=beta):
                if time.time() - start_time > time_budget:
                    raise Exception("timeout")
                pts = normalize(x.reshape(N, 3))
                soft_r, grads = soft_max_radius_and_grad(pts, beta=_beta)
                grad_flat = grads.reshape(-1)
                return soft_r, grad_flat

            x0 = best_pts.reshape(-1)

            try:
                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200}
                )

                if result.fun < objective_and_grad(x0)[0]:
                    new_pts = normalize(result.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
            except Exception:
                pass

        return best_pts, best_radius

    # Generate multiple initial configurations
    initial_configs = []

    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initial_configs.append(fib_pts)

    # 2. Fibonacci with different phase offsets
    for phase in [0.37, 0.618, 0.25]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + phase)
        r2 = np.sqrt(1.0 - z2 * z2)
        fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initial_configs.append(fib2_pts)

    # 3. Icosahedral geodesic dome initialization
    try:
        ico_pts = icosahedral_init()
        initial_configs.append(ico_pts)
    except Exception:
        pass

    # 4. Random points (seeded for reproducibility)
    for seed_offset in range(3):
        np.random.seed(42 + seed_offset * 100)
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initial_configs.append(pts)

    # 5. Greedy initialization
    initial_configs.append(greedy_init(101))
    initial_configs.append(greedy_init(202))
    initial_configs.append(greedy_init(303))

    best_pts = initial_configs[0].copy()
    best_radius = compute_covering_radius(best_pts)

    # Phase 1: Repulsion relaxation + Pressure minimization to get good spread
    for idx, init_pts in enumerate(initial_configs[:6]):
        if time.time() > deadline - 80.0:
            break

        # Repulsion relaxation
        relaxed = repulsion_relax(init_pts, time_budget=3.0, alpha_start=0.03)
        rad_relaxed = compute_covering_radius(relaxed)
        if rad_relaxed < best_radius:
            best_radius = rad_relaxed
            best_pts = relaxed.copy()

        # Pressure minimization (new structural component)
        if time.time() < deadline - 75.0:
            pressure_pts = pressure_minimization(relaxed, time_budget=2.0, alpha_start=0.02)
            rad_pressure = compute_covering_radius(pressure_pts)
            if rad_pressure < best_radius:
                best_radius = rad_pressure
                best_pts = pressure_pts.copy()

    # Phase 2: L-BFGS-B optimization with soft-max and gradients
    if time.time() < deadline - 40.0:
        pts_res, rad_res = lbfgs_stage(best_pts, time_budget=30.0)
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()

    # Phase 3: Facet-based local search
    if time.time() < deadline - 25.0:
        facet_pts = facet_local_search(best_pts, n_iters=80, time_budget=15.0)
        rad_facet = compute_covering_radius(facet_pts)
        if rad_facet < best_radius:
            best_radius = rad_facet
            best_pts = facet_pts.copy()

        # Another round of L-BFGS-B after facet search
        if time.time() < deadline - 15.0:
            pts_res2, rad_res2 = lbfgs_stage(best_pts, time_budget=8.0)
            if rad_res2 < best_radius:
                best_radius = rad_res2
                best_pts = pts_res2.copy()

        # Another facet search round
        if time.time() < deadline - 8.0:
            facet_pts2 = facet_local_search(best_pts, n_iters=50, time_budget=5.0)
            rad_facet2 = compute_covering_radius(facet_pts2)
            if rad_facet2 < best_radius:
                best_radius = rad_facet2
                best_pts = facet_pts2.copy()

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        def obj_final(x, _beta=2000.0):
            pts = normalize(x.reshape(N, 3))
            soft_r, grads = soft_max_radius_and_grad(pts, beta=_beta)
            return soft_r, grads.reshape(-1)

        try:
            res = minimize(
                obj_final,
                best_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 500}
            )
            if res.fun < obj_final(best_pts.flatten())[0]:
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = compute_covering_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass

        # Final facet-based polish
        if time.time() < deadline - 3.0:
            facet_pts3 = facet_local_search(best_pts, n_iters=30, time_budget=2.0)
            rad_facet3 = compute_covering_radius(facet_pts3)
            if rad_facet3 < best_radius:
                best_radius = rad_facet3
                best_pts = facet_pts3.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_pts
