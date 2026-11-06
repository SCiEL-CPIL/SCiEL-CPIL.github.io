"""Covering the sphere with 50 equal caps: smooth log-sum-exp energy optimization + facet refinement."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. Multiple initializations (Fibonacci, random, perturbed).
    2. Lloyd's relaxation for coarse tiling.
    3. Smooth log-sum-exp covering energy gradient descent.
    4. Geodesic gradient refinement on convex-hull facet circumradii.
    5. High-precision micro-polish on worst facets.
    6. Multiple restarts with different seeds.
    7. Explicit deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 145.0

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

    def lloyd_iteration(points, test_points, step_size=1.0):
        dots = np.clip(test_points @ points.T, -1.0, 1.0)
        nearest = np.argmax(dots, axis=1)
        n_pts = len(points)
        sums = np.zeros((n_pts, 3))
        np.add.at(sums, nearest, test_points)
        counts = np.bincount(nearest, minlength=n_pts)
        counts_safe = np.where(counts == 0, 1, counts)
        centroids = sums / counts_safe[:, np.newaxis]
        centroids = normalize(centroids)
        empty = counts == 0
        if np.any(empty):
            centroids[empty] = points[empty]
        result = (1 - step_size) * points + step_size * centroids
        return normalize(result)

    def compute_covering_radius_precise(points):
        """Exact covering radius via convex hull facet circumradii (in degrees)."""
        try:
            hull = ConvexHull(points)
        except Exception:
            test = fibonacci_sphere(200000)
            dots = np.clip(test @ points.T, -1.0, 1.0)
            min_dot = np.min(dots, axis=1)
            return np.degrees(np.arccos(np.max(min_dot)))

        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2 * np.dot(e1, e1), 2 * np.dot(e1, e2)],
                          [2 * np.dot(e1, e2), 2 * np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1), np.dot(e2, e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0] * e1 + params[1] * e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            if ang > max_rad:
                max_rad = ang
        return np.degrees(max_rad)

    def get_facet_data(points):
        """Return arrays: (facet_radii [F], facet_ccs [F,3], facet_verts [F,3] int)."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return None, None, None

        radii = []
        ccs = []
        verts = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2 * np.dot(e1, e1), 2 * np.dot(e1, e2)],
                          [2 * np.dot(e1, e2), 2 * np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1), np.dot(e2, e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0] * e1 + params[1] * e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            radii.append(ang)
            ccs.append(cc)
            verts.append(simplex)

        if not radii:
            return None, None, None

        return np.array(radii), np.array(ccs), np.array(verts, dtype=int)

    def smooth_covering_energy(points, grid, beta=50.0):
        """
        Compute smooth log-sum-exp approximation of the max covering radius
        and its gradient with respect to each point.
        
        Energy = (1/beta) * log(sum_j exp(beta * d_j)) where d_j is the angular
        distance from grid point j to its nearest site.
        
        The gradient pushes each site away from the grid points where it is the
        nearest neighbor, weighted by the soft-max over all sites.
        """
        dots = np.clip(grid @ points.T, -1.0, 1.0)  # [G, N]
        # Distance from each grid point to each site
        dists = -dots  # larger distance = smaller dot
        
        # Soft-max over sites for each grid point (which site is "responsible")
        # We want to compute: for each grid point j, the max over sites of dist(j, site)
        # But actually the covering radius is max_j min_i dist(j, i)
        # So we need: for each grid point, the min distance to any site
        # Then the covering radius is max over grid points of that min distance
        
        min_dots = np.max(dots, axis=1)  # [G] - max dot = min distance
        min_dists = -min_dots  # [G]
        
        # Soft-max of min_dists
        m = np.max(min_dists)
        exp_vals = np.exp(beta * (min_dists - m))
        log_sum_exp = m + np.log(np.sum(exp_vals)) / beta
        
        # Gradient: d(energy)/d(point_i)
        # energy = (1/beta) * log(sum_j exp(beta * min_i dist(j,i)))
        # d(energy)/d(point_i) = sum_j [exp(beta * min_dist_j) / sum_k exp(beta * min_dist_k)] * d(min_dist_j)/d(point_i)
        
        weights = exp_vals / np.sum(exp_vals)  # [G]
        
        # For each grid point j, find which site is nearest (the one achieving min_dist_j)
        nearest = np.argmax(dots, axis=1)  # [G]
        
        # d(min_dist_j)/d(point_i) = -dot(grid_j, point_i) / |grid_j - ... | 
        # Actually: min_dist_j = -max_i dot(grid_j, point_i)
        # If site i is the nearest: d(min_dist_j)/d(point_i) = -grid_j
        # (since d(-dot(g, p))/dp = -g)
        
        grad = np.zeros_like(points)
        for j in range(len(grid)):
            w = weights[j]
            if w < 1e-10:
                continue
            i = nearest[j]
            # Gradient contribution: -w * grid_j projected onto tangent plane at point_i
            to_grad = -grid[j]
            radial_comp = np.dot(to_grad, points[i]) * points[i]
            tangent = to_grad - radial_comp
            grad[i] += w * tangent
        
        return log_sum_exp, grad

    def smooth_energy_optimization(points, grid, num_rounds=100, base_lr=0.02, beta=50.0):
        """Optimize using smooth log-sum-exp covering energy."""
        pts = points.copy()
        
        for rnd in range(num_rounds):
            if time.time() > deadline - 5:
                break
            
            # Adapt beta: start high (sharp), decrease for smoother gradients
            current_beta = beta * max(0.3, 1.0 - 0.7 * rnd / max(1, num_rounds))
            
            energy, grad = smooth_covering_energy(pts, grid, current_beta)
            
            gn = np.linalg.norm(grad)
            if gn > 1e-14:
                lr = base_lr * (1.0 - 0.5 * rnd / max(1, num_rounds))
                pts = pts + lr * (grad / gn)
                pts = normalize(pts)
        
        return pts

    def refine_geodesic(points, num_rounds=100, base_lr=0.01):
        """Geodesic gradient refinement using exponential weighting toward worst facets."""
        pts = points.copy()
        for rnd in range(num_rounds):
            if time.time() > deadline - 2:
                break

            radii, ccs, verts = get_facet_data(pts)
            if radii is None:
                break

            max_r = np.max(radii)
            # Use log-sum-exp style smooth weighting
            beta = 30.0
            shifted = (radii - max_r) / 0.02
            exp_vals = np.exp(beta * shifted / 30.0)
            total_w = np.sum(exp_vals)
            if total_w < 1e-14:
                break
            weights = exp_vals / total_w

            grad = np.zeros_like(pts)
            for f in range(len(radii)):
                w = weights[f]
                if w < 1e-10:
                    continue
                cc = ccs[f]
                for idx in verts[f]:
                    cur = pts[idx]
                    to_cc = cc - cur
                    radial_comp = np.dot(to_cc, cur) * cur
                    tangent = to_cc - radial_comp
                    nt = np.linalg.norm(tangent)
                    if nt > 1e-12:
                        grad[idx] += w * (tangent / nt)

            gn = np.linalg.norm(grad)
            if gn > 1e-14:
                lr = base_lr * (1.0 - 0.5 * rnd / max(1, num_rounds))
                pts = pts + lr * (grad / gn)
                pts = normalize(pts)

        return pts

    def micro_polish(points, num_iters=300, top_k=8):
        """Iterative micro-polish: repeatedly solve each worst facet's circumradius with very small steps."""
        pts = points.copy()
        prev_best = compute_covering_radius_precise(pts)

        for it in range(num_iters):
            if time.time() > deadline - 2:
                break

            radii, ccs, verts = get_facet_data(pts)
            if radii is None:
                break

            order = np.argsort(radii)[::-1]
            top_order = order[:min(top_k, len(order))]

            improved = False
            for f in top_order:
                cc = ccs[f]
                facet_verts = verts[f]

                step = 0.005
                for _ in range(4):
                    new_pts = pts.copy()
                    for idx in facet_verts:
                        cur = new_pts[idx]
                        to_cc = cc - cur
                        radial_comp = np.dot(to_cc, cur) * cur
                        tangent = to_cc - radial_comp
                        nt = np.linalg.norm(tangent)
                        if nt > 1e-12:
                            new_pts[idx] = cur + step * (tangent / nt)
                            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])

                    new_score = compute_covering_radius_precise(new_pts)
                    if new_score < prev_best - 1e-9:
                        pts = new_pts
                        prev_best = new_score
                        improved = True
                        step *= 1.3
                        break
                    else:
                        step *= 0.5
                        if step < 1e-7:
                            break

            if not improved:
                break

        return pts

    def separate_close_points(points, min_cos=0.9995):
        """Push apart points that are nearly coincident."""
        pts = points.copy()
        for _ in range(3):
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.dot(pts[i], pts[j])
                    if d > min_cos:
                        diff = pts[i] - pts[j]
                        nd = np.linalg.norm(diff)
                        if nd > 1e-10:
                            diff = diff / nd
                            push = 0.002 * (1.0 - d) / (1.0 - min_cos + 1e-9)
                            pts[i] = pts[i] + push * diff
                            pts[j] = pts[j] - push * diff
                            pts[i] = pts[i] / np.linalg.norm(pts[i])
                            pts[j] = pts[j] / np.linalg.norm(pts[j])
        return pts

    # Pre-compute grids
    lloyd_test = fibonacci_sphere(80000)
    smooth_grid = fibonacci_sphere(30000)

    best_points = None
    best_score = float('inf')

    rng = np.random.RandomState(42)

    max_restarts = 15
    restart = 0

    while time.time() < deadline - 12 and restart < max_restarts:
        restart += 1

        # Initialization
        if restart == 1:
            points = fibonacci_sphere(N)
        elif restart == 2:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.02
            points = normalize(points)
        elif restart <= 5:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * (0.03 + 0.02 * (restart - 3))
            points = normalize(points)
        elif best_points is not None and rng.random() < 0.7:
            points = best_points + rng.standard_normal((N, 3)) * 0.02
            points = normalize(points)
        else:
            points = rng.standard_normal((N, 3))
            points = normalize(points)

        # Phase 1: Lloyd's relaxation
        step_size = 0.5
        for it in range(50):
            if time.time() > deadline - 35:
                break
            points = lloyd_iteration(points, lloyd_test, step_size)
            step_size *= 0.95
            if step_size < 0.02:
                step_size = 0.02

        # Phase 2: Separate close points
        points = separate_close_points(points)

        # Phase 3: Smooth log-sum-exp energy optimization
        if time.time() < deadline - 30:
            points = smooth_energy_optimization(points, smooth_grid, num_rounds=80, base_lr=0.015, beta=40.0)
            points = separate_close_points(points, min_cos=0.9998)

        # Phase 4: Geodesic refinement on facets
        if time.time() < deadline - 20:
            points = refine_geodesic(points, num_rounds=60, base_lr=0.01)
            points = separate_close_points(points, min_cos=0.9998)

        # Evaluate
        score = compute_covering_radius_precise(points)

        if score < best_score:
            best_score = score
            best_points = points.copy()

        # Phase 5: Micro-polish on worst facets
        if time.time() < deadline - 15:
            points = micro_polish(points, num_iters=200, top_k=8)
            points = separate_close_points(points, min_cos=0.9999)
            score2 = compute_covering_radius_precise(points)
            if score2 < best_score:
                best_score = score2
                best_points = points.copy()

        # Phase 6: Additional refinement if close to known best
        if best_score < 18.4 and time.time() < deadline - 10:
            points = refine_geodesic(points, num_rounds=30, base_lr=0.003)
            points = micro_polish(points, num_iters=150, top_k=6)
            points = separate_close_points(points, min_cos=0.9999)
            score3 = compute_covering_radius_precise(points)
            if score3 < best_score:
                best_score = score3
                best_points = points.copy()

    # Final normalization
    if best_points is not None:
        best_points = normalize(best_points)
    else:
        best_points = fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
