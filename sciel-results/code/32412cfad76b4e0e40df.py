import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def fibonacci_initial(phase=0.0):
        i = np.arange(N) + 0.5 + phase
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]

        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)

        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def compute_covering_radius(p):
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return float('inf')
        return np.max(radii)

    def compute_hull_volume(p):
        """Compute the volume of the convex hull using the divergence theorem."""
        try:
            hull = ConvexHull(p)
            return hull.volume
        except Exception:
            return 0.0

    def compute_volume_gradient(p):
        """Compute gradient of hull volume w.r.t. point positions.
        For a convex polyhedron, dV/dx_i = (1/2) * sum over facets containing i of 
        (area_f * n_f) where n_f is outward normal and area_f is facet area.
        This pushes points outward to increase volume (spread them out)."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None
        
        try:
            hull = ConvexHull(p)
        except Exception:
            return None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None
        
        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]
        
        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)
        
        valid = norm_n > 1e-12
        if not np.any(valid):
            return None
        
        # Facet areas (half of cross product magnitude)
        areas = 0.5 * norm_n
        
        # Outward normals (pointing away from origin for unit sphere)
        n_unit = n / norm_n[:, np.newaxis]
        # Ensure normals point outward (away from centroid ~ origin)
        centroids = (a + b + c) / 3.0
        dots = np.sum(n_unit * centroids, axis=1)
        n_unit[dots < 0] *= -1
        
        # Gradient contribution: each facet contributes area * normal to its 3 vertices
        grad = np.zeros_like(p)
        for v in range(3):
            idxs = facets[:, v]
            contrib = areas[:, np.newaxis] * n_unit
            np.add.at(grad, idxs, contrib)
        
        return grad

    def compute_forces(p, beta=50.0):
        """Compute smooth forces from soft-max energy + repulsion (vectorized)."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None

        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        weights = (2.0 / beta) * radii * exp_terms / sum_exp

        # Vectorized force accumulation: for each facet, add weight * (cc - p[idx])
        # to the three vertices of that facet
        forces = np.zeros_like(p)
        # For each of the 3 vertices in each facet
        for v in range(3):
            idxs = facets[:, v]
            w = weights
            cc = circumcenters
            # Accumulate: forces[idx] += w[k] * (cc[k] - p[idx])
            # Use np.add.at for repeated indices
            contrib = w[:, np.newaxis] * (cc - p[idxs])
            np.add.at(forces, idxs, contrib)

        # Repulsion: push points apart
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += 0.03 * forces_rep

        return forces, facets, circumcenters, radii

    def lloyd_iteration(pts, n_iter=50, aggressiveness=1.0):
        """Aggressive Lloyd iteration: move vertices toward circumcenters of worst facets."""
        pts = pts.copy()
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            if len(facets) == 0:
                break
            
            # Move vertices of the worst facets toward their circumcenters
            max_r = np.max(radii)
            if max_r < 1e-10:
                break
            # Consider facets with radius > 90% of max (more aggressive)
            active = np.where(radii >= 0.90 * max_r)[0]
            
            forces = np.zeros_like(pts)
            for k in active:
                cc = circumcenters[k]
                # Weight strongly by how much this facet exceeds the threshold
                w = ((radii[k] - 0.90 * max_r) / max_r) ** 1.5 + 0.2
                for idx in facets[k]:
                    radial = np.dot(cc - pts[idx], pts[idx])
                    tangential = (cc - pts[idx]) - radial * pts[idx]
                    forces[idx] += w * tangential
            
            # Scale forces to reasonable magnitude
            f_max = np.max(np.linalg.norm(forces, axis=1))
            if f_max > 1e-12:
                # Adaptive step: larger when far from equilibrium
                progress = max_r / (np.pi / 2)  # fraction of max possible
                step = aggressiveness * 0.08 * (0.5 + 0.5 * progress)
                pts += step * (forces / f_max)
            
            pts = normalize(pts)
        
        return pts

    def volume_relaxation(pts, n_iter=100, strength=0.02):
        """Use hull volume gradient to spread points outward (soap-film equilibrium)."""
        pts = pts.copy()
        for _ in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            grad = compute_volume_gradient(pts)
            if grad is None:
                break
            # Project gradient to tangent plane
            radial = np.sum(grad * pts, axis=1, keepdims=True)
            tangential = grad - radial * pts
            pts += strength * tangential
            pts = normalize(pts)
        return pts

    def optimize_from_initial(pts_init, time_budget):
        """Run multi-phase optimization from a given initial configuration."""
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget

        # Phase 0: Volume relaxation to spread points
        if time.time() < end_time - 25.0:
            pts_vol = volume_relaxation(best_pts, n_iter=80, strength=0.03)
            rad_vol = compute_covering_radius(pts_vol)
            if rad_vol < best_radius:
                best_radius = rad_vol
                best_pts = pts_vol.copy()

        # Phase 1: Coarse soft-max gradient descent
        pts = best_pts.copy()
        eta = 0.02
        beta = 25.0
        n_steps = 800

        for step in range(n_steps):
            if time.time() > end_time - 20.0:
                break

            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break

            pts += eta * forces
            pts = normalize(pts)

            if radii is not None:
                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

            eta *= 0.997
            beta += 0.3

        # Phase 1.5: Aggressive Lloyd iteration on worst facets
        if time.time() < end_time - 18.0:
            pts_lloyd = lloyd_iteration(best_pts, n_iter=40, aggressiveness=1.8)
            rad_lloyd = compute_covering_radius(pts_lloyd)
            if rad_lloyd < best_radius:
                best_radius = rad_lloyd
                best_pts = pts_lloyd.copy()

        # Phase 2: L-BFGS-B on soft-max surrogate for covering radius
        if time.time() < end_time - 15.0:
            pts = best_pts.copy()

            def objective_softmax(params, beta=100.0):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                r2 = radii * radii
                max_r2 = np.max(r2)
                exp_terms = np.exp(beta * (r2 - max_r2))
                soft_max = (1.0 / beta) * np.log(np.sum(exp_terms)) + max_r2
                return soft_max

            def objective_sum_sq(params):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                return np.sum(radii**2) + 0.01 * np.var(radii)

            for obj_func in [objective_softmax, objective_sum_sq]:
                if time.time() > end_time - 10.0:
                    break
                try:
                    res = minimize(obj_func, pts.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 300})
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                except Exception:
                    pass

        # Phase 2.5: More aggressive Lloyd after L-BFGS
        if time.time() < end_time - 12.0:
            pts_lloyd2 = lloyd_iteration(best_pts, n_iter=25, aggressiveness=1.2)
            rad_lloyd2 = compute_covering_radius(pts_lloyd2)
            if rad_lloyd2 < best_radius:
                best_radius = rad_lloyd2
                best_pts = pts_lloyd2.copy()

        # Phase 3: Direct minimax projected gradient with adaptive threshold
        if time.time() < end_time - 8.0:
            pts = best_pts.copy()
            eta = 0.004
            threshold_frac = 0.90

            for step in range(800):
                if time.time() > end_time - 5.0:
                    break

                facets, circumcenters, radii = compute_hull_data(pts)
                if facets is None:
                    break

                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

                threshold = threshold_frac * current_radius
                active_facets = np.where(radii >= threshold)[0]

                if len(active_facets) == 0:
                    break

                forces = np.zeros_like(pts)

                facet_weights = np.zeros(len(active_facets))
                for j, k in enumerate(active_facets):
                    facet_weights[j] = (radii[k] - threshold) / max(current_radius - threshold, 1e-12)
                facet_weights = facet_weights / (np.sum(facet_weights) + 1e-12)

                for j, k in enumerate(active_facets):
                    w = facet_weights[j]
                    cc = circumcenters[k]
                    for idx in facets[k]:
                        radial = np.dot(cc - pts[idx], pts[idx])
                        tangential = (cc - pts[idx]) - radial * pts[idx]
                        forces[idx] += w * tangential

                # Scale forces to unit magnitude per point for stable stepping
                f_norms = np.linalg.norm(forces, axis=1, keepdims=True)
                f_norms[f_norms < 1e-12] = 1.0
                forces = forces / f_norms

                pts += eta * forces
                pts = normalize(pts)

                if step % 100 == 0:
                    threshold_frac = min(0.97, threshold_frac + 0.01)
                eta *= 0.996

        # Phase 4: Fine soft-max polish with higher beta
        if time.time() < end_time - 2.0:
            pts = best_pts.copy()
            eta = 0.0015
            beta = 200.0

            for step in range(300):
                if time.time() > end_time - 1.0:
                    break

                forces, facets, circumcenters, radii = compute_forces(pts, beta)
                if forces is None:
                    break

                pts += eta * forces
                pts = normalize(pts)

                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()

                eta *= 0.994
                beta += 0.5

        # Phase 5: Final aggressive Lloyd polish
        if time.time() < end_time - 1.0:
            pts_lloyd3 = lloyd_iteration(best_pts, n_iter=15, aggressiveness=1.0)
            rad_lloyd3 = compute_covering_radius(pts_lloyd3)
            if rad_lloyd3 < best_radius:
                best_radius = rad_lloyd3
                best_pts = pts_lloyd3.copy()

        # Phase 6: Final volume + Lloyd combined polish
        if time.time() < end_time - 0.5:
            pts_final = volume_relaxation(best_pts, n_iter=30, strength=0.01)
            pts_final = lloyd_iteration(pts_final, n_iter=10, aggressiveness=0.8)
            rad_final = compute_covering_radius(pts_final)
            if rad_final < best_radius:
                best_radius = rad_final
                best_pts = pts_final.copy()

        return best_pts, best_radius

    # Multi-restart strategy
    # Restart 1: Fibonacci
    fib_pts = fibonacci_initial(0.0)
    best_pts, best_radius = optimize_from_initial(fib_pts, 70.0)

    # Restart 2: Fibonacci with phase shift
    if time.time() < deadline - 30.0:
        fib2 = fibonacci_initial(0.5)
        pts2, rad2 = optimize_from_initial(fib2, 18.0)
        if rad2 < best_radius:
            best_radius = rad2
            best_pts = pts2

    # Restart 3: Random perturbation of best (smaller perturbation for refinement)
    if time.time() < deadline - 18.0:
        rng = np.random.default_rng(42)
        perturbed = best_pts + rng.normal(0, 0.012, (N, 3))
        perturbed = normalize(perturbed)
        pts3, rad3 = optimize_from_initial(perturbed, 12.0)
        if rad3 < best_radius:
            best_radius = rad3
            best_pts = pts3

    # Restart 4: Different random initial
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = normalize(random_pts)
        pts4, rad4 = optimize_from_initial(random_pts, 7.0)
        if rad4 < best_radius:
            best_radius = rad4
            best_pts = pts4

    # Restart 5: Another perturbation with different seed
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(777)
        perturbed2 = best_pts + rng.normal(0, 0.008, (N, 3))
        perturbed2 = normalize(perturbed2)
        pts5, rad5 = optimize_from_initial(perturbed2, 3.5)
        if rad5 < best_radius:
            best_radius = rad5
            best_pts = pts5

    # Final high-precision L-BFGS polish
    if time.time() < deadline - 4.0:
        try:
            def final_objective(params, beta=400.0):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                r2 = radii * radii
                max_r2 = np.max(r2)
                exp_terms = np.exp(beta * (r2 - max_r2))
                soft_max = (1.0 / beta) * np.log(np.sum(exp_terms)) + max_r2
                return soft_max

            res = minimize(final_objective, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 250, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 400})
            new_pts = normalize(res.x.reshape(N, 3))
            current_radius = compute_covering_radius(new_pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = new_pts.copy()
        except Exception:
            pass

    # Final Lloyd polish
    if time.time() < deadline - 2.0:
        pts_final_lloyd = lloyd_iteration(best_pts, n_iter=20, aggressiveness=1.0)
        rad_final_lloyd = compute_covering_radius(pts_final_lloyd)
        if rad_final_lloyd < best_radius:
            best_radius = rad_final_lloyd
            best_pts = pts_final_lloyd.copy()

    # Final volume + Lloyd micro-polish
    if time.time() < deadline - 1.0:
        pts_micro = volume_relaxation(best_pts, n_iter=20, strength=0.005)
        pts_micro = lloyd_iteration(pts_micro, n_iter=8, aggressiveness=0.7)
        rad_micro = compute_covering_radius(pts_micro)
        if rad_micro < best_radius:
            best_radius = rad_micro
            best_pts = pts_micro.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial(0.0)
        best_radius = compute_covering_radius(best_pts)

    best_pts = normalize(best_pts)
    return best_pts
