import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin

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

    def compute_forces(p, beta=50.0, repulsion_strength=0.0):
        """Compute smooth forces from soft-max energy (vectorized)."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None

        # Soft-max on radii directly for better conditioning
        max_r = np.max(radii)
        if max_r < 1e-12:
            return np.zeros_like(p), facets, circumcenters, radii
        
        exp_terms = np.exp(beta * (radii - max_r))
        sum_exp = np.sum(exp_terms)
        # Gradient of soft-max is proportional to exp(beta * (r - max_r)) * r
        # We want to move points towards circumcenters of high-weight facets
        weights = (1.0 / beta) * radii * exp_terms / sum_exp

        forces = np.zeros_like(p)
        vertex_indices = facets.ravel()
        facet_indices = np.repeat(np.arange(n_facets), 3)
        w_expanded = weights[facet_indices]
        cc_expanded = circumcenters[facet_indices]
        p_expanded = p[vertex_indices]
        
        # Move vertex p_v towards circumcenter cc with weight w
        # Force is proportional to (cc - p_v)
        contributions = w_expanded[:, np.newaxis] * (cc_expanded - p_expanded)
        np.add.at(forces, vertex_indices, contributions)

        if repulsion_strength > 0:
            diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diffs * diffs, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            inv_dist_sq = 1.0 / dist_sq
            np.fill_diagonal(inv_dist_sq, 0.0)
            forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
            forces += repulsion_strength * forces_rep

        return forces, facets, circumcenters, radii

    def facet_targeted_local_search(pts, time_budget=5.0):
        """
        Targeted local search: Identify worst facets and optimize their vertices.
        """
        pts = pts.copy()
        best_radius = compute_covering_radius(pts)
        end_time = time.time() + time_budget

        # Identify worst facets
        facets, circumcenters, radii = compute_hull_data(pts)
        if facets is None:
            return pts, best_radius

        # Select top 10 worst facets
        n_worst = min(10, len(facets))
        worst_indices = np.argsort(radii)[-n_worst:]
        
        # Group vertices by facet to avoid redundant optimization of same vertex sets
        # We will just iterate through the worst facets and try to improve them
        
        for w_idx in worst_indices:
            if time.time() > end_time - 0.5:
                break
                
            f = facets[w_idx]
            # Objective: minimize max radius of facets involving these 3 vertices
            # We use a soft-max on the radii of all facets, but only move these 3 vertices
            
            def local_obj(params):
                new_pts = pts.copy()
                new_pts[f[0]] = params[0:3]
                new_pts[f[1]] = params[3:6]
                new_pts[f[2]] = params[6:9]
                new_pts = normalize(new_pts)
                
                ff, cc, rr = compute_hull_data(new_pts)
                if ff is None:
                    return 1e10
                # Soft max with high beta to focus on the max
                beta = 200.0
                max_r = np.max(rr)
                shifted = beta * (rr - max_r)
                log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + max_r
                return log_sum_exp

            x0 = np.concatenate([pts[f[0]], pts[f[1]], pts[f[2]]])
            
            try:
                res = minimize(local_obj, x0, method='L-BFGS-B',
                               options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < local_obj(x0) - 1e-12:
                    new_pts = pts.copy()
                    new_pts[f[0]] = res.x[0:3]
                    new_pts[f[1]] = res.x[3:6]
                    new_pts[f[2]] = res.x[6:9]
                    new_pts = normalize(new_pts)
                    
                    new_radius = compute_covering_radius(new_pts)
                    if new_radius < best_radius:
                        best_radius = new_radius
                        pts = new_pts
            except Exception:
                pass

        return pts, best_radius

    def optimize_from_initial(pts_init, time_budget):
        """Run multi-phase optimization from a given initial configuration."""
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget

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

        # Phase 2: L-BFGS-B on soft-max surrogate
        if time.time() < end_time - 15.0:
            pts = best_pts.copy()

            def objective_softmax(params, beta=100.0):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                max_r = np.max(radii)
                shifted = beta * (radii - max_r)
                log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + max_r
                return log_sum_exp

            try:
                res = minimize(objective_softmax, pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 150})
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = compute_covering_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass

        # Phase 3: Facet-targeted local search
        if time.time() < end_time - 10.0:
            pts = best_pts.copy()
            time_left = min(10.0, end_time - time.time() - 2.0)
            if time_left > 1.0:
                new_pts, new_radius = facet_targeted_local_search(pts, time_left)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()

        # Phase 4: Fine soft-max polish with higher beta
        if time.time() < end_time - 5.0:
            pts = best_pts.copy()
            eta = 0.001
            beta = 100.0
            n_steps_p4 = 600

            for step in range(n_steps_p4):
                if time.time() > end_time - 3.0:
                    break

                forces, facets, circumcenters, radii = compute_forces(pts, beta, repulsion_strength=0.01 if step % 5 == 0 else 0.0)
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
                beta += 0.8

        # Phase 5: Final L-BFGS with multiple beta values
        if time.time() < end_time - 1.0:
            pts = best_pts.copy()
            for beta_val in [300.0, 800.0, 2000.0]:
                if time.time() > end_time - 0.5:
                    break
                try:
                    def obj_lbfgs(params, b=beta_val):
                        X = params.reshape(N, 3)
                        X = normalize(X)
                        facets, circumcenters, radii = compute_hull_data(X)
                        if facets is None:
                            return 1e10
                        max_r = np.max(radii)
                        shifted = b * (radii - max_r)
                        log_sum_exp = np.log(np.sum(np.exp(shifted))) / b + max_r
                        return log_sum_exp

                    res = minimize(obj_lbfgs, pts.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-17, 'gtol': 1e-16, 'maxls': 200})
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                except Exception:
                    pass

        return best_pts, best_radius

    # Multi-restart strategy
    # Restart 1: Fibonacci
    fib_pts = fibonacci_initial(0.0)
    best_pts, best_radius = optimize_from_initial(fib_pts, 85.0)

    # Restart 2: Fibonacci with phase shift
    if time.time() < deadline - 40.0:
        fib2 = fibonacci_initial(0.5)
        pts2, rad2 = optimize_from_initial(fib2, 25.0)
        if rad2 < best_radius:
            best_radius = rad2
            best_pts = pts2

    # Restart 3: Random perturbation of best
    if time.time() < deadline - 25.0:
        rng = np.random.default_rng(42)
        perturbed = best_pts + rng.normal(0, 0.05, (N, 3))
        perturbed = normalize(perturbed)
        pts3, rad3 = optimize_from_initial(perturbed, 18.0)
        if rad3 < best_radius:
            best_radius = rad3
            best_pts = pts3

    # Restart 4: Different random initial
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(123)
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = normalize(random_pts)
        pts4, rad4 = optimize_from_initial(random_pts, 10.0)
        if rad4 < best_radius:
            best_radius = rad4
            best_pts = pts4

    # Restart 5: Another perturbation with different seed
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(777)
        perturbed2 = best_pts + rng.normal(0, 0.025, (N, 3))
        perturbed2 = normalize(perturbed2)
        pts5, rad5 = optimize_from_initial(perturbed2, 7.0)
        if rad5 < best_radius:
            best_radius = rad5
            best_pts = pts5

    # Restart 6: Very small perturbation for fine-tuning
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(999)
        perturbed3 = best_pts + rng.normal(0, 0.005, (N, 3))
        perturbed3 = normalize(perturbed3)
        pts6, rad6 = optimize_from_initial(perturbed3, 4.0)
        if rad6 < best_radius:
            best_radius = rad6
            best_pts = pts6

    # Final high-precision L-BFGS polish
    if time.time() < deadline - 1.0:
        for beta_val in [1000.0, 3000.0, 8000.0]:
            if time.time() > deadline - 0.5:
                break
            try:
                def final_objective(params, b=beta_val):
                    X = params.reshape(N, 3)
                    X = normalize(X)
                    facets, circumcenters, radii = compute_hull_data(X)
                    if facets is None:
                        return 1e10
                    max_r = np.max(radii)
                    shifted = b * (radii - max_r)
                    log_sum_exp = np.log(np.sum(np.exp(shifted))) / b + max_r
                    return log_sum_exp

                res = minimize(final_objective, best_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 300})
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = compute_covering_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial(0.0)
        best_radius = compute_covering_radius(best_pts)

    best_pts = normalize(best_pts)
    return best_pts
