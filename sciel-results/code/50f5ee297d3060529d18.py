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

    def compute_facet_gradients(p, beta=50.0):
        """Compute smooth gradients for soft-max covering radius (vectorized)."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        weights = (2.0 / beta) * radii * exp_terms / sum_exp

        # Vectorized gradient accumulation
        forces = np.zeros_like(p)
        vertex_indices = facets.ravel()
        facet_indices = np.repeat(np.arange(n_facets), 3)
        w_expanded = weights[facet_indices]
        cc_expanded = circumcenters[facet_indices]
        p_expanded = p[vertex_indices]

        contributions = w_expanded[:, np.newaxis] * (cc_expanded - p_expanded)
        np.add.at(forces, vertex_indices, contributions)

        return forces, radii, max_r2

    def compute_covering_radius(p):
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return float('inf')
        return np.max(radii)

    def compute_forces(p, beta=50.0, repulsion_strength=0.03):
        """Compute smooth forces from soft-max energy + repulsion (vectorized)."""
        forces, radii, max_r2 = compute_facet_gradients(p, beta)
        if forces is None:
            return None, None, None, None

        if repulsion_strength > 0:
            diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diffs * diffs, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            inv_dist_sq = 1.0 / dist_sq
            np.fill_diagonal(inv_dist_sq, 0.0)
            forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
            forces += repulsion_strength * forces_rep

        facets, circumcenters, _ = compute_hull_data(p)
        return forces, facets, circumcenters, radii

    def optimize_from_initial(pts_init, time_budget):
        """Run multi-phase optimization from a given initial configuration."""
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget

        # Phase 1: Coarse soft-max gradient descent with adaptive step
        pts = best_pts.copy()
        eta = 0.015
        beta = 20.0
        n_steps = 600
        patience = 0
        best_local = best_radius

        for step in range(n_steps):
            if time.time() > end_time - 25.0:
                break

            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break

            # Adaptive step size: reduce if radius increases
            pts_new = pts + eta * forces
            pts_new = normalize(pts_new)
            current_radius = np.max(radii) if radii is not None else best_radius

            if current_radius < best_local:
                pts = pts_new
                best_local = current_radius
                patience = 0
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
            else:
                patience += 1
                if patience > 50:
                    eta *= 0.5
                    patience = 0

            pts = normalize(pts)
            eta *= 0.998
            beta += 0.2

        # Phase 2: L-BFGS-B on soft-max surrogate with multiple betas
        if time.time() < end_time - 20.0:
            pts = best_pts.copy()

            def objective_softmax(params, beta=50.0):
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

            for beta_val in [30.0, 80.0, 200.0]:
                if time.time() > end_time - 15.0:
                    break
                try:
                    res = minimize(objective_softmax, pts.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 120, 'maxfun': 2000},
                                   args=(beta_val,))
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                except Exception:
                    pass

        # Phase 3: Direct minimax projected gradient with adaptive threshold (vectorized)
        if time.time() < end_time - 12.0:
            pts = best_pts.copy()
            eta = 0.002
            threshold_frac = 0.85
            n_steps_p3 = 2000
            patience = 0
            best_local = best_radius

            for step in range(n_steps_p3):
                if time.time() > end_time - 8.0:
                    break

                facets, circumcenters, radii = compute_hull_data(pts)
                if facets is None:
                    break

                current_radius = np.max(radii)
                if current_radius < best_local:
                    best_local = current_radius
                    patience = 0
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()
                else:
                    patience += 1

                threshold = threshold_frac * current_radius
                active_facets = np.where(radii >= threshold)[0]

                if len(active_facets) == 0:
                    break

                forces = np.zeros_like(pts)

                # Use smooth quadratic weights
                excess = radii[active_facets] - threshold
                max_excess = np.max(excess) if len(excess) > 0 else 1e-12
                if max_excess < 1e-12:
                    max_excess = 1e-12
                facet_weights = (excess / max_excess) ** 2
                facet_weights = facet_weights / (np.sum(facet_weights) + 1e-12)

                # Vectorized tangential force accumulation
                af = active_facets
                w_exp = facet_weights
                cc_exp = circumcenters[af]
                verts = facets[af]
                p_verts = pts[verts]
                
                diff = cc_exp[:, np.newaxis, :] - p_verts
                radial = np.sum(diff * p_verts, axis=2)
                tangential = diff - radial[:, :, np.newaxis] * p_verts
                contrib = w_exp[:, np.newaxis, np.newaxis] * tangential
                
                np.add.at(forces, verts.ravel(), contrib.reshape(-1, 3))

                # Add small repulsion every 3 steps
                if step % 3 == 0:
                    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                    dist_sq = np.sum(diffs * diffs, axis=2)
                    np.fill_diagonal(dist_sq, 1.0)
                    inv_dist_sq = 1.0 / dist_sq
                    np.fill_diagonal(inv_dist_sq, 0.0)
                    forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
                    forces += 0.008 * forces_rep

                # Normalize forces
                f_norms = np.linalg.norm(forces, axis=1, keepdims=True)
                f_norms[f_norms < 1e-12] = 1.0
                forces /= f_norms

                pts += eta * forces
                pts = normalize(pts)

                if step % 100 == 0:
                    threshold_frac = min(0.96, threshold_frac + 0.004)
                eta *= 0.9985

                if patience > 200:
                    eta *= 0.5
                    patience = 0

        # Phase 4: Fine soft-max polish with higher beta (vectorized forces)
        if time.time() < end_time - 6.0:
            pts = best_pts.copy()
            eta = 0.0008
            beta = 150.0
            n_steps_p4 = 800
            patience = 0
            best_local = best_radius

            for step in range(n_steps_p4):
                if time.time() > end_time - 3.0:
                    break

                forces, facets, circumcenters, radii = compute_forces(pts, beta, repulsion_strength=0.005 if step % 5 == 0 else 0.0)
                if forces is None:
                    break

                pts += eta * forces
                pts = normalize(pts)

                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_local:
                        best_local = current_radius
                        patience = 0
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_pts = pts.copy()
                    else:
                        patience += 1

                eta *= 0.9975
                beta += 0.6

                if patience > 100:
                    eta *= 0.5
                    patience = 0

        # Phase 5: Final L-BFGS with multiple beta values and tighter tolerances
        if time.time() < end_time - 1.0:
            pts = best_pts.copy()
            for beta_val in [300.0, 800.0, 2000.0, 5000.0]:
                if time.time() > end_time - 0.5:
                    break
                try:
                    def obj_lbfgs(params, b=beta_val):
                        X = params.reshape(N, 3)
                        X = normalize(X)
                        facets, circumcenters, radii = compute_hull_data(X)
                        if facets is None:
                            return 1e10
                        r2 = radii * radii
                        max_r2 = np.max(r2)
                        exp_terms = np.exp(b * (r2 - max_r2))
                        soft_max = (1.0 / b) * np.log(np.sum(exp_terms)) + max_r2
                        return soft_max

                    res = minimize(obj_lbfgs, pts.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 120, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 200, 'maxfun': 3000})
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                except Exception:
                    pass

        return best_pts, best_radius

    # Multi-restart strategy with better time allocation
    # Restart 1: Fibonacci - primary optimization
    fib_pts = fibonacci_initial(0.0)
    best_pts, best_radius = optimize_from_initial(fib_pts, 90.0)

    # Restart 2: Fibonacci with phase shift
    if time.time() < deadline - 35.0:
        fib2 = fibonacci_initial(0.5)
        pts2, rad2 = optimize_from_initial(fib2, 20.0)
        if rad2 < best_radius:
            best_radius = rad2
            best_pts = pts2

    # Restart 3: Small perturbation of best
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        perturbed = best_pts + rng.normal(0, 0.02, (N, 3))
        perturbed = normalize(perturbed)
        pts3, rad3 = optimize_from_initial(perturbed, 12.0)
        if rad3 < best_radius:
            best_radius = rad3
            best_pts = pts3

    # Restart 4: Different random initial
    if time.time() < deadline - 12.0:
        rng = np.random.default_rng(123)
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = normalize(random_pts)
        pts4, rad4 = optimize_from_initial(random_pts, 8.0)
        if rad4 < best_radius:
            best_radius = rad4
            best_pts = pts4

    # Restart 5: Very small perturbation for fine-tuning
    if time.time() < deadline - 6.0:
        rng = np.random.default_rng(777)
        perturbed2 = best_pts + rng.normal(0, 0.01, (N, 3))
        perturbed2 = normalize(perturbed2)
        pts5, rad5 = optimize_from_initial(perturbed2, 5.0)
        if rad5 < best_radius:
            best_radius = rad5
            best_pts = pts5

    # Final high-precision L-BFGS polish with multiple beta values
    if time.time() < deadline - 1.0:
        for beta_val in [300.0, 1000.0, 3000.0, 8000.0]:
            if time.time() > deadline - 0.5:
                break
            try:
                def final_objective(params, b=beta_val):
                    X = params.reshape(N, 3)
                    X = normalize(X)
                    facets, circumcenters, radii = compute_hull_data(X)
                    if facets is None:
                        return 1e10
                    r2 = radii * radii
                    max_r2 = np.max(r2)
                    exp_terms = np.exp(b * (r2 - max_r2))
                    soft_max = (1.0 / b) * np.log(np.sum(exp_terms)) + max_r2
                    return soft_max

                res = minimize(final_objective, best_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 250, 'maxfun': 4000})
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
