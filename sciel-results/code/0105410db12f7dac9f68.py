import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin before 150s kill

    def fibonacci_initial(phase=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + phase)
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

    def get_max_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def soft_max_objective(params, beta=200.0):
        """
        Soft-max surrogate: approximates max(radii) via log-sum-exp.
        """
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms

        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10

        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def sum_sq_objective(params):
        """Sum of squared radii + small variance term for balance."""
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms

        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10

        r = radii
        return np.sum(r**2) + 0.01 * np.var(r)

    def compute_forces(p, beta=50.0):
        """Compute smooth forces from soft-max energy + repulsion."""
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

        forces = np.zeros_like(p)

        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                forces[idx] += w * (cc - p[idx])

        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)

        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)

        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += 0.1 * forces_rep

        return forces, facets, circumcenters, radii

    def tangent_gradient_descent(pts, eta=0.005, iterations=200, beta=100.0):
        """
        Projected gradient descent on tangent space for the soft-max objective.
        """
        pts = pts.copy()
        best_local = pts.copy()
        best_radius = get_max_radius(pts)

        for step in range(iterations):
            if time.time() > deadline - 2:
                break

            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            r_max = np.max(radii)
            shifted = beta * (radii - r_max)
            exp_terms = np.exp(shifted)
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp

            grad = np.zeros_like(pts)
            for k in range(len(facets)):
                w = weights[k]
                if w < 1e-15:
                    continue
                cc = circumcenters[k]
                for idx in facets[k]:
                    dot_val = np.dot(cc, pts[idx])
                    dot_val = np.clip(dot_val, -0.999999, 0.999999)
                    denom = np.sqrt(1.0 - dot_val * dot_val)
                    if denom < 1e-10:
                        continue
                    grad[idx] += w * (-cc / denom)

            grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-12:
                break

            pts_new = pts - eta * grad_tangent
            pts_new = normalize(pts_new)

            new_radius = get_max_radius(pts_new)
            if new_radius < best_radius:
                best_radius = new_radius
                best_local = pts_new.copy()
                pts = pts_new
                eta *= 0.99
            else:
                eta *= 0.8
                if eta < 1e-6:
                    break

        return best_local, best_radius

    def epigraph_lbfgs(params, beta=100.0):
        """
        Epigraph formulation: minimize t subject to soft_max(radii) <= t.
        params: [x1..x150, t] where x is the 50x3 point array and t is the epigraph variable.
        """
        X = params[:N*3].reshape(N, 3)
        t = params[N*3]

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms

        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10

        r_max = np.max(radii)
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max

        # Penalty for violating t >= soft_max
        penalty = max(0.0, log_sum_exp - t) ** 2
        return t + 100.0 * penalty

    def epigraph_lbfgs_grad(params, beta=100.0):
        """Gradient for epigraph formulation."""
        X = params[:N*3].reshape(N, 3)
        t = params[N*3]

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms

        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return np.ones_like(params)

        r_max = np.max(radii)
        shifted = beta * (radii - r_max)
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp

        # Gradient of soft-max w.r.t. points
        grad_x = np.zeros_like(X)
        for k in range(len(facets)):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                dot_val = np.dot(cc, X[idx])
                dot_val = np.clip(dot_val, -0.999999, 0.999999)
                denom = np.sqrt(1.0 - dot_val * dot_val)
                if denom < 1e-10:
                    continue
                grad_x[idx] += w * (-cc / denom)

        # Project to tangent space
        grad_x_tangent = grad_x - np.sum(grad_x * X, axis=1, keepdims=True) * X

        # Penalty gradient
        log_sum_exp = np.log(sum_exp) / beta + r_max
        penalty_val = max(0.0, log_sum_exp - t)
        penalty_grad_t = -200.0 * penalty_val if penalty_val > 0 else 0.0
        penalty_grad_x = 200.0 * penalty_val * grad_x_tangent.flatten() if penalty_val > 0 else np.zeros(N*3)

        grad = np.zeros_like(params)
        grad[:N*3] = penalty_grad_x
        grad[N*3] = 1.0 + penalty_grad_t

        return grad

    def optimize_worst_facet(pts, step_size, iterations=100):
        """Iteratively move vertices of worst facet toward its circumcenter."""
        pts = pts.copy()
        prev_radius = np.pi

        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            current_radius = np.max(radii)
            worst_idx = np.argmax(radii)
            worst_center = circumcenters[worst_idx]

            if current_radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = current_radius

            target_indices_set = set()
            for k in range(len(facets)):
                if np.dot(circumcenters[k], worst_center) > 0.999:
                    for idx in facets[k]:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = pts[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    # Generate initial configurations
    initial_configs = [
        fibonacci_initial(0.0),
        fibonacci_initial(0.37),
        fibonacci_initial(0.618),
        fibonacci_initial(0.25),
        fibonacci_initial(0.1),
    ]

    # Add greedy initialization
    def greedy_init(seed_val):
        rng = np.random.RandomState(seed_val)
        pts = [rng.randn(3)]
        pts[0] = pts[0] / np.linalg.norm(pts[0])
        for _ in range(N - 1):
            candidates = rng.randn(3000, 3)
            candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)

    initial_configs.append(greedy_init(101))
    initial_configs.append(greedy_init(202))
    initial_configs.append(greedy_init(303))

    best_pts = initial_configs[0].copy()
    best_radius = get_max_radius(best_pts)

    # Phase 1: Soft-max L-BFGS from multiple starts
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 40.0:
            break

        def obj_sm(params, _beta=150.0):
            return soft_max_objective(params, beta=_beta)

        try:
            res = minimize(obj_sm, init_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 150})
            if res.fun < obj_sm(init_pts.flatten()):
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Phase 2: Epigraph L-BFGS (auxiliary variable t)
    if time.time() < deadline - 35.0:
        try:
            x0 = np.concatenate([best_pts.flatten(), [best_radius + 0.01]])
            res = minimize(epigraph_lbfgs, x0, jac=epigraph_lbfgs_grad, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 150})
            if res.fun < epigraph_lbfgs(x0):
                new_pts = normalize(res.x[:N*3].reshape(N, 3))
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Phase 3: Sum-of-squares L-BFGS for robust structure
    if time.time() < deadline - 30.0:
        try:
            res = minimize(sum_sq_objective, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 120, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 200})
            if res.fun < sum_sq_objective(best_pts.flatten()):
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Phase 4: Soft-max forces gradient descent
    if time.time() < deadline - 20.0:
        pts = best_pts.copy()
        eta = 0.02
        beta = 30.0

        for step in range(1500):
            if time.time() > deadline - 15.0:
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

            eta *= 0.998
            beta += 0.3

    # Phase 5: Tangent-space gradient descent (direct minimax approach)
    if time.time() < deadline - 12.0:
        pts_td, rad_td = tangent_gradient_descent(best_pts, eta=0.005, iterations=400, beta=100.0)
        if rad_td < best_radius:
            best_radius = rad_td
            best_pts = pts_td.copy()

    # Phase 6: Worst-facet optimization with multiple step sizes
    if time.time() < deadline - 10.0:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        for step in step_sizes:
            if time.time() > deadline - 6.0:
                break

            current = optimize_worst_facet(best_pts.copy(), step, iterations=80)
            facets, circumcenters, radii = compute_hull_data(current)
            if radii is not None:
                radius = np.max(radii)
                if radius < best_radius:
                    best_radius = radius
                    best_pts = current.copy()

    # Phase 7: Random restarts with perturbation
    if time.time() < deadline - 6.0:
        rng = np.random.RandomState(42)
        for restart in range(30):
            if time.time() > deadline - 3.0:
                break

            noise = rng.randn(N, 3) * 0.04
            perturbed = best_pts + noise
            perturbed = normalize(perturbed)

            pts = perturbed.copy()
            eta = 0.01
            beta = 80.0

            for step in range(300):
                if time.time() > deadline - 2.0:
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

                eta *= 0.995
                beta += 0.2

            # Tangent-space polish from restart
            if time.time() < deadline - 2.0:
                pts_td, rad_td = tangent_gradient_descent(pts, eta=0.003, iterations=150, beta=150.0)
                if rad_td < best_radius:
                    best_radius = rad_td
                    best_pts = pts_td.copy()

    # Phase 8: Final fine polish
    if time.time() < deadline - 1.5:
        pts = best_pts.copy()
        eta = 0.002
        beta = 150.0

        for step in range(300):
            if time.time() > deadline - 0.8:
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

        # Final tangent-space polish
        if time.time() < deadline - 0.5:
            pts_td, rad_td = tangent_gradient_descent(best_pts, eta=0.001, iterations=200, beta=200.0)
            if rad_td < best_radius:
                best_radius = rad_td
                best_pts = pts_td.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()

    best_pts = normalize(best_pts)

    return best_pts
