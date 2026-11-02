import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

# Precompute quadrature grid for smooth potential (Fibonacci lattice with ~2000 points)
_QUAD_N = 2000
def _make_quadrature_grid():
    i = np.arange(_QUAD_N) + 0.5
    z = 1.0 - 2.0 * i / _QUAD_N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return grid

_QUAD_GRID = _make_quadrature_grid()

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
        forces += 0.03 * forces_rep

        return forces, facets, circumcenters, radii

    def compute_reciprocal_distance_energy(p, epsilon=0.01):
        """Compute mean reciprocal distance to nearest site over quadrature grid.
        This provides a smooth, globally differentiable signal that penalizes large empty caps."""
        # Compute distances from each quadrature point to each site
        # dist_sq[i,j] = ||quad_grid[i] - p[j]||^2
        # Angular distance: arccos(quad_grid[i] . p[j])
        dots = _QUAD_GRID @ p.T  # shape (QUAD_N, N)
        dots = np.clip(dots, -1.0, 1.0)
        # Angular distance in radians
        ang_dist = np.arccos(dots)  # shape (QUAD_N, N)
        # Find minimum angular distance to any site
        min_ang = np.min(ang_dist, axis=1)  # shape (QUAD_N,)
        # Mean reciprocal: integral of 1/(d + epsilon) over sphere, approximated
        energy = np.mean(1.0 / (min_ang + epsilon))
        return energy

    def compute_reciprocal_distance_gradient(p, epsilon=0.01):
        """Compute gradient of mean reciprocal distance energy w.r.t. site positions.
        Uses analytical derivative through the min operation (subgradient)."""
        dots = _QUAD_GRID @ p.T  # shape (QUAD_N, N)
        dots = np.clip(dots, -1.0, 1.0)
        ang_dist = np.arccos(dots)  # shape (QUAD_N, N)
        min_idx = np.argmin(ang_dist, axis=1)  # shape (QUAD_N,)
        min_ang = ang_dist[np.arange(_QUAD_N), min_idx]
        
        # Gradient of 1/(min_ang + eps) w.r.t. ang_dist for the nearest site
        # d/dx [1/(x+eps)] = -1/(x+eps)^2
        weight = -1.0 / (min_ang + epsilon) ** 2  # shape (QUAD_N,)
        
        # Gradient of arccos(u) w.r.t. u = -1/sqrt(1-u^2)
        u = dots[np.arange(_QUAD_N), min_idx]  # shape (QUAD_N,)
        sqrt_term = np.sqrt(np.maximum(1e-16, 1.0 - u * u))
        dang_du = -1.0 / sqrt_term  # shape (QUAD_N,)
        
        # Combine: d energy / d dots[i, min_idx[i]]
        d_energy_d_dots = weight * dang_du  # shape (QUAD_N,)
        
        # Gradient w.r.t. p: dots[i,j] = quad_grid[i] . p[j]
        # So d dots[i,j] / d p[j] = quad_grid[i]
        # Only the nearest site contributes for each quadrature point
        grad = np.zeros_like(p)  # shape (N, 3)
        for i in range(_QUAD_N):
            j = min_idx[i]
            grad[j] += d_energy_d_dots[i] * _QUAD_GRID[i]
        
        # Normalize by number of quadrature points (mean)
        grad /= _QUAD_N
        return grad

    def optimize_from_initial(pts_init, time_budget):
        """Run multi-phase optimization from a given initial configuration."""
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget

        # Phase 1: Coarse soft-max gradient descent
        pts = best_pts.copy()
        eta = 0.02
        beta = 25.0
        n_steps = 600

        for step in range(n_steps):
            if time.time() > end_time - 25.0:
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

        # Phase 2: L-BFGS-B on soft-max surrogate for covering radius
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

            def objective_sum_sq(params):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                return np.sum(radii**2) + 0.005 * np.var(radii)

            for obj_func in [objective_softmax, objective_sum_sq]:
                if time.time() > end_time - 15.0:
                    break
                try:
                    res = minimize(obj_func, pts.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 60, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100})
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                except Exception:
                    pass

        # Phase 2b: NEW - L-BFGS-B on mean reciprocal distance smooth potential
        if time.time() < end_time - 12.0:
            pts = best_pts.copy()
            
            def objective_reciprocal(params, epsilon=0.01):
                X = params.reshape(N, 3)
                X = normalize(X)
                return compute_reciprocal_distance_energy(X, epsilon)

            def grad_reciprocal(params, epsilon=0.01):
                X = params.reshape(N, 3)
                X = normalize(X)
                grad = compute_reciprocal_distance_gradient(X, epsilon)
                # Project gradient to tangent space
                for i in range(N):
                    radial = np.dot(grad[i], X[i])
                    grad[i] -= radial * X[i]
                return grad.flatten()

            try:
                res = minimize(objective_reciprocal, pts.flatten(), jac=grad_reciprocal,
                              method='L-BFGS-B',
                              options={'maxiter': 40, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = compute_covering_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
            except Exception:
                pass

        # Phase 3: Direct minimax projected gradient with adaptive threshold
        if time.time() < end_time - 10.0:
            pts = best_pts.copy()
            eta = 0.004
            threshold_frac = 0.92

            for step in range(600):
                if time.time() > end_time - 6.0:
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

                for i in range(N):
                    f_norm = np.linalg.norm(forces[i])
                    if f_norm > 1e-12:
                        forces[i] /= f_norm

                pts += eta * forces
                pts = normalize(pts)

                if step % 80 == 0:
                    threshold_frac = min(0.98, threshold_frac + 0.01)
                eta *= 0.996

        # Phase 4: Fine soft-max polish with higher beta
        if time.time() < end_time - 3.0:
            pts = best_pts.copy()
            eta = 0.0015
            beta = 150.0

            for step in range(200):
                if time.time() > end_time - 1.5:
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

        return best_pts, best_radius

    # Multi-restart strategy
    # Restart 1: Fibonacci
    fib_pts = fibonacci_initial(0.0)
    best_pts, best_radius = optimize_from_initial(fib_pts, 80.0)

    # Restart 2: Fibonacci with phase shift
    if time.time() < deadline - 35.0:
        fib2 = fibonacci_initial(0.5)
        pts2, rad2 = optimize_from_initial(fib2, 22.0)
        if rad2 < best_radius:
            best_radius = rad2
            best_pts = pts2

    # Restart 3: Random perturbation of best
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        perturbed = best_pts + rng.normal(0, 0.03, (N, 3))
        perturbed = normalize(perturbed)
        pts3, rad3 = optimize_from_initial(perturbed, 15.0)
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

    # Restart 5: Another perturbation with different seed
    if time.time() < deadline - 7.0:
        rng = np.random.default_rng(777)
        perturbed2 = best_pts + rng.normal(0, 0.02, (N, 3))
        perturbed2 = normalize(perturbed2)
        pts5, rad5 = optimize_from_initial(perturbed2, 5.0)
        if rad5 < best_radius:
            best_radius = rad5
            best_pts = pts5

    # Final high-precision L-BFGS polish with soft-max
    if time.time() < deadline - 4.0:
        try:
            def final_objective(params, beta=300.0):
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
                           options={'maxiter': 80, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200})
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
