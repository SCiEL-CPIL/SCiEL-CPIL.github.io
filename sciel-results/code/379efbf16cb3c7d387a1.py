"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    def compute_facet_data(pts):
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.array([]), np.array([]), np.array([])

        simplices = hull.simplices
        F = len(simplices)

        v1 = pts[simplices[:, 0]]
        v2 = pts[simplices[:, 1]]
        v3 = pts[simplices[:, 2]]

        edge1 = v2 - v1
        edge2 = v3 - v1
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)

        valid = (norms[:, 0] > 1e-12)
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([])

        normals = normals[valid]
        norms = norms[valid]
        v1 = v1[valid]
        v2 = v2[valid]
        v3 = v3[valid]
        simplices = simplices[valid]

        centers = normals / norms

        dots = np.sum(centers * v1, axis=1)
        centers[dots < 0] *= -1

        diffs = centers - v1
        d = np.linalg.norm(diffs, axis=1)
        radii = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))

        return radii, centers, simplices

    def compute_covering_radius(pts):
        radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            return np.pi, np.array([0.0, 0.0, 1.0])
        max_idx = np.argmax(radii)
        return radii[max_idx], centers[max_idx]

    def soft_max_radius_and_grad(pts, beta=50.0):
        radii, centers, simplices = compute_facet_data(pts)
        if len(radii) == 0:
            return np.pi, np.zeros_like(pts)

        max_r = np.max(radii)
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)

        grads = np.zeros_like(pts)
        F = len(radii)

        for i in range(F):
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

            simplex = simplices[i]
            p_pts = pts[simplex]

            grad_contrib = -C / sin_r

            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps

            for k in range(3):
                grads[simplex[k]] += w * tangential_grads[k]

        return soft_radius, grads

    def lbfgs_stage(pts, beta_schedule, time_budget):
        current = pts.copy()
        best = current.copy()
        best_radius, _ = compute_covering_radius(best)

        for beta in beta_schedule:
            if time.time() > time_budget:
                break

            def objective_and_grad(x, b=beta):
                if time.time() > time_budget:
                    return 1e10, np.zeros_like(x)
                pts_local = x.reshape(N, 3)
                norms = np.linalg.norm(pts_local, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                pts_local = pts_local / norms
                soft_r, grads = soft_max_radius_and_grad(pts_local, beta=b)
                return soft_r, grads.reshape(-1)

            try:
                x0 = current.flatten()
                remaining = time_budget - (time.time() - start_time)
                if remaining < 5:
                    break
                max_iter = int(min(500, max(100, 100 * (remaining / 30.0))))

                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
                )

                if result.success or result.fun < objective_and_grad(x0)[0]:
                    lbfgs_pts = project_to_sphere(result.x.reshape(N, 3))
                    lbfgs_radius, _ = compute_covering_radius(lbfgs_pts)
                    if lbfgs_radius < best_radius:
                        best_radius = lbfgs_radius
                        best = lbfgs_pts.copy()
                        current = lbfgs_pts.copy()
            except Exception:
                pass

        return best, best_radius

    def facet_targeted_move(pts, step_size, iterations=100):
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

            if radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = radius

            try:
                hull = ConvexHull(pts)
            except:
                break

            facet_data = []
            for simplex in hull.simplices:
                v1 = pts[simplex[0]]
                v2 = pts[simplex[1]]
                v3 = pts[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                facet_data.append((c, simplex))

            if not facet_data:
                break

            target_indices_set = set()
            for c, simplex in facet_data:
                if np.dot(c, worst_center) > 0.999:
                    for idx in simplex:
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

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: L-BFGS with increasing beta
    if time.time() < deadline - 60:
        try:
            best, best_radius = lbfgs_stage(current, [30.0, 80.0, 150.0, 300.0], deadline - 60)
            current = best.copy()
        except Exception:
            pass

    # Phase 3: Facet-targeted moves
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for step in step_sizes:
        if time.time() > deadline - 40:
            break
        current = facet_targeted_move(current, step, iterations=100)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 4: More L-BFGS polishing
    if time.time() < deadline - 30:
        try:
            best, best_radius = lbfgs_stage(best, [500.0, 1000.0, 2000.0], deadline - 30)
            current = best.copy()
        except Exception:
            pass

    # Phase 5: Final facet moves with very small steps
    for step in [0.0005, 0.0002, 0.0001]:
        if time.time() > deadline - 10:
            break
        current = facet_targeted_move(current, step, iterations=200)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 6: Final high-precision L-BFGS
    if time.time() < deadline - 5:
        try:
            best, best_radius = lbfgs_stage(best, [5000.0, 10000.0], deadline - 5)
        except Exception:
            pass

    # Final normalization and validation
    best = project_to_sphere(best)
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
