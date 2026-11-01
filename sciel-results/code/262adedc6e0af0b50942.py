import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # Step 1: Generate hexagonal lattice points
    def hex_lattice_init(n):
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        for _ in range(20):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            spacing *= 0.98
        # Fallback: random points
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.uniform(0, 0.9, n)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    pts = hex_lattice_init(N)

    # Step 2: Compute initial radius
    def compute_radius(pts):
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    best_radius = compute_radius(pts)
    best_pts = pts.copy()

    # Step 3: Smooth optimization using L-BFGS-B with soft-min surrogate
    def smooth_min(x, alpha=50.0):
        """Smooth approximation to min(x) using log-sum-exp"""
        x = np.asarray(x)
        x_clipped = np.clip(x, -100, 100)
        m = np.min(x_clipped)
        if m < -100:
            m = -100
        shifted = x_clipped - m
        log_sum = np.log(np.sum(np.exp(-alpha * shifted)))
        return m - log_sum / alpha

    def neg_smooth_min(x_flat, alpha=50.0):
        """Objective: negative of smooth min of all constraints (to minimize)"""
        if time.time() > deadline:
            return 1e10

        pts = x_flat.reshape(-1, 2)
        n = len(pts)

        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)

        # Boundary distances
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts

        # Collect all constraint values
        iu = np.triu_indices(n, k=1)
        pair_vals = dist[iu] / 2.0
        all_vals = np.concatenate([pair_vals, boundary])

        return -smooth_min(all_vals, alpha=alpha)

    # Optimize with L-BFGS-B using numerical gradient
    x0 = pts.flatten()

    # Use multiple alphas for annealing effect
    for alpha in [20.0, 40.0, 80.0, 120.0, 200.0, 400.0]:
        if time.time() > deadline - 20:
            break

        def obj(x_flat, a=alpha):
            return neg_smooth_min(x_flat, a)

        try:
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-13, 'gtol': 1e-12, 'maxls': 50}
            )

            if result.fun < obj(x0):
                pts_new = result.x.reshape(-1, 2)
                # Ensure points are inside the disk
                r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
                mask = r_new > 1.0
                if np.any(mask):
                    pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999

                r_new_val = compute_radius(pts_new)
                if r_new_val > best_radius:
                    best_radius = r_new_val
                    best_pts = pts_new.copy()
                    x0 = pts_new.flatten()
        except Exception:
            pass

    # Step 4: Final polish with a local search using small perturbations
    if time.time() < deadline - 10:
        # Try small perturbations around the best solution
        for trial in range(50):
            if time.time() > deadline - 5:
                break

            # Random small perturbation
            noise_scale = 0.001 * (1.0 - trial / 50.0)
            noise = np.random.normal(0, noise_scale, size=best_pts.shape)
            pts_perturbed = best_pts + noise

            # Keep inside disk
            r_p = np.hypot(pts_perturbed[:, 0], pts_perturbed[:, 1])
            mask = r_p > 0.999
            if np.any(mask):
                pts_perturbed[mask] = pts_perturbed[mask] / r_p[mask, np.newaxis] * 0.999

            # Try to optimize from this perturbed state with a high alpha
            if time.time() > deadline - 8:
                break

            def obj_pert(x_flat, a=500.0):
                return neg_smooth_min(x_flat, a)

            try:
                result = minimize(
                    obj_pert,
                    pts_perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-13}
                )

                if result.fun < obj_pert(pts_perturbed.flatten()):
                    pts_opt = result.x.reshape(-1, 2)
                    r_opt = np.hypot(pts_opt[:, 0], pts_opt[:, 1])
                    mask = r_opt > 1.0
                    if np.any(mask):
                        pts_opt[mask] = pts_opt[mask] / r_opt[mask, np.newaxis] * 0.9999

                    r_val = compute_radius(pts_opt)
                    if r_val > best_radius:
                        best_radius = r_val
                        best_pts = pts_opt.copy()
            except Exception:
                pass

    # Final validation: ensure all points are strictly inside and distinct
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r_final > 0.9999
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r_final[mask, np.newaxis] * 0.9999

    return best_pts
