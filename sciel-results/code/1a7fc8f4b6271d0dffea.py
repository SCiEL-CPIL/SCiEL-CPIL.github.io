"""Improved packing of 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    def compute_exact_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0

        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0

        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)

        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)

        r = min(min_pair_dist / 2.0, min_radial_dist)
        return r

    def generate_hexagonal_initialization():
        """Generate a hexagonal lattice clipped to the unit circle."""
        best_pts = None
        best_score = -1.0

        for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
            spacing = 2.0 / (np.sqrt(N) * scale)

            nx = int(2.0 / spacing) + 5
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 5

            x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
            y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0

            pts_list = []
            for i, y in enumerate(y_coords):
                offset = 0.0 if i % 2 == 0 else spacing / 2.0
                for x in x_coords:
                    px = x + offset
                    py = y
                    if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                        pts_list.append([px, py])

            if len(pts_list) < N:
                continue

            pts = np.array(pts_list)
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:N]]

            score = compute_exact_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts

        if best_pts is None:
            best_pts = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_exact_score(best_pts)

        return best_pts, best_score

    def soft_min(x, y, beta):
        """Smooth approximation of min(x, y)."""
        if np.isscalar(x) and np.isscalar(y):
            m = min(x, y)
            return -1.0 / beta * (np.log(np.exp(-beta * (x - m)) + np.exp(-beta * (y - m))) - beta * m)
        # Vectorized
        m = np.minimum(x, y)
        return -1.0 / beta * (np.log(np.exp(-beta * (x - m)) + np.exp(-beta * (y - m))) - beta * m)

    def create_objective(pts_init, beta):
        """
        Create the objective function for L-BFGS-B.
        Returns negative of the soft-min radius.
        """
        def objective(params):
            if time.time() > deadline:
                return 1e10

            pts = params.reshape(-1, 2)

            if not np.all(np.isfinite(pts)):
                return 1e10

            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))

            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)

            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)

            x = min_pair_dist / 2.0
            y = min_radial_dist

            val = soft_min(x, y, beta)

            return -val

        return objective

    # 1. Generate initial points
    pts, best_score = generate_hexagonal_initialization()

    # 2. Multi-stage optimization with increasing beta
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]

    current_pts = pts
    current_score = best_score

    for beta in betas:
        if time.time() > deadline - 10:
            break

        objective = create_objective(current_pts, beta)
        x0 = current_pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-10})

            if result.success or result.fun < -current_score:
                new_pts = result.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)

                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score

        except Exception:
            pass

    # 3. Final polish with very high beta
    if time.time() < deadline - 5:
        beta_final = 5000.0
        objective = create_objective(current_pts, beta_final)
        x0 = current_pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-12})

            if result.fun < -current_score:
                new_pts = result.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)

                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # 4. Local perturbation search for further improvement
    if time.time() < deadline - 10:
        # Try small random perturbations of the best configuration
        best_local = current_pts.copy()
        best_local_score = current_score

        for _ in range(50):
            if time.time() > deadline - 10:
                break

            # Random perturbation
            noise = np.random.randn(N, 2) * 0.001
            perturbed = best_local + noise
            # Project back inside unit circle
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                perturbed[mask] *= (0.999 / norms[mask])[:, None]

            # Quick score check
            quick_score = compute_exact_score(perturbed)
            if quick_score > best_local_score:
                best_local = perturbed
                best_local_score = quick_score
                # Refine this perturbation
                objective = create_objective(best_local, 2000.0)
                x0 = best_local.flatten()
                bounds = [(-1.0, 1.0)] * (2 * N)
                try:
                    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                    if result.fun < -best_local_score:
                        new_pts = result.x.reshape(-1, 2)
                        new_score = compute_exact_score(new_pts)
                        if new_score > best_local_score:
                            best_local = new_pts
                            best_local_score = new_score
                except Exception:
                    pass

        if best_local_score > current_score:
            current_pts = best_local
            current_score = best_local_score

    # Final check and return
    if current_score <= 0:
        current_pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(current_pts, axis=1)
        current_pts = current_pts * (0.9 / norms)[:, None]

    return current_pts
# EVOLVE-BLOCK-END
