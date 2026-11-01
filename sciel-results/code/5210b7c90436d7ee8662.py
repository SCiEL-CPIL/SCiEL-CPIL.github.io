"""Combined program: hex init + force-directed relaxation + L-BFGS-B soft-min polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_score(pts):
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
        """Generate a hexagonal lattice clipped to the unit circle, pick best N."""
        best_pts = None
        best_score = -1.0

        for spacing in [0.10, 0.105, 0.11, 0.115, 0.12, 0.125]:
            k = int(1.0 / spacing) + 3
            points_list = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    points_list.append([x, y])

            points = np.array(points_list)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii <= 1.0
            points = points[mask]
            radii = radii[mask]

            if len(points) < N:
                continue

            # Select N points closest to center for a compact disk shape
            idx = np.argsort(radii)[:N]
            init_points = points[idx]

            score = compute_score(init_points)
            if score > best_score:
                best_score = score
                best_pts = init_points.copy()

        if best_pts is None:
            # Fallback: random
            best_pts = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            norms[norms < 1e-9] = 1e-9
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_score(best_pts)

        return best_pts, best_score

    def soft_min(x, y, beta):
        """Smooth approximation of min(x, y) via log-sum-exp."""
        m = min(x, y)
        return -1.0 / beta * (np.log(np.exp(-beta * (x - m)) + np.exp(-beta * (y - m))) - beta * m)

    def force_directed_relax(pts, max_iters=3000, step_size=0.002, decay=0.9995):
        """Force-directed relaxation to resolve overlaps and optimize boundary usage."""
        pts = pts.copy()
        current_r = compute_score(pts)

        for it in range(max_iters):
            if time.time() > deadline - 30:  # Leave time for L-BFGS-B
                break

            # Compute pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)

            # Distance to boundary
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_dist_boundary = np.min(1.0 - radii)

            r = min(min_dist / 2.0, min_dist_boundary)

            if r > current_r:
                current_r = r

            # Target radius slightly larger than current to drive improvement
            target_r = r * 1.001 + 0.00005

            forces = np.zeros_like(pts)

            # Pairwise repulsion for overlapping pairs
            mask_overlap = dists < 2.0 * target_r
            mask_overlap[np.diag_indices_from(mask_overlap)] = False

            safe_dists = np.where(dists < 1e-9, 1e-9, dists)
            overlap = np.where(mask_overlap, 2.0 * target_r - dists, 0.0)

            force_matrix = overlap[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
            forces += np.sum(force_matrix, axis=1)

            # Boundary constraint: push inward if too close to edge
            mask_boundary = radii > 1.0 - target_r
            boundary_overlap = np.where(mask_boundary, radii - (1.0 - target_r), 0.0)
            safe_radii = np.where(radii < 1e-9, 1e-9, radii)
            boundary_forces = -pts / safe_radii[:, np.newaxis] * boundary_overlap[:, np.newaxis]
            forces += boundary_forces

            # Normalize and update
            force_magnitudes = np.linalg.norm(forces, axis=1, keepdims=True)
            max_force = np.max(force_magnitudes)
            if max_force > 1e-15:
                normalized_forces = forces / max_force
                pts += step_size * normalized_forces
            else:
                # No significant force; add tiny jitter
                pts += np.random.randn(N, 2) * 1e-5

            # Project back inside unit circle
            radii_new = np.hypot(pts[:, 0], pts[:, 1])
            out_of_bounds = radii_new > 1.0
            if np.any(out_of_bounds):
                pts[out_of_bounds] *= (0.9999 / radii_new[out_of_bounds])[:, np.newaxis]

            step_size *= decay

        return pts, current_r

    def l_bfgsb_polish(pts, beta, max_iter=150, max_fun=1000):
        """L-BFGS-B optimization with soft-min objective."""
        def objective(params):
            if time.time() > deadline:
                return 1e10
            p = params.reshape(-1, 2)
            if not np.all(np.isfinite(p)):
                return 1e10
            dists = np.linalg.norm(p, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = p[i_idx] - p[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            x = min_pair_dist / 2.0
            y = min_radial_dist
            val = soft_min(x, y, beta)
            return -val

        x0 = pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': max_iter, 'maxfun': max_fun,
                                       'ftol': 1e-13, 'gtol': 1e-11})
            new_pts = result.x.reshape(-1, 2)
            new_score = compute_score(new_pts)
            if new_score > 0:
                return new_pts, new_score
        except Exception:
            pass
        return pts, compute_score(pts)

    # 1. Generate initial points
    pts, best_score = generate_hexagonal_initialization()

    # 2. Force-directed relaxation
    pts, score_fd = force_directed_relax(pts, max_iters=3000)
    if score_fd > best_score:
        best_score = score_fd

    # 3. Multi-stage L-BFGS-B polish with increasing beta
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0]
    for beta in betas:
        if time.time() > deadline - 5:
            break
        pts, score_lb = l_bfgsb_polish(pts, beta)
        if score_lb > best_score:
            best_score = score_lb

    # 4. Final high-precision polish
    if time.time() < deadline - 2:
        pts, score_final = l_bfgsb_polish(pts, 2000.0, max_iter=100, max_fun=500)
        if score_final > best_score:
            best_score = score_final

    # Final validity check
    if best_score <= 0 or not np.all(np.isfinite(pts)):
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        norms[norms < 1e-9] = 1e-9
        pts = pts * (0.9 / norms)[:, None]

    return pts
