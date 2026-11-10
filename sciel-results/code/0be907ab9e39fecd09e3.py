import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    # Generate initial hexagonal lattice
    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts)

    # Initial spacing estimate based on area
    initial_spacing = 2.0 * np.sqrt(0.907 / N)

    # Generate initial hexagonal lattice
    spacing = initial_spacing
    while True:
        pts = gen_hex_lattice(spacing)
        if len(pts) >= N:
            break
        spacing *= 0.98

    # Select N points closest to center
    radii = np.hypot(pts[:, 0], pts[:, 1])
    points = pts[np.argsort(radii, kind="stable")[:N]].copy()

    best_score = compute_score(points)
    best_points = points.copy()

    # Phase 1: Coarse force-directed optimization
    step_size = spacing * 0.1
    lr_decay = 0.9995
    max_iter = 3000

    for iteration in range(max_iter):
        if time.time() > deadline - 30:
            break

        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist < 1e-10] = 1e-10
        np.fill_diagonal(dist, np.inf)

        # Only consider close pairs for efficiency
        close_mask = (dist < 5 * spacing) & (dist < np.inf)
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[close_mask] = 1.0 / (dist[close_mask] ** 2)

        # Repulsion forces
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

        # Boundary constraint force
        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        boundary_margin = 0.1
        over = radii > (1.0 - boundary_margin)
        if np.any(over):
            mag = (radii[over] - (1.0 - boundary_margin)) * 5.0
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]

        forces += boundary_force

        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force

        # Update positions
        points += step_size * forces

        # Project back inside unit circle
        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii[mask_out, np.newaxis]

        # Compute score
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()

        # Anneal step size
        step_size *= lr_decay

        if step_size < 1e-6:
            break

    # Phase 2: Fine-tuning with gradient-based approach
    if time.time() < deadline - 15:
        points = best_points.copy()
        polish_step = 0.005
        polish_decay = 0.999
        polish_deadline = deadline - 5

        for step in range(5000):
            if time.time() > polish_deadline:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)

            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(points, axis=1)
            min_boundary_dist = np.min(1.0 - norms)

            current_score = min(min_pair_dist / 2.0, min_boundary_dist)

            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()

            # Use top 20 closest pairs for precision
            flat_dists = dists.flatten()
            k = min(20, N * (N - 1) // 2)
            part_idx = np.argpartition(flat_dists, k)[:k]

            pair_grad = np.zeros_like(points)
            for idx in part_idx:
                i = int(idx // N)
                j = int(idx % N)
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-6:
                    continue
                vec = (points[i] - points[j]) / d
                pair_grad[i] += vec
                pair_grad[j] -= vec

            # Boundary gradient for points close to boundary
            boundary_grad = np.zeros_like(points)
            close_to_boundary = norms > (1.0 - current_score - 0.002)
            for i in np.where(close_to_boundary)[0]:
                if norms[i] > 1e-6:
                    boundary_grad[i] = -points[i] / norms[i]

            # Combine based on which constraint is tighter
            pair_score = min_pair_dist / 2.0
            bound_score = min_boundary_dist

            if pair_score <= bound_score:
                w_pair = 1.0
                w_bound = 0.2
            else:
                w_pair = 0.2
                w_bound = 1.0

            grad = w_pair * pair_grad + w_bound * boundary_grad

            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-6:
                grad = grad / g_norm

            points += polish_step * grad

            # Keep inside
            norms_new = np.linalg.norm(points, axis=1)
            outside = norms_new > 0.999
            if np.any(outside):
                points[outside] = points[outside] * (0.999 / norms_new[outside, np.newaxis])

            polish_step = max(1e-5, polish_step * polish_decay)

    # Phase 3: Final local optimization with scipy
    if time.time() < deadline - 3:
        points = best_points.copy()
        n = len(points)

        # Get the k closest pairs
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)

        flat_dist = dist.flatten()
        k = min(30, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k)[:k]

        pairs = []
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            if i < j:
                pairs.append((i, j))

        # Also check boundary
        radii = np.linalg.norm(points, axis=1)
        boundary_violators = np.where(radii > 0.95)[0]

        def objective(x):
            pts = x.reshape(-1, 2)
            penalty = 0.0
            for i, j in pairs:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 0.12:
                    penalty += (0.12 - d) ** 2
            r = np.linalg.norm(pts, axis=1)
            over = r > 0.95
            if np.any(over):
                penalty += np.sum((r[over] - 0.95) ** 2) * 10
            return penalty

        x0 = points.flatten()
        try:
            result = minimize(objective, x0, method='BFGS', options={'maxiter': 50, 'gtol': 1e-7})
            if result.success:
                new_points = result.x.reshape(-1, 2)
                radii = np.linalg.norm(new_points, axis=1)
                mask_out = radii > 1.0
                if np.any(mask_out):
                    new_points[mask_out] /= radii[mask_out, np.newaxis]

                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass

    return best_points
