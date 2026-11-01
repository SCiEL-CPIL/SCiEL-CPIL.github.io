"""Combined program: Force-directed relaxation + Soft-max constraint optimization with radius lifting."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def _score(points):
        """Compute packing radius."""
        if points.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms >= 1.0):
            return 0.0
        boundary_margin = np.min(1.0 - norms)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        return min(min_pair_dist / 2.0, boundary_margin)

    def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
        """Generate points from concentric rings."""
        points = np.zeros((N, 2))
        idx = 0
        if angle_offsets is None:
            angle_offsets = [0.0] * len(ring_sizes)
        for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
            if ring_idx >= len(angle_offsets):
                offset = 0.0
            else:
                offset = angle_offsets[ring_idx]
            for j in range(size):
                angle = 2 * np.pi * j / size + offset
                points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
                idx += 1
        return points

    def _hex_lattice_init(r_guess=0.12):
        """Hexagonal lattice initialization."""
        spacing = 2.0 * r_guess
        pts = []
        k = int(1.0 / spacing) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - r_guess) ** 2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            return pts[idx]
        else:
            return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])

    def _random_init(seed=None):
        if seed is not None:
            np.random.seed(seed)
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 0.9, N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points

    def _force_directed(points, max_iters=1000, dt=0.005):
        """Force-directed optimization with repulsion."""
        points = points.copy()
        n = points.shape[0]
        best_score = _score(points)
        best_points = points.copy()

        for it in range(max_iters):
            if time.time() - start_time > deadline:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)

            forces = np.zeros_like(points)
            current_r = _score(points)
            threshold = 2.5 * current_r if current_r > 0.01 else 0.4

            # Repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    if d[i, j] < threshold:
                        f = diff[i, j] / (d[i, j] ** 3 + 1e-8)
                        forces[i] += f
                        forces[j] -= f

            # Boundary constraint
            dists = np.linalg.norm(points, axis=1)
            for i in range(n):
                if dists[i] > 0.95:
                    forces[i] -= 0.5 * points[i] / dists[i]

            points += dt * forces

            # Project inside
            dists = np.linalg.norm(points, axis=1)
            for i in range(n):
                if dists[i] > 0.99:
                    points[i] *= 0.99 / dists[i]

            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()

            dt *= 0.999

        return best_points, best_score

    def _soft_max_obj_grad(pts, r_target, soft_beta):
        """
        Soft-max constraint violation objective.
        Minimizes the maximum violation of:
        1. |c_i| <= 1 - r_target
        2. |c_i - c_j| >= 2 * r_target
        """
        n = len(pts)
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        cont_viol = norms - (1.0 - r_target)

        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_viol = 2.0 * r_target - dists
        iu = np.triu_indices(n, k=1)
        pair_viol_flat = pair_viol[iu]

        all_viol = np.concatenate([cont_viol, pair_viol_flat])

        max_v = np.max(all_viol)
        if max_v > 50.0 / soft_beta:
            return max_v, np.zeros_like(pts)

        exp_v = np.exp(soft_beta * (all_viol - max_v))
        sum_exp = np.sum(exp_v)
        soft_max_val = max_v + np.log(sum_exp) / soft_beta
        weights = exp_v / sum_exp

        grad = np.zeros_like(pts)

        # Container gradient
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_vecs = pts / safe_norms[:, np.newaxis]
        grad += weights[:n, np.newaxis] * unit_vecs

        # Pairwise gradient
        iw = weights[n:]
        valid = dists[iu] > 1e-10
        if np.any(valid):
            iw_valid = iw[valid]
            iu_valid = (iu[0][valid], iu[1][valid])
            d_valid = dists[iu_valid]
            diff_valid = pts[iu_valid[0]] - pts[iu_valid[1]]
            unit_diff = diff_valid / d_valid[:, np.newaxis]

            np.add.at(grad, iu_valid[0], -iw_valid[:, np.newaxis] * unit_diff)
            np.add.at(grad, iu_valid[1], iw_valid[:, np.newaxis] * unit_diff)

        return soft_max_val, grad

    def _optimize_for_radius(pts_init, r, soft_beta, max_iter=500):
        """Minimize soft constraint violation for fixed r."""
        n = pts_init.shape[0]

        def objective(x):
            pts = x.reshape(n, 2)
            val, grad = _soft_max_obj_grad(pts, r, soft_beta)
            return val, grad.flatten()

        x0 = pts_init.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8,
                                   'maxls': 50})
        pts_opt = result.x.reshape(n, 2)
        return pts_opt, result.fun

    def _lift_and_refine(pts_init, r_start, time_budget):
        """Iteratively increase target radius and refine."""
        current_pts = pts_init.copy()
        current_score = _score(current_pts)
        r_target = max(r_start, current_score * 1.001)
        
        # Adaptive beta: start low (smooth), increase (sharp)
        betas = [10.0, 50.0, 200.0, 1000.0]
        
        for beta in betas:
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Force-directed step to escape local minima
            current_pts, fd_score = _force_directed(current_pts, max_iters=200, dt=0.002)
            if fd_score > current_score:
                current_score = fd_score
                r_target = max(r_target, current_score * 1.001)

            # L-BFGS refinement
            current_pts, val = _optimize_for_radius(current_pts, r_target, beta, max_iter=300)
            
            # Check if we improved
            new_score = _score(current_pts)
            if new_score > current_score:
                current_score = new_score
                # Try to lift the target radius
                r_target = new_score * 1.005
            else:
                # If no improvement, try a smaller target to relax
                r_target = current_score * 0.999

        return current_pts, current_score

    # Initializations
    inits = []
    # 1. Ring structure (1+6+12+18+13)
    ring_sizes = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.2, 0.4, 0.6, 0.85]
    inits.append(_ring_init(ring_sizes, ring_radii))
    
    # 2. Hex lattice
    inits.append(_hex_lattice_init(0.125))
    
    # 3. Random
    inits.append(_random_init(42))
    inits.append(_random_init(123))

    best_pts = None
    best_score = -1.0

    for idx, init_pts in enumerate(inits):
        if time.time() - start_time > deadline - 10.0:
            break
        
        # Allocate time budget
        remaining = deadline - time.time() - 5.0
        time_budget = min(20.0, remaining * 0.8)
        
        try:
            pts, score = _lift_and_refine(init_pts, r_start=0.12, time_budget=time_budget)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue

    # Final polish on best result
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            # High precision polish
            current_score = _score(best_pts)
            r_target = current_score * 1.001
            beta = 5000.0
            best_pts, _ = _optimize_for_radius(best_pts, r_target, beta, max_iter=1000)
            final_score = _score(best_pts)
            if final_score > best_score:
                best_score = final_score
        except Exception:
            pass

    if best_pts is None:
        best_pts = _hex_lattice_init(0.125)

    return best_pts
