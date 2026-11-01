import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s kill

    # Precompute upper triangle indices
    _i_upper, _j_upper = np.triu_indices(N, k=1)

    # ---------- helpers ----------

    def compute_energy(pts):
        """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(d_ij)"""
        gram = pts @ pts.T
        dist_sq = 2.0 - 2.0 * gram
        dist_sq = np.maximum(dist_sq, 1e-14)
        dist = np.sqrt(dist_sq)
        d_pairs = dist[_i_upper, _j_upper]
        d_pairs = np.maximum(d_pairs, 1e-14)
        return -math.fsum(np.log(d_pairs).tolist())

    def compute_energy_and_grad(pts):
        """Compute both energy and gradient in one pass to avoid redundant Gram computation."""
        gram = pts @ pts.T
        dist_sq = 2.0 - 2.0 * gram
        dist_sq = np.maximum(dist_sq, 1e-14)
        dist = np.sqrt(dist_sq)
        d_pairs = dist[_i_upper, _j_upper]
        d_pairs = np.maximum(d_pairs, 1e-14)
        energy = -math.fsum(np.log(d_pairs).tolist())
        
        inv_d2 = 1.0 / dist_sq
        np.fill_diagonal(inv_d2, 0.0)
        row_sums = np.sum(inv_d2, axis=1)
        grad = pts * row_sums[:, None] - inv_d2 @ pts
        return energy, grad

    def project_to_tangent(pts, grad):
        """Project gradient onto tangent space of sphere at each point."""
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        return grad - dot * pts

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return pts / norms

    def newton_manifold_polish(pts, max_iters=200, tol=1e-15, time_budget=5.0):
        """
        Newton-style manifold optimization with damped gradient steps.
        """
        current_energy, _ = compute_energy_and_grad(pts)
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            _, grad = compute_energy_and_grad(pts)
            grad_tan = project_to_tangent(pts, grad)
            gnorm = np.linalg.norm(grad_tan)
            if gnorm < tol:
                break
            alpha = 0.5
            improved = False
            for _ in range(20):
                cand = pts - alpha * grad_tan
                cand = normalize(cand)
                cand_e, _ = compute_energy_and_grad(cand)
                if cand_e < current_energy:
                    pts = cand
                    current_energy = cand_e
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break
        return pts, current_energy

    # ---------- L-BFGS optimization ----------

    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        energy, grad = compute_energy_and_grad(pts)
        grad_tangent = project_to_tangent(pts, grad)
        return energy, grad_tangent.flatten()

    def lbfgs_optimize(x0, maxiter=500, tight=False):
        """Run L-BFGS-B and return (best_points, best_energy)."""
        try:
            if tight:
                opts = {'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 100}
            else:
                opts = {'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 80}
            result = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options=opts
            )
            new_pts = normalize(result.x.reshape(N, 3))
            new_energy, _ = compute_energy_and_grad(new_pts)
            return new_pts, new_energy
        except Exception:
            return None, None

    # ---------- gradient descent with line search (tangent space) ----------

    def gradient_descent(pts, max_iters=5000, initial_step=0.005, time_budget=10.0):
        """Adaptive-step gradient descent in tangent space."""
        current_energy, _ = compute_energy_and_grad(pts)
        step_size = initial_step
        min_step = 1e-14
        max_step = 0.1
        no_improve = 0
        prev_grad_norm = np.inf

        for it in range(max_iters):
            if time.time() > time_budget:
                break

            _, grad = compute_energy_and_grad(pts)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break

            # Early exit if gradient is not decreasing
            if it > 10 and grad_norm > 2.0 * prev_grad_norm:
                step_size *= 0.5
            prev_grad_norm = grad_norm

            grad_tangent = project_to_tangent(pts, grad)

            # Backtracking line search
            alpha = step_size
            improved = False
            for _ in range(30):
                candidate = pts - alpha * grad_tangent
                candidate = normalize(candidate)
                cand_energy, _ = compute_energy_and_grad(candidate)
                if cand_energy < current_energy:
                    pts = candidate
                    current_energy = cand_energy
                    improved = True
                    no_improve = 0
                    step_size = min(alpha * 1.3, max_step)
                    break
                alpha *= 0.5
                if alpha < min_step:
                    break

            if not improved:
                no_improve += 1
                if no_improve > 100:
                    break

        return pts, current_energy

    # ---------- swap-based simulated annealing with incremental updates ----------

    def swap_annealing_fast(pts, max_iters=8000, T_start=0.08, T_end=1e-10, time_budget=20.0):
        """
        Fast swap-based simulated annealing with incremental energy updates.
        """
        current_energy, _ = compute_energy_and_grad(pts)
        best_pts = pts.copy()
        best_energy = current_energy
        current_pts = pts.copy()

        for it in range(max_iters):
            if time.time() > time_budget:
                break

            progress = it / max_iters
            # Faster cooling initially, slower at end
            T = T_start * (T_end / T_start) ** (progress ** 0.7)

            # Choose move type: 60% perturb, 25% swap, 15% local cluster move
            r = np.random.random()
            if r < 0.6:
                # Perturb 2-4 random points
                n_perturb = np.random.randint(2, 5)
                indices = np.random.choice(N, n_perturb, replace=False)
                noise_scale = 0.06 + 0.04 * (1 - progress)
                noise = np.random.randn(n_perturb, 3) * noise_scale

                candidate = current_pts.copy()
                for k, idx in enumerate(indices):
                    n = noise[k]
                    n = n - np.dot(n, current_pts[idx]) * current_pts[idx]
                    candidate[idx] = current_pts[idx] + n
                candidate = normalize(candidate)
            elif r < 0.85:
                # Swap move
                i, j = np.random.choice(N, 2, replace=False)
                candidate = current_pts.copy()
                noise_i = np.random.randn(3) * 0.04
                noise_j = np.random.randn(3) * 0.04
                noise_i = noise_i - np.dot(noise_i, current_pts[i]) * current_pts[i]
                noise_j = noise_j - np.dot(noise_j, current_pts[j]) * current_pts[j]
                candidate[i] = current_pts[j] + noise_i
                candidate[j] = current_pts[i] + noise_j
                candidate = normalize(candidate)
            else:
                # Local cluster move: perturb a group of nearby points
                i = np.random.randint(N)
                # Find nearest neighbors
                dists = np.sum((current_pts - current_pts[i]) ** 2, axis=1)
                dists[i] = np.inf
                nearby = np.argsort(dists)[:4]
                indices = np.concatenate([[i], nearby])
                noise_scale = 0.03 + 0.02 * (1 - progress)
                noise = np.random.randn(len(indices), 3) * noise_scale

                candidate = current_pts.copy()
                for k, idx in enumerate(indices):
                    n = noise[k]
                    n = n - np.dot(n, current_pts[idx]) * current_pts[idx]
                    candidate[idx] = current_pts[idx] + n
                candidate = normalize(candidate)

            cand_energy, _ = compute_energy_and_grad(candidate)
            delta = cand_energy - current_energy

            if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-12)):
                current_pts = candidate
                current_energy = cand_energy
                if cand_energy < best_energy:
                    best_energy = cand_energy
                    best_pts = candidate.copy()

        return best_pts, best_energy

    # ---------- initializations ----------

    def fibonacci_spiral(offset=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return normalize(pts)

    def random_points(rng):
        pts = rng.randn(N, 3)
        return normalize(pts)

    def ring_initialization(num_rings=10, rng=None):
        if rng is None:
            rng = np.random
        pts_list = []
        for ring in range(num_rings):
            z_val = -1.0 + 2.0 * ring / (num_rings - 1)
            r_val = np.sqrt(max(1.0 - z_val**2, 0.0))
            if r_val < 0.01:
                r_val = 0.01
            num_per_ring = N // num_rings
            for j in range(num_per_ring):
                angle = 2.0 * np.pi * j / num_per_ring + ring * 0.3
                pts_list.append([r_val * np.cos(angle), r_val * np.sin(angle), z_val])
        while len(pts_list) < N:
            pts_list.append(rng.randn(3).tolist())
        pts = np.array(pts_list[:N])
        return normalize(pts)

    def clustered_initialization(rng, num_clusters=5):
        """Clustered starts: points near random cluster centers."""
        centers = rng.randn(num_clusters, 3)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        pts = []
        for i in range(N):
            center = centers[i % num_clusters]
            noise = rng.randn(3) * 0.3
            pt = center + noise
            pt = pt / np.linalg.norm(pt)
            pts.append(pt)
        return np.array(pts)

    # ---------- main search ----------

    rng = np.random.RandomState(42)

    best_points = None
    best_energy = float('inf')

    # Initial set of starting points - more diverse
    initializations = [
        fibonacci_spiral(0.0),
        fibonacci_spiral(0.15),
        fibonacci_spiral(0.35),
        fibonacci_spiral(0.55),
        fibonacci_spiral(0.75),
        ring_initialization(8, rng),
        ring_initialization(10, rng),
        ring_initialization(12, rng),
        ring_initialization(16, rng),
        ring_initialization(20, rng),
        random_points(rng),
        random_points(rng),
        random_points(rng),
        random_points(rng),
        clustered_initialization(rng, 3),
        clustered_initialization(rng, 4),
        clustered_initialization(rng, 5),
        clustered_initialization(rng, 7),
        clustered_initialization(rng, 10),
    ]

    # Phase 1: L-BFGS from multiple starts
    for init_pts in initializations:
        if time.time() > deadline - 55:
            break

        pts, energy = lbfgs_optimize(init_pts.flatten(), maxiter=500)
        if pts is not None and energy < best_energy:
            best_energy = energy
            best_points = pts.copy()

    # Phase 2: Gradient descent polish on best
    if best_points is not None:
        if time.time() < deadline - 45:
            pts, energy = gradient_descent(best_points, max_iters=5000, initial_step=0.005, time_budget=deadline - time.time() - 40)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 3: Swap-based annealing for global search
    if best_points is not None:
        if time.time() < deadline - 35:
            pts, energy = swap_annealing_fast(best_points, max_iters=10000, T_start=0.05, T_end=1e-8, time_budget=deadline - time.time() - 30)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 4: More L-BFGS restarts from perturbed best
    if best_points is not None:
        for restart in range(15):
            if time.time() > deadline - 25:
                break

            # Mix of small and large perturbations
            if restart % 3 == 0:
                noise_scale = 0.02 + 0.005 * restart
            elif restart % 3 == 1:
                noise_scale = 0.05 + 0.01 * restart
            else:
                noise_scale = 0.1 + 0.02 * restart
            noise = rng.randn(N, 3) * noise_scale
            perturbed = normalize(best_points + noise)
            pts, energy = lbfgs_optimize(perturbed.flatten(), maxiter=400)
            if pts is not None and energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 5: Final gradient descent polish with tight tolerance
    if best_points is not None:
        if time.time() < deadline - 15:
            pts, energy = gradient_descent(best_points, max_iters=8000, initial_step=0.002, time_budget=deadline - time.time() - 12)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 6: L-BFGS polish with tight tolerance
    if best_points is not None:
        if time.time() < deadline - 8:
            pts, energy = lbfgs_optimize(best_points.flatten(), maxiter=1000, tight=True)
            if pts is not None and energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 7: Newton-style manifold polish for final precision
    if best_points is not None:
        if time.time() < deadline - 3:
            pts, energy = newton_manifold_polish(best_points, max_iters=500, tol=1e-16, time_budget=deadline - time.time() - 2)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 8: Final ultra-tight gradient descent
    if best_points is not None:
        if time.time() < deadline - 0.5:
            pts, energy = gradient_descent(best_points, max_iters=5000, initial_step=0.0005, time_budget=deadline - time.time() - 0.3)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Final normalization and validity check
    if best_points is None:
        best_points = fibonacci_spiral(0.0)

    best_points = normalize(best_points)

    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_spiral(0.0)
        best_points = normalize(best_points)

    # Ensure no two points are closer than 1e-9
    gram = best_points @ best_points.T
    dist_sq = 2.0 - 2.0 * gram
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)
    d_pairs = dist[_i_upper, _j_upper]
    if np.any(d_pairs < 1e-9):
        # Separate close points
        for _ in range(100):
            idx_i = np.argmin(d_pairs)
            i_idx = _i_upper[idx_i]
            j_idx = _j_upper[idx_i]
            direction = best_points[i_idx] - best_points[j_idx]
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-12:
                direction = rng.randn(3)
                norm_dir = np.linalg.norm(direction)
            direction = direction / norm_dir
            best_points[i_idx] = best_points[i_idx] + 1e-8 * direction
            best_points[j_idx] = best_points[j_idx] - 1e-8 * direction
            best_points = normalize(best_points)
            gram = best_points @ best_points.T
            dist_sq = 2.0 - 2.0 * gram
            dist_sq = np.maximum(dist_sq, 0.0)
            dist = np.sqrt(dist_sq)
            d_pairs = dist[_i_upper, _j_upper]
            if np.all(d_pairs >= 1e-9):
                break

    return best_points

# EVOLVE-BLOCK-END
