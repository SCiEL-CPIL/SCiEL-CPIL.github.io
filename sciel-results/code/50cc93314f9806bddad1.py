import numpy as np
import time
import math
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 300

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def remaining_time():
        return deadline - (time.time() - start_time)

    def is_finished():
        return (time.time() - start_time) > deadline

    def _compute_radius_fast(p):
        """Compute the minimum radius for a given set of points."""
        p = np.asarray(p)
        if p.shape != (N, 2):
            raise ValueError("Wrong shape")
        norms = np.linalg.norm(p, axis=1)
        min_bound = 1.0 - np.min(norms)
        if min_bound <= 0:
            return 0.0
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        return min(min_pair, min_bound)

    def _init_hex():
        """Initialize with hexagonal lattice."""
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points, radius = points[mask], radius[mask]
            if len(points) >= N:
                points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
        return points

    def _hertzian_dynamics(x0, n_iter=2000, step=0.1):
        """Simple Hertzian dynamics to spread points."""
        p = x0.copy()
        for i in range(n_iter):
            if is_finished():
                break
            # Compute forces
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            # Repulsive forces (inverse square)
            inv_dists = 1.0 / np.maximum(dists, 1e-6) ** 2
            np.fill_diagonal(inv_dists, 0.0)
            forces = np.einsum('ij,ijk->ik', inv_dists, diff)
            # Attractive force toward center (weak)
            forces -= 0.1 * p
            # Update
            p += step * forces
            # Keep inside unit circle
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            p = p * np.minimum(1.0, (1.0 - 0.05) / norms)
        return p

    def _compute_delaunay_edges(points):
        """Compute Delaunay triangulation edges."""
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            return list(edges)
        except Exception:
            return []

    def _spectral_regularizer(points, beta_spec=1.0):
        """
        Compute spectral regularizer: variance of Delaunay edge lengths.
        This encourages uniform spacing across the lattice.
        """
        edges = _compute_delaunay_edges(points)
        if len(edges) < 2:
            return 0.0, np.zeros_like(points)
        
        edge_lengths = np.array([np.linalg.norm(points[a] - points[b]) for a, b in edges])
        if len(edge_lengths) == 0:
            return 0.0, np.zeros_like(points)
        
        mean_len = np.mean(edge_lengths)
        var = np.var(edge_lengths)
        
        grad = np.zeros_like(points)
        for (a, b), L in zip(edges, edge_lengths):
            if L < 1e-10:
                continue
            coeff = (2.0 / len(edges)) * (L - mean_len) / L
            diff = points[a] - points[b]
            grad[a] += coeff * diff
            grad[b] -= coeff * diff
        
        return beta_spec * var, beta_spec * grad

    def _soft_min_optimize(x0, beta=1000, max_iter=500, use_spectral=False, beta_spec=0.0):
        """Optimize using soft-min formulation with optional spectral regularizer."""
        
        def get_points(x):
            return x.reshape(N, 2)

        def objective_and_grad(x):
            points = get_points(x)
            # Pairwise distances using efficient computation
            sq_norms = np.sum(points * points, axis=1)
            dot_prod = points @ points.T
            sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * dot_prod
            sq_dists = np.maximum(sq_dists, 0.0)
            dists = np.sqrt(sq_dists)
            iu = np.triu_indices(N, k=1)
            d_ij = dists[iu]
            # Boundary distances
            norms = np.sqrt(sq_norms)
            d_b = 1.0 - norms
            v = np.concatenate([d_ij / 2.0, d_b])
            # Soft-min
            v_min = np.min(v)
            exp_v = np.exp(-beta * (v - v_min))
            S = np.sum(exp_v)
            weights = exp_v / S
            softmin_val = v_min - (1.0 / beta) * np.log(S)
            # Gradient
            w_pairs = weights[:len(d_ij)]
            w_bound = weights[len(d_ij):]
            w_mat = np.zeros((N, N))
            w_mat[iu] = w_pairs
            w_sym = w_mat + w_mat.T
            d_safe = np.where(dists < 1e-9, 1e-9, dists)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :]
            factor = w_sym / (2.0 * d_safe)
            np.fill_diagonal(factor, 0.0)
            grad_points = np.einsum('ij,ijk->ik', factor, diff)
            norms_safe = np.where(norms < 1e-9, 1e-9, norms)
            grad_bound_points = - (w_bound[:, np.newaxis] * points) / norms_safe[:, np.newaxis]
            total_grad = grad_points + grad_bound_points
            
            if use_spectral and beta_spec > 0:
                spec_val, spec_grad = _spectral_regularizer(points, beta_spec)
                total_grad = total_grad + spec_grad
                obj_val = -softmin_val + spec_val
            else:
                obj_val = -softmin_val
            
            grad_val = -total_grad.flatten()
            return obj_val, grad_val

        try:
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return res.x
        except Exception:
            return x0

    def _polish(x0):
        """Final polish with high beta."""
        return _soft_min_optimize(x0, beta=50000, max_iter=300, use_spectral=True, beta_spec=0.5)

    def _random_perturb(x0, magnitude=0.01):
        """Random perturbation for restarts."""
        return x0 + np.random.randn(*x0.shape) * magnitude

    def _local_search(points, n_iter=200):
        """
        Local search: move points to increase minimum distance.
        Uses a greedy approach: for each point, try small moves in random directions.
        """
        p = points.copy()
        current_score = _compute_radius_fast(p)
        
        for i in range(n_iter):
            if is_finished():
                break
            
            idx = np.random.randint(N)
            original = p[idx].copy()
            
            best_move = None
            best_local_score = current_score
            
            for _ in range(20):
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                for step in [0.005, 0.01, 0.02, 0.05]:
                    p[idx] = original + step * direction
                    if np.linalg.norm(p[idx]) > 1.0:
                        p[idx] = original
                        continue
                    local_score = _compute_radius_fast(p)
                    if local_score > best_local_score:
                        best_local_score = local_score
                        best_move = p[idx].copy()
            
            if best_move is not None:
                p[idx] = best_move
                current_score = best_local_score
            else:
                p[idx] = original
        
        return p, current_score

    def _simulated_annealing(points, time_budget, initial_temp=0.02, cooling_rate=0.98, step_size=0.01):
        """
        Simulated Annealing to escape local minima.
        """
        sa_start = time.time()
        current_points = points.copy()
        current_score = _compute_radius_fast(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        T = initial_temp
        
        while time.time() - sa_start < time_budget:
            if is_finished():
                break
            
            num_moves = np.random.randint(5, 20)
            indices = np.random.choice(N, num_moves, replace=False)
            
            proposed_points = current_points.copy()
            noise = np.random.normal(0, step_size, size=(num_moves, 2))
            proposed_points[indices] += noise
            
            rad = np.linalg.norm(proposed_points, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                proposed_points[mask] /= rad[mask, np.newaxis]
            
            proposed_score = _compute_radius_fast(proposed_points)
            
            delta = proposed_score - current_score
            if delta > 0:
                current_points = proposed_points
                current_score = proposed_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current_points = proposed_points
                    current_score = proposed_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
            
            T *= cooling_rate
            if T < 1e-5:
                T = 1e-5
                
        return best_points, best_score

    # Main optimization loop
    best_x = None
    best_score = 0.0

    # Step 1: Hexagonal initialization
    hex_points = _init_hex()
    score = _compute_radius_fast(hex_points)
    if score > best_score:
        best_score = score
        best_x = hex_points.copy()

    if is_finished():
        return best_x

    # Step 2: Hertzian dynamics to explore
    if remaining_time() > 30:
        hert_points = _hertzian_dynamics(hex_points, n_iter=1000, step=0.05)
        score = _compute_radius_fast(hert_points)
        if score > best_score:
            best_score = score
            best_x = hert_points.copy()

    if is_finished():
        return best_x

    # Step 3: Simulated annealing to explore
    sa_budget = min(30.0, remaining_time() - 10.0)
    if sa_budget > 5.0:
        sa_points, sa_score = _simulated_annealing(best_x, sa_budget)
        if sa_score > best_score:
            best_score = sa_score
            best_x = sa_points.copy()

    if is_finished():
        return best_x

    # Step 4: Soft-min optimization with annealing over beta
    betas = [10, 50, 200, 1000, 5000]
    for beta in betas:
        if is_finished() or remaining_time() < 15.0:
            break
        x_opt = _soft_min_optimize(best_x.flatten(), beta=beta, max_iter=300)
        x_opt = x_opt.reshape(N, 2)
        # Project inside
        rad = np.linalg.norm(x_opt, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x_opt[mask] /= rad[mask, np.newaxis]
        sc = _compute_radius_fast(x_opt)
        if sc > best_score:
            best_score = sc
            best_x = x_opt.copy()

    if is_finished():
        return best_x

    # Step 5: Local search
    ls_budget = min(15.0, remaining_time() - 10.0)
    if ls_budget > 5.0:
        n_iter = int(150 * (ls_budget / 15.0))
        ls_points, ls_score = _local_search(best_x, n_iter=n_iter)
        if ls_score > best_score:
            best_score = ls_score
            best_x = ls_points.copy()

    if is_finished():
        return best_x

    # Step 6: Random restarts with polish
    restart_budget = remaining_time() - 10.0
    num_restarts = max(1, int(restart_budget / 15.0))
    for r in range(num_restarts):
        if is_finished() or remaining_time() < 10.0:
            break
        perturbed = _random_perturb(best_x, magnitude=0.015 * (1 + r * 0.2))
        rad = np.linalg.norm(perturbed, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            perturbed[mask] /= rad[mask, np.newaxis]
        
        x_opt = _soft_min_optimize(perturbed.flatten(), beta=2000, max_iter=200)
        x_opt = x_opt.reshape(N, 2)
        rad = np.linalg.norm(x_opt, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x_opt[mask] /= rad[mask, np.newaxis]
        sc = _compute_radius_fast(x_opt)
        if sc > best_score:
            best_score = sc
            best_x = x_opt.copy()

    if is_finished():
        return best_x

    # Step 7: Final polish
    final_x = _polish(best_x.flatten())
    final_x = final_x.reshape(N, 2)
    rad = np.linalg.norm(final_x, axis=1)
    mask = rad > 1.0
    if np.any(mask):
        final_x[mask] /= rad[mask, np.newaxis]
    final_score = _compute_radius_fast(final_x)
    if final_score > best_score:
        best_score = final_score
        best_x = final_x.copy()

    # Final validation
    rad_final = np.linalg.norm(best_x, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_x)):
        return np.zeros((N, 2))
        
    return best_x
