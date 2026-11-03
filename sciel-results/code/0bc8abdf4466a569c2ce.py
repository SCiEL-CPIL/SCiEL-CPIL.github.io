import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit

    def check_time():
        return time.time() - start_time

    def remaining_time():
        return deadline - (time.time() - start_time)

    def is_finished():
        return (time.time() - start_time) > deadline

    def _compute_radius_fast(p):
        """Compute the minimum radius for a given set of points."""
        p = np.asarray(p)
        if p.shape != (N, 2):
            raise ValueError("Wrong shape")
        # Boundary constraint
        norms = np.linalg.norm(p, axis=1)
        min_bound = 1.0 - np.min(norms)
        if min_bound <= 0:
            return 0.0
        # Pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
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
            diff = p[:, np.newaxis, :] - p[np.newaxis, :]
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

    def _soft_min_optimize(x0, beta=1000, max_iter=1000):
        """Optimize using soft-min formulation."""
        def get_points(x):
            return x.reshape(N, 2)

        def objective_and_grad(x):
            points = get_points(x)
            # Pairwise distances
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
            total_grad = (grad_points + grad_bound_points).flatten()
            obj_val = -softmin_val
            grad_val = -total_grad
            return obj_val, grad_val

        try:
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': max_iter * 2}
            )
            return res.x
        except Exception:
            return x0

    def _polish(x0):
        """Final polish with high beta."""
        return _soft_min_optimize(x0, beta=50000, max_iter=200)

    def _random_perturb(x0, magnitude=0.01):
        """Random perturbation for restarts."""
        return x0 + np.random.randn(*x0.shape) * magnitude

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

    # Step 2: Hertzian dynamics
    if remaining_time() > 30:
        hert_points = _hertzian_dynamics(hex_points, n_iter=1000, step=0.05)
        score = _compute_radius_fast(hert_points)
        if score > best_score:
            best_score = score
            best_x = hert_points.copy()

    if is_finished():
        return best_x

    # Step 3: Soft-min optimization with annealing
    betas = [10, 50, 200, 1000, 5000]
    current_x = best_x.flatten()

    for beta in betas:
        if is_finished() or remaining_time() < 15:
            break
        max_iter = min(500, int(remaining_time() * 3))
        if max_iter < 50:
            break
        new_x = _soft_min_optimize(current_x, beta=beta, max_iter=max_iter)
        score = _compute_radius_fast(new_x.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = new_x.reshape(N, 2).copy()
            current_x = new_x
        elif is_finished():
            break

    if is_finished():
        return best_x

    # Step 4: Random restarts if time remains
    n_restarts = 3
    for i in range(n_restarts):
        if is_finished() or remaining_time() < 20:
            break
        perturbed = _random_perturb(best_x, magnitude=0.05)
        new_x = _soft_min_optimize(perturbed.flatten(), beta=1000, max_iter=200)
        score = _compute_radius_fast(new_x.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = new_x.reshape(N, 2).copy()

    # Step 5: Final polish
    if remaining_time() > 10:
        polished = _polish(best_x.flatten())
        score = _compute_radius_fast(polished.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = polished.reshape(N, 2).copy()

    return best_x.reshape(N, 2)
