"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False

class Optimizer:
    """Gaussian Process Optimizer for Büche-Rastrigin function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State
        X_obs = []
        Y_obs = []
        
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        # Helper to update best
        def update_best(x, f):
            nonlocal global_best_x, global_best_fx
            if f < global_best_fx:
                global_best_fx = f
                global_best_x = x.copy()

        # 1. Initial Exploration: 10 random points
        n_init = 10
        for _ in range(n_init):
            x = rng.uniform(-5, 5, size=dim)
            f = func(x)
            calls_made += 1
            X_obs.append(x)
            Y_obs.append(f)
            update_best(x, f)
            
            if calls_made >= budget:
                break

        if calls_made >= budget:
            return global_best_x, global_best_fx

        # GP Parameters
        # RBF Kernel: k(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
        # Fixed hyperparameters for speed and stability
        l = 1.0
        sigma_f = 1.0
        epsilon = 1e-6

        def rbf_kernel(X1, X2):
            """Compute RBF kernel matrix between X1 and X2."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / (l**2))

        def compute_posterior_fast(X_new, K_inv, Y_obs_np, X_obs_np):
            """
            Compute GP posterior mean and std for X_new.
            Assumes K_inv is the inverse of the kernel matrix for X_obs.
            K_inv: (N, N)
            Y_obs_np: (N,)
            X_obs_np: (N, D)
            X_new: (M, D)
            """
            # K_s: (M, N)
            K_s = rbf_kernel(X_new, X_obs_np)
            
            # mu: (M,)
            # mu = K_s @ (K_inv @ Y)
            alpha = K_inv @ Y_obs_np
            mu = K_s @ alpha
            
            # var: (M,)
            # var = k(x_new, x_new) - K_s @ K_inv @ K_s.T
            # k(x_new, x_new) is sigma_f^2 for RBF (diagonal)
            k_new_new = np.full(X_new.shape[0], sigma_f**2)
            
            # v = K_s @ K_inv : (M, N) @ (N, N) = (M, N)
            v = K_s @ K_inv
            # var = k_new_new - sum(v * K_s, axis=1)
            var = k_new_new - np.sum(v * K_s, axis=1)
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            
            return mu, std

        def expected_improvement(mu, std, best_f, xi=0.01):
            """Compute Expected Improvement acquisition function."""
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (best_f - mu) / std
                ei = (best_f - mu) * np.maximum(Z, 0) + std * (Z * np.maximum(Z, 0) - np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi))
                ei = np.where(std < 1e-8, 0.0, ei)
            return ei

        def acquire_next_point(K_inv, X_obs_np, Y_obs_np):
            """Find the next point to evaluate using EI."""
            best_f = global_best_fx

            # 1. Generate a grid of candidate points
            n_candidates = 200
            candidates = rng.uniform(-5, 5, size=(n_candidates, dim))
            
            # Add perturbations around current best
            if global_best_x is not None:
                for scale in [0.5, 0.1, 0.01]:
                    for _ in range(20):
                        p = global_best_x + scale * rng.standard_normal(dim)
                        p = np.clip(p, -5, 5)
                        candidates = np.vstack([candidates, p])
            
            # Compute EI for all candidates
            mu, std = compute_posterior_fast(candidates, K_inv, Y_obs_np, X_obs_np)
            ei = expected_improvement(mu, std, best_f)
            
            # Find top 5 candidates
            top_idx = np.argsort(ei)[-5:]
            top_candidates = candidates[top_idx]
            
            best_ei_val = -np.inf
            best_next_x = top_candidates[0]
            
            # Refine top candidates using local optimization
            if HAS_SCIPY_OPT:
                for x0 in top_candidates:
                    def neg_ei(x):
                        x = np.clip(x, -5, 5)
                        mu_x, std_x = compute_posterior_fast(x.reshape(1, -1), K_inv, Y_obs_np, X_obs_np)
                        ei_x = expected_improvement(mu_x, std_x, best_f)
                        return -ei_x[0]
                    
                    try:
                        res = minimize(neg_ei, x0, method='L-BFGS-B', bounds=[(-5, 5)]*dim, options={'maxiter': 10})
                        if res.success:
                            x_opt = np.clip(res.x, -5, 5)
                            mu_opt, std_opt = compute_posterior_fast(x_opt.reshape(1, -1), K_inv, Y_obs_np, X_obs_np)
                            ei_opt = expected_improvement(mu_opt, std_opt, best_f)[0]
                            if ei_opt > best_ei_val:
                                best_ei_val = ei_opt
                                best_next_x = x_opt
                    except Exception:
                        pass
            else:
                best_idx = np.argmax(ei)
                best_next_x = candidates[best_idx]

            return best_next_x

        # 2. Main Loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Prepare data for GP
            X_obs_np = np.array(X_obs)
            Y_obs_np = np.array(Y_obs)
            
            # Compute Kernel Matrix and its Inverse
            K = rbf_kernel(X_obs_np, X_obs_np)
            K += epsilon * np.eye(len(X_obs_np))
            
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)

            x_next = acquire_next_point(K_inv, X_obs_np, Y_obs_np)
            f_next = func(x_next)
            calls_made += 1
            
            X_obs.append(x_next)
            Y_obs.append(f_next)
            update_best(x_next, f_next)

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
