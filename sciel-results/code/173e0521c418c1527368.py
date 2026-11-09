# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def eval_pt(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of best points to use for mean update
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial settings
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Pre-compute some factors for efficiency? For dim=5, direct inverse is fine.
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rules
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) + c_sigma)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim**2))
        
        best_f_restart = np.inf
        no_improve_count = 0
        restart_count = 0
        
        # Main loop
        while calls < budget and time.time() < deadline:
            # Check for restart
            if no_improve_count > 15 or sigma < 1e-8 or sigma > 1e3:
                # Restart strategy
                if restart_count == 0:
                    # First restart: try a random point far from current
                    x_mean = np.random.uniform(-4.0, 4.0, dim)
                    sigma = 2.0
                else:
                    # Subsequent restarts: perturb best_x
                    if best_x is not None:
                        x_mean = best_x + np.random.normal(0, 1.0, dim)
                        x_mean = np.clip(x_mean, -5.0, 5.0)
                    else:
                        x_mean = np.zeros(dim)
                    sigma = 1.0
                C = np.eye(dim)
                C_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                restart_count += 1
                continue

            # Sample population
            X = []
            for i in range(lambda_):
                # Sample from N(0, C)
                # Use Cholesky decomposition for C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if Cholesky fails
                    L = np.eye(dim)
                z = np.random.standard_normal(dim)
                x_i = x_mean + sigma * (L @ z)
                X.append(x_i)
            
            # Evaluate population
            F = np.array([eval_pt(x_i) for x_i in X])
            
            # Sort by fitness
            order = np.argsort(F)
            X = [X[i] for i in order]
            F = F[order]
            
            # Update best
            if F[0] < best_f_restart - 1e-12:
                best_f_restart = F[0]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update mean
            x_new_mean = np.sum(weights[:, None] * np.array(X[:mu]), axis=0)
            
            # Update covariance
            # Calculate diff
            diff = np.array(X[:mu]) - x_mean
            # Rank-1 update
            C = (1 - c_c * (2 - c_c)) * C + c_c * (2 - c_c) * (x_new_mean - x_mean)[:, None] * (x_new_mean - x_mean)[None, :]
            
            # Rank-mu update
            # w_i * diff_i * diff_i^T
            # weights is 1D, diff is (mu, dim)
            rank_mu_update = np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            C = C + c_c * (2 - c_c - c_sigma * (1 - d_sigma)) * rank_mu_update
            
            # Update paths
            C_inv = np.linalg.inv(C)  # For dim=5, this is fast
            sigma_old = sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # p_sigma update
            # Need sqrt(C_inv)
            # Use eigendecomposition for sqrt
            w, V = np.linalg.eigh(C_inv)
            # Ensure positive
            w = np.clip(w, 0, None)
            sqrt_C_inv = V @ np.diag(np.sqrt(w)) @ V.T
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt_C_inv @ (x_new_mean - x_mean) / sigma_old
            
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma_old
            
            # Update mean
            x_mean = x_new_mean
            
            # Clip mean to box? Not strictly necessary for CMA-ES, but good for stability.
            # Actually, CMA-ES handles bounds via clipping in eval_pt, but the mean should stay reasonable.
            x_mean = np.clip(x_mean, -5.0, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
