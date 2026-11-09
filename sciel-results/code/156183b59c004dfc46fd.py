import numpy as np
import time

class Optimizer:
    """
    Robust CMA-ES optimizer for BBOB f18 (Schaffers F7).
    Optimized for small dimensions (dim=5) and tight budgets (500 calls).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0
        
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

        # CMA-ES Parameters
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) + c_sigma)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim**2))
        
        # Restart parameters
        stall_counter = 0
        max_stall = 8
        
        last_mean_move = 0.0
        
        while calls < budget and time.time() < deadline:
            # Check for restart conditions
            if last_mean_move < 0.1 * sigma:
                stall_counter += 1
            else:
                stall_counter = 0
            
            if sigma > 100 or sigma < 1e-6:
                stall_counter = max_stall + 1

            if stall_counter >= max_stall:
                if best_x is not None:
                    x_mean = best_x.copy()
                else:
                    x_mean = np.zeros(dim)
                C = np.eye(dim)
                C_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 0.5 if sigma > 0.5 else sigma * 0.5
                stall_counter = 0
                last_mean_move = 0.0
                continue

            # Sample population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                C = np.eye(dim)
                C_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

            X = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                X[i] = x_mean + sigma * (L @ z)
            
            # Evaluate population
            F = np.array([eval_pt(x_i) for x_i in X])
            
            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update mean
            x_new_mean = np.sum(weights[:, None] * X[:mu], axis=0)
            last_mean_move = np.linalg.norm(x_new_mean - x_mean)
            
            # Standard CMA-ES update order:
            # 1. p_sigma update (using OLD C_inv)
            y_sigma = x_new_mean - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (sqrt_C_inv @ y_sigma) / sigma
            
            # 2. Sigma update
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e6)
            
            # 3. p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma / sigma
            
            # 4. C update (rank-1 + rank-mu)
            y = y_sigma
            diff = X[:mu] - x_mean
            rank_mu = np.einsum('i,ij,ik->jk', weights, diff, diff)
            C = (1 - c_c * (2 - c_c) - c_c * (2 - c_c - c_sigma * (1 - d_sigma))) * C \
                + c_c * (2 - c_c) * np.outer(y, y) \
                + c_c * (2 - c_c - c_sigma * (1 - d_sigma)) * rank_mu
            
            # 5. Update C_inv
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)
                C = np.eye(dim)
            
            # 6. Compute sqrt(C_inv) for next iteration's p_sigma update
            try:
                w, V = np.linalg.eigh(C_inv)
                w = np.clip(w, 0, None)
                sqrt_C_inv = V @ np.diag(np.sqrt(w)) @ V.T
            except np.linalg.LinAlgError:
                sqrt_C_inv = np.eye(dim)

        return best_x, best_f
