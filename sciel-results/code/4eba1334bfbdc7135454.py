"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Time guard
        start_time = time.time()
        deadline = start_time + 320.0

        # Initial parameters
        # Start near center, but with some spread
        x = np.zeros(dim)
        sigma = 2.0
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, lambda_ + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w[:mu])
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial evaluation
        x0 = np.zeros(dim)
        fx0 = func(x0)
        best_x = x0.copy()
        best_fx = fx0
        
        # Population
        pop_x = []
        pop_f = []
        
        # Initial population evaluation
        for i in range(lambda_):
            y = x + sigma * (np.linalg.cholesky(C) @ rng.standard_normal(dim))
            y = np.clip(y, -5, 5)
            fy = func(y)
            pop_x.append(y)
            pop_f.append(fy)
            if fy < best_fx:
                best_x, best_fx = y.copy(), fy

        calls = lambda_
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort
            order = np.argsort(pop_f)
            x_k = [pop_x[i] for i in order[:mu]]
            f_k = [pop_f[i] for i in order[:mu]]
            
            # Update mean
            x_new = np.sum(np.array(x_k) * w[:, None], axis=0)
            
            # Update step size control variables
            xbar_old = x
            x = x_new
            
            # Normalize
            norm_x = np.linalg.norm(x - xbar_old)
            if norm_x > 1e-30:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_N * (x - xbar_old) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls / lambda_ + 1)))) < (1.4 + 2 / (dim + 1)) * chi_N
            else:
                hsig = True
                
            # Update covariance
            if hsig:
                pc = (1 - c_c) * pc + c_c * (x - xbar_old) / sigma
            
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + delta_hsig) \
                + c_mu / mueff * np.sum([np.outer(x_k[i] - xbar_old, x_k[i] - xbar_old) / sigma**2 for i in range(mu)], axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chi_N - 1))
            
            # Ensure symmetry and positive definiteness (simple fix)
            C = (C + C.T) / 2
            # Re-cholesky for next step, handle failure
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset to identity with smaller sigma or just use identity
                C = np.eye(dim)
                L = np.eye(dim)
                sigma = max(sigma * 0.5, 0.1)
            
            # Generate new population
            new_pop_x = []
            new_pop_f = []
            for i in range(lambda_):
                y = x + sigma * (L @ rng.standard_normal(dim))
                y = np.clip(y, -5, 5)
                fy = func(y)
                new_pop_x.append(y)
                new_pop_f.append(fy)
                calls += 1
                if fy < best_fx:
                    best_x, best_fx = y.copy(), fy
                    
            pop_x = new_pop_x
            pop_f = new_pop_f
            
            # Check for convergence or time
            if sigma < 1e-10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
