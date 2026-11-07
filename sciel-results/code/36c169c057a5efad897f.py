"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size: 30 for dim=5
        lam = max(4 + int(3 * np.log(dim)), 30)
        
        # Initial mean and sigma
        x_mean = np.zeros(dim)
        sigma = 2.0  # Start with reasonable step size for [-5,5]^5
        
        # CMA-ES parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best solution
        best_x = x_mean.copy()
        best_f = float('inf')
        
        used_budget = 0
        n_eval = 0
        
        def evaluate(x):
            nonlocal used_budget, n_eval
            x = np.clip(x, -5, 5)
            f = func(x)
            used_budget += 1
            n_eval += 1
            return x, f
        
        def update_decompositions():
            nonlocal C, C_inv, C_sqrt, C_inv_sqrt
            # Eigen decomposition for sqrt and inverse sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        update_decompositions()
        
        while used_budget < budget and time.time() < deadline:
            # Generate population
            population = np.zeros((lam, dim))
            fvals = np.zeros(lam)
            
            for i in range(lam):
                if used_budget >= budget or time.time() > deadline:
                    break
                # Sample from multivariate normal using Cholesky-like decomposition
                z = rng.standard_normal(dim)
                # Use C_inv_sqrt for sampling: x = mean + sigma * C_inv_sqrt @ z
                # Actually, we want to sample from N(mean, sigma^2 * C)
                # So x = mean + sigma * C_sqrt @ z
                x_i = x_mean + sigma * (C_sqrt @ z)
                x_i, f_i = evaluate(x_i)
                population[i] = x_i
                fvals[i] = f_i
            
            if n_eval == 0:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            population = population[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = population[0].copy()
            
            # Select top mu individuals
            x_selected = population[:mu]
            f_selected = fvals[:mu]
            
            # Weighted mean
            x_new = np.sum(weights[:, np.newaxis] * x_selected, axis=0)
            
            # Update sigma (CSA)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ ((x_new - x_mean) / sigma)
            
            hsig = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (n_eval + 1) / lam)) <
                          (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))) else 0.0
            
            # Update p_c
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * C_inv_sqrt @ ((x_new - x_mean) / sigma)
            
            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(x_selected - x_mean, x_selected - x_mean) / sigma**2, axis=0)
            
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Regularize C to ensure positive definiteness
            C = (C + C.T) / 2
            eigvals_C = np.linalg.eigvalsh(C)
            if np.min(eigvals_C) < 1e-12:
                C += 1e-12 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(cs / (5 * dim) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10)
            
            # Update mean
            x_mean = x_new
            
            # Update decompositions periodically (every 10 generations or so)
            if n_eval % (10 * lam) < lam:
                update_decompositions()
            
            # Adaptive stopping: if sigma is very small or progress is minimal
            if sigma < 1e-12:
                break
            if n_eval > budget * 0.8:
                # Use remaining budget for local refinement around best
                remaining = budget - used_budget
                if remaining > 5 and time.time() < deadline:
                    # Simple local search with small steps
                    step_size = max(sigma * 0.1, 1e-6)
                    for _ in range(min(remaining, 20)):
                        if used_budget >= budget or time.time() > deadline:
                            break
                        for d in range(dim):
                            if used_budget >= budget or time.time() > deadline:
                                break
                            for direction in [-1, 1]:
                                if used_budget >= budget or time.time() > deadline:
                                    break
                                x_trial = best_x.copy()
                                x_trial[d] += direction * step_size
                                x_trial = np.clip(x_trial, -5, 5)
                                f_trial = func(x_trial)
                                used_budget += 1
                                if f_trial < best_f:
                                    best_f = f_trial
                                    best_x = x_trial.copy()
                            if used_budget >= budget or time.time() > deadline:
                                break
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
