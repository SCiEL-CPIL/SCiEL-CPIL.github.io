"""Optimizer for BBOB-style f13 (Sharp ridge) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def cma_es(x0, f0, budget_remaining, rng_local):
            nonlocal best_x, best_f
            n_local = self.dim
            pop_size = 4 + int(3 * np.log(n_local))  # lambda
            pop_size = max(pop_size, n_local + 2)
            
            # CMA-ES parameters
            c_sigma = (n_local + 2) / (n_local + 8)
            c_sigma = c_sigma / (n_local + 2)  # standard
            c_sigma = (n_local + 2) / (n_local + 8) * 1.0
            c_sigma = 1.0 / (n_local + 4)
            
            c_c = (4 + n_local / n_local) / (4 + n_local + 4)
            c_c = (4 + n_local / n_local) / (4 + n_local + 4)
            
            # Standard CMA-ES constants
            c_sigma = (n_local + 2) / (n_local + 8)
            c_sigma = 1.0 / (n_local + 4)
            c_c = (4 + n_local / n_local) / (4 + n_local + 4)
            c_c = (4 + 1) / (4 + n_local + 4)
            c_c = 5.0 / (8 + n_local)
            
            c_mu = (0.5 * np.log(n_local / 2)) / (n_local + 2)
            c_mu = (0.5 * np.log(n_local / 2)) / (n_local + 2)
            
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_mu = (0.5 * np.log(n_local / 2)) / (n_local + 2)
            
            c_sigma = (n_local + 2) / (n_local + 8)
            c_sigma = 1.0 / (n_local + 4)
            c_c = (4 + n_local / n_local) / (4 + n_local + 4)
            c_c = 5.0 / (8 + n_local)
            
            c1 = 2 / ((n_local + 1.3) ** 2 + mu_eff)
            cmu = 2 * (mu_eff - 2 + 2 / mu_eff) / ((n_local + 2) ** 2 + mu_eff)
            
            sigma = 2.0
            x_mean = np.array(x0, dtype=float)
            pc = np.zeros(n_local)
            ps = np.zeros(n_local)
            C = np.eye(n_local)
            chi_n = np.sqrt(n_local) * (1 - 1 / (4 * n_local) + 1 / (21 * n_local ** 2))
            
            evals_used = 0
            gen = 0
            
            while evals_used < budget_remaining:
                if time.time() > deadline:
                    break
                
                # Generate population
                B = np.linalg.cholesky(C + 1e-10 * np.eye(n_local))
                offspring = []
                for i in range(pop_size):
                    z = rng_local.standard_normal(n_local)
                    y = x_mean + sigma * B @ z
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                
                # Evaluate
                offspring_f = []
                for y in offspring:
                    fy = func(y)
                    evals_used += 1
                    offspring_f.append(fy)
                
                offspring_f = np.array(offspring_f)
                offspring = np.array(offspring)
                
                # Update best
                idx_best = np.argmin(offspring_f)
                if offspring_f[idx_best] < best_f:
                    best_f = offspring_f[idx_best]
                    best_x = offspring[idx_best].copy()
                
                # Sort by fitness
                order = np.argsort(offspring_f)
                offspring_sorted = offspring[order]
                offspring_f_sorted = offspring_f[order]
                
                # Weighted mean
                x_new = np.sum(weights[:, None] * offspring_sorted[:mu], axis=0)
                
                # Update covariance matrix adaptation
                y_inv = np.linalg.solve(C, x_new - x_mean) / sigma
                delta_sig = (np.linalg.norm(ps) / chi_n - 1) if np.linalg.norm(ps) > 0 else 0
                
                pc = (1 - c_c) * pc + c_c * (np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma) * (x_new - x_mean)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_inv
                
                # Rank-mu update
                C = (1 + c1 + cmu - c1 - cmu) * C
                C = (1 - c1 - cmu) * C
                C += c1 * np.outer(pc, pc)
                
                for i in range(mu):
                    wi = weights[i]
                    zi = np.linalg.solve(C, offspring_sorted[i] - x_mean) / sigma
                    C += cmu * wi * np.outer(zi, zi)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)
                
                x_mean = x_new
                gen += 1
                
                # Stagnation check: if no improvement in last pop_size generations, restart
                # For simplicity, we'll rely on the outer restart mechanism
                
            return evals_used, best_x, best_f

        # Multiple restarts with different initializations
        n_restarts = 5
        evals_per_restart = budget // n_restarts
        evals_per_restart = max(evals_per_restart, 20)
        
        total_evals = 0
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if total_evals >= budget:
                break
            
            # Different initialization strategies
            if restart == 0:
                x0 = np.zeros(n)
            elif restart == 1:
                x0 = rng.uniform(-2, 2, n)
            elif restart == 2:
                x0 = rng.uniform(-4, 4, n)
            elif restart == 3:
                x0 = rng.uniform(-5, 5, n)
            else:
                # Perturb the best found so far
                if best_x is not None:
                    x0 = best_x + 0.5 * rng.standard_normal(n)
                else:
                    x0 = rng.uniform(-3, 3, n)
            
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            total_evals += 1
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            remaining = min(evals_per_restart, budget - total_evals)
            if remaining <= 0:
                break
            
            rng_local = np.random.default_rng(42 + restart * 1000)
            used, bx, bf = cma_es(x0, f0, remaining, rng_local)
            total_evals += used
        
        return best_x, best_f
# EVOLVE-BLOCK-END
