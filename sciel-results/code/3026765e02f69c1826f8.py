"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and adaptive population for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 320  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Strategy: multiple restarts of CMA-ES with different initializations
        # Budget allocation: split budget among restarts
        # For 5D, we can afford several restarts
        
        # First, do a quick random search to find a good starting point
        # Then run CMA-ES from that point
        
        # Phase 1: Quick random search (10% of budget)
        n_random = max(10, budget // 10)
        for i in range(n_random):
            if time.time() - start_time > timeout:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with restarts (remaining budget)
        remaining_budget = budget - n_random
        if remaining_budget <= 0:
            return best_x, best_f

        # Number of restarts
        n_restarts = min(3, max(1, remaining_budget // 50))
        budget_per_restart = remaining_budget // n_restarts

        for restart in range(n_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Initialize CMA-ES
            if restart == 0:
                x0 = best_x.copy()
            else:
                # Perturb from best found
                x0 = best_x + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5, 5)
            
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~9 for dim=5
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            # Initial step size
            sigma = 1.0
            
            # State variables
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            invsqrtC = np.eye(dim)
            D = 1.0
            B = np.eye(dim)
            
            # Track best
            restart_best_x = x0.copy()
            restart_best_f = func(x0)
            if restart_best_f < best_f:
                best_f = restart_best_f
                best_x = restart_best_x.copy()
            
            calls_used = 1
            max_iter = budget_per_restart - 1
            
            for it in range(max_iter):
                if time.time() - start_time > timeout:
                    break
                if calls_used >= budget_per_restart:
                    break
                
                # Generate population
                population = []
                for i in range(pop_size):
                    bi = rng.standard_normal(dim)
                    xi = x0 + sigma * (B @ (D * bi))
                    xi = np.clip(xi, -5, 5)
                    population.append(xi)
                
                # Evaluate
                fvals = np.array([func(x) for x in population])
                calls_used += pop_size
                
                # Sort by fitness
                indices = np.argsort(fvals)
                population = np.array(population[indices])
                fvals = fvals[indices]
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = population[0].copy()
                if fvals[0] < restart_best_f:
                    restart_best_f = fvals[0]
                    restart_best_x = population[0].copy()
                
                # Update mean
                w = weights[:mu]
                x0_new = np.sum(w[:, np.newaxis] * population[:mu], axis=0)
                
                # Update covariance matrix adaptation
                y = (x0_new - x0) / sigma if sigma > 1e-15 else np.zeros(dim)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1))) <
                         (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y
                
                delta_sigma = (np.linalg.norm(ps) / np.sqrt(mueff) -
                               np.exp(-0.5 * cs * (it + 1))) / (1 - (1 - cs)**(0.5 * (it + 1)))
                sigma *= np.exp(cs / (dim + 3) * (delta_sigma - 1))
                sigma = np.clip(sigma, 1e-15, 1e6)
                
                rank_mu = np.sum(w**2)
                rank_one = 1 - (2 * (1 - hsig) * cc * mueff / dim)
                C = ((1 + c1 * rank_one + c1 * rank_mu - c1) * C +
                     c1 * np.outer(pc, pc) +
                     cmu * np.sum(w[:, np.newaxis, np.newaxis] *
                                  np.outer(population[:mu] - x0, population[:mu] - x0) / sigma**2, axis=0))
                
                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update B and D
                eigvals, B = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = 1.0 / np.sqrt(eigvals)
                invsqrtC = B @ np.diag(D) @ B.T
                
                x0 = x0_new
                
                # Convergence check
                if sigma < 1e-12 or np.all(np.abs(population[:mu] - x0) < 1e-12):
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
