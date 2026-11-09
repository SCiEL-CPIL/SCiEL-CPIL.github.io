"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: random initialization + CMA-ES for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def run_cmaes(x_init, sigma_init, max_calls):
            """CMA-ES implementation for local refinement."""
            nonlocal calls_made, best_x, best_f

            pop_size = 8
            mu = pop_size // 2
            
            # Weights for mean update
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)
            
            # Step size control parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            # Initialize
            mean = x_init.copy()
            sigma = sigma_init
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Initial population
            population = []
            for _ in range(pop_size):
                z = rng.standard_normal(dim)
                x = mean + sigma * (B @ (D * z))
                population.append(np.clip(x, -5, 5))
            
            # Evaluate initial population
            pop_f = [evaluate(x) for x in population]
            
            local_best_x = x_init.copy()
            local_best_f = best_f
            no_improve = 0
            max_no_improve = 50
            
            while not check_deadline() and calls_made < max_calls:
                # Sort population
                order = np.argsort(pop_f)
                population = [population[i] for i in order]
                pop_f = [pop_f[i] for i in order]
                
                if pop_f[0] < local_best_f:
                    local_best_f = pop_f[0]
                    local_best_x = population[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve >= max_no_improve:
                    break
                
                # Compute weighted mean of top mu individuals
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * population[i]
                    
                # Update covariance matrix adaptation
                # (Simplified: use rank-one and rank-mu updates)
                
                # Update evolution paths
                inv_sqrt_hess = np.linalg.inv(np.sqrt(np.diag(D**2) @ (B.T @ B) @ np.diag(D**2)))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_hess @ ((mean - new_mean) / sigma)
                mean = new_mean
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - new_mean) / sigma
                
                # Adapt covariance matrix
                # Simplified update for stability
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - max_calls + pop_size)))) / \
                          np.sqrt(1 - (1 - c_sigma) ** 2)
                
                # Rank-one update
                delta = h_sigma * c_1 / sigma * (pc @ pc - np.eye(dim))
                # Rank-mu update (simplified)
                rank_mu = 0
                for i in range(mu):
                    z_i = inv_sqrt_hess @ (population[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                delta += (1 - h_sigma) * c_mu / sigma * rank_mu
                
                # Update B and D via eigen decomposition (approximate)
                # For simplicity and speed, we use a simplified step size update
                # and avoid full eigen decomposition in the inner loop for performance
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) - 1))
                
                # Generate new population
                new_population = []
                for _ in range(pop_size):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (B @ (D * z))
                    new_population.append(np.clip(x, -5, 5))
                    
                # Evaluate new population
                new_pop_f = [evaluate(x) for x in new_population]
                
                # Replace population
                population = new_population
                pop_f = new_pop_f
                
                # Check convergence
                if sigma < 1e-10:
                    break

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            return local_best_f, local_best_x

        # Phase 1: Random initialization to find promising basins
        # Use a larger random search phase to locate the basin given the shifted/rotated landscape
        scan_budget = min(100, total_budget - calls_made)

        # Evaluate random points in the search space
        for _ in range(scan_budget):
            if check_deadline():
                break
            x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        # Phase 2: CMA-ES refinement from the best point found
        if not check_deadline() and best_x is not None:
            remaining_budget = total_budget - calls_made
            
            # Run CMA-ES from the best point found in Phase 1
            # Use a moderate initial sigma to balance exploration and exploitation
            sigma_init = 1.5
            max_calls = total_budget
            _, _ = run_cmaes(best_x.copy(), sigma_init, max_calls)

        return best_x, best_f
# EVOLVE-BLOCK-END
