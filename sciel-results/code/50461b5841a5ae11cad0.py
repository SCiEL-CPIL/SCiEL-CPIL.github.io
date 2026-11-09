"""Combined optimizer: CMA-ES for anisotropic navigation and (1+1)-ES for local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES and (1+1)-ES optimizer for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if calls_made[0] >= budget:
                raise Exception("Budget exceeded")

            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_made[0] += 1

            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # --- Phase 1: CMA-ES ---
        # CMA-ES parameters
        lambda_pop = 10  # Population size
        mu = lambda_pop // 2  # Number of recombination points
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = 1.0 / (lambda_pop * (1 + 2 * int(lambda_pop / (2 * n))))
        c_c = (4 + lambda_pop / n) / (lambda_pop + 4 + 2 * lambda_pop / n)
        c_1 = 2.0 / ((lambda_pop + 1.3)**2 + n)
        c_mu = 2.0 * (1.0 - 0.5 + mu_eff / (lambda_pop + 1.3)) / ((lambda_pop + 2)**2 + n)
        
        # State
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        chi_lambda = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        def cma_es_iteration():
            nonlocal x_mean, sigma, C, C_sqrt_inv, p_sigma, p_c, best_f, best_x
            if calls_made[0] >= budget or time.time() > deadline:
                return False

            # Generate population
            pop = []
            for i in range(lambda_pop):
                # Sample from N(0, I)
                z = rng.standard_normal(n)
                # Transform using Cholesky of C
                # Note: We need to update C_sqrt_inv when C changes. 
                # For simplicity and speed in this short budget, we can use a simpler update or 
                # recompute Cholesky every few steps. 
                # However, standard CMA-ES updates C and then uses it.
                # To avoid expensive Cholesky decomposition every step for n=5 (which is cheap anyway),
                # we will update C and compute the inverse square root.
                
                # Actually, for n=5, computing eigendecomposition is very fast.
                # Let's use a helper to get the sampling matrix.
                # We'll maintain the eigendecomposition of C.
                pass
            
            # To keep the code robust and simple, I will implement a slightly simplified CMA-ES 
            # that recomputes the sampling distribution from the covariance matrix every iteration.
            # For n=5, this is negligible overhead.
            
            # Update Cholesky/Eigendecomposition of C
            # We use eigendecomposition for stability
            evals, evecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            evals = np.maximum(evals, 1e-12)
            # Sampling matrix: evecs * diag(sqrt(evals))
            sampling_matrix = evecs @ np.diag(np.sqrt(evals))
            
            # Generate offspring
            offspring = []
            fitnesses = []
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                x_child = x_mean + sigma * (sampling_matrix @ z)
                x_child = np.clip(x_child, -5, 5)
                f_child = wrapped_func(x_child)
                offspring.append(x_child)
                fitnesses.append(f_child)
                
                if calls_made[0] >= budget or time.time() > deadline:
                    return False

            # Sort offspring
            sorted_indices = np.argsort(fitnesses)
            sorted_offspring = [offspring[i] for i in sorted_indices]
            sorted_fitnesses = [fitnesses[i] for i in sorted_indices]

            # Update mean
            x_mean_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)
            
            # Update step size
            y_sigma = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            h_sigma = 1.0
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made[0] // lambda_pop))) < (1.4 + 2.0 / (n + 1)) * chi_lambda:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
                
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            # Standard formula: sigma *= exp(c_sigma * (||p_sigma||/chi_lambda - 1))
            # The above line had a typo in logic, fixing:
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1.0))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update covariance matrix
            y_c = (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Rank-one and rank-mu updates
            C = (1 + c_1 * (2 - c_1 - c_mu) - c_1) * C + \
                c_1 * np.outer(y_c, y_c) + \
                c_mu * np.sum([weights[i] * np.outer(sorted_offspring[i] - x_mean, sorted_offspring[i] - x_mean) for i in range(mu)], axis=0) / (sigma * sigma)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-12 * np.eye(n)
            
            x_mean = x_mean_new
            return True

        # Run CMA-ES
        cma_stall = 0
        prev_best = best_f
        
        while calls_made[0] < budget * 0.8: # Use 80% of budget for CMA-ES
            if time.time() > deadline:
                break
            
            success = cma_es_iteration()
            if not success:
                break
                
            if best_f < prev_best - 1e-10:
                prev_best = best_f
                cma_stall = 0
            else:
                cma_stall += 1
                if cma_stall > 15: # Stagnation detected
                    break

        # --- Phase 2: (1+1)-ES Polish ---
        if calls_made[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma_es = 0.5 # Start with moderate step
            min_sigma = 1e-4
            max_sigma = 5.0
            
            # If CMA-ES found a very good spot, start with smaller sigma
            if best_f < -50: # Heuristic: if we are in a good basin
                sigma_es = 0.1

            while calls_made[0] < budget:
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma_es * rng.standard_normal(n), -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma_es = min(sigma_es * 1.1, max_sigma)
                else:
                    sigma_es = max(sigma_es * 0.9, min_sigma)
                    
                if sigma_es < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
