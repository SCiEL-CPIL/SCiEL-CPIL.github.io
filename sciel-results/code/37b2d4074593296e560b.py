"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    This is a derivative-free, population-based optimization algorithm.
    It is particularly effective for non-convex, ill-conditioned functions like Rosenbrock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Parameters for CMA-ES
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Maximum number of generations
        max_gens = budget // lam + 1
        
        # Initial parameters
        # Start at center of the box
        x_mean = np.zeros(n)
        sigma = 2.0  # Initial step size, covering [-2, 2] roughly, box is [-5, 5]
        C = np.eye(n)  # Covariance matrix
        C_inv_sqrt = np.eye(n) # Inverse square root of C
        ps = np.zeros(n)  # Evolution path for step size control
        pc = np.zeros(n)  # Evolution path for covariance matrix control
        
        # Learning rates
        mu = lam // 2  # Number of best points used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_cov_pos = (2 + 2 * mu_eff / n) / (n + 1 + 3 * mu_eff / n)
        c_cov_neg = (c_cov_pos + 2 - c_c) / (n + 3)
        
        # Initial best
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0

        # Helper to evaluate function with budget check
        def eval_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            fx = func(x)
            return fx

        # Evaluate initial mean
        f_mean = eval_func(x_mean)
        if f_mean is not None:
            best_f = f_mean
            best_x = x_mean.copy()

        for gen in range(max_gens):
            if calls_made >= budget:
                break
            
            # Sample population
            pop = []
            f_pop = []
            
            for i in range(lam):
                if calls_made >= budget:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                # Use Cholesky decomposition of C for sampling
                # Since C is updated, we need to maintain C_sqrt or use eigen decomposition
                # For simplicity and stability in small n, we use the current C_inv_sqrt
                # Actually, standard CMA-ES samples: z ~ N(0, I), x = mean + sigma * C_sqrt @ z
                # We need C_sqrt. Let's maintain it or compute it.
                # To avoid complex matrix square root updates, we can use the property:
                # x = mean + sigma * (C_inv_sqrt.T @ z) is NOT correct for sampling from C.
                # We need y ~ N(0, C). y = C_sqrt @ z.
                
                # Let's compute C_sqrt via eigen decomposition each generation for n=5, it's cheap.
                # Or maintain the factorization.
                # Given n=5, eigen decomposition is O(n^3) = 125 ops, negligible.
                
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive semi-definite
                eigvals = np.maximum(eigvals, 1e-16)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                z = np.random.randn(n)
                x_candidate = x_mean + sigma * (C_sqrt @ z)
                
                # Clip to box? The task says func clips. But for CMA-ES, clipping can break assumptions.
                # However, the domain is bounded. We should probably stay within bounds or let func clip.
                # Letting func clip is safer for the optimizer's internal state if we assume the function is defined everywhere.
                # But if we clip, the effective domain is smaller.
                # Let's not clip x_candidate before evaluation, but rely on func's clipping.
                
                fx = eval_func(x_candidate)
                if fx is None:
                    break
                pop.append(x_candidate)
                f_pop.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x_candidate.copy()
            
            if len(pop) < mu:
                break

            # Sort by fitness
            indices = np.argsort(f_pop)
            pop = [pop[i] for i in indices]
            f_pop = [f_pop[i] for i in indices]
            
            # Update mean
            x_new_mean = np.zeros(n)
            for i in range(mu):
                x_new_mean += weights[i] * pop[i]
            
            # Update evolution path for sigma
            x_mean_diff = x_new_mean - x_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ x_mean_diff) / sigma
            
            # Check sigma adaptation
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(n):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_diff) / sigma
            
            # Update covariance matrix
            delta_h = (1.1 + 2 / (n + 1)) * np.sqrt((1 - (1 - c_c) ** (gen + 1)) * (1 - (1 - c_c) ** (2 * (gen + 1))))
            rank_mu = np.outer(x_mean_diff, x_mean_diff)
            rank_one = np.outer(pc, pc)
            
            C = (1 + c_cov_pos * c_c * (1 - c_c) * delta_h * delta_h - c_cov_neg) * C \
                + c_cov_pos * rank_mu + c_cov_neg * rank_one
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-16)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update C_inv_sqrt
            C_inv_sqrt = np.linalg.inv(np.sqrtm(C))
            
            x_mean = x_new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
