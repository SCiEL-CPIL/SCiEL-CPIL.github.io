"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES style optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        # CMA-ES parameters
        # Population size lambda. For dim=5, lambda is typically 4 + floor(3*log(5+1)) ~ 12-16
        # Let's use a standard population size that fits well within the budget.
        # Budget 500. If lambda=10, we can do 50 generations.
        # If lambda=20, we can do 25 generations.
        # Rosenbrock is hard, so a larger population helps navigate the valley.
        # Let's try lambda=16 (4 + 3*log(6) ~ 10.8 -> 16 is a common choice for dim=5).
        # Actually, standard CMA-ES lambda is 4 + floor(3*log(n+1)). For n=5, 4+3*1.79=9.37 -> 13.
        # Let's use lambda=14.
        lambda_pop = 14
        max_generations = self.budget // lambda_pop
        
        # Initial mean: Center of the box
        mean = np.zeros(self.dim)
        # Initial sigma: Heuristic based on box size. Box is [-5, 5], width 10.
        # 0.5 * width / sqrt(dim) is a common heuristic? Or just 2.0?
        # The minimum is in [-4, 4]. Let's start with sigma=2.0.
        sigma = 2.0
        
        # CMA-ES weights
        mu = lambda_pop // 2  # Number of parents
        # Weights: standard CMA-ES weights
        weights = np.zeros(mu)
        # Logarithmic weights
        for i in range(mu):
            weights[i] = np.log(i + 1.5) - np.log(1.5)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Memory
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Main loop
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
                
            # Generate population
            pop = []
            for i in range(lambda_pop):
                # Standard normal
                z = np.random.standard_normal(self.dim)
                # Transform using CMA-ES formula: x = mean + sigma * B * D * z
                # We don't explicitly store B and D, we update C.
                # x = mean + sigma * np.linalg.cholesky(C) @ z
                # To avoid full Cholesky decomposition every time (expensive), 
                # we can approximate or just use it. dim=5 is small.
                L = np.linalg.cholesky(C)
                x = mean + sigma * (L @ z)
                pop.append(x)
            
            # Evaluate population
            fvals = np.array([evaluate(x) for x in pop])
            
            # Sort by fitness
            idx = np.argsort(fvals)
            pop_sorted = [pop[i] for i in idx]
            fvals_sorted = fvals[idx]
            
            # Update mean
            new_mean = np.sum([weights[i] * pop_sorted[i] for i in range(mu)], axis=0)
            
            # Update sigma
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * (C^-1/2 * (new_mean - mean))
            # C^-1/2 is inverse of Cholesky
            # L_inv = np.linalg.inv(L)
            # But L is from previous C. We should use the C used to generate the population.
            # Let's stick to the standard update.
            
            # Calculate step
            step = (new_mean - mean) / sigma
            # C^-1/2 @ step
            # Since C = B D^2 B^T, C^-1/2 = B^-T D^-1 B^-1
            # We have L = B D. So L @ z = B D z.
            # We need C^-1/2.
            # Let's compute C_inv_sqrt explicitly for dim=5.
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C).T) # This is not quite right. 
            # Better: use scipy.linalg.sqrtm or just invert C and sqrtm.
            # For dim=5, np.linalg.inv(C) is fast.
            C_inv = np.linalg.inv(C)
            # C^-1/2 = (C^-1)^1/2
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C_inv)
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            ps = (1 - c_sigma) * ps + C_inv_sqrt @ step * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            # Correct formula: sigma *= exp(c_sigma * (||ps||/chi_n - 1))
            # Wait, the standard formula is:
            # sigma = sigma * exp(c_sigma / c_sigma * (||ps||/chi_n - 1)) -> c_sigma cancels?
            # No, the update is: sigma_new = sigma * exp(c_sigma * (||ps||/chi_n - 1))
            # Let's re-verify.
            # Yes, sigma *= exp(c_sigma * (norm(ps)/chi_n - 1))
            # I already applied it above? No, I just calculated the factor.
            # Let's redo:
            sigma_factor = np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma *= sigma_factor
            
            # Update C
            # pc = (1-c_c)*pc + sqrt(c_c*(2-c_c)*mu_eff) * (new_mean - mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            # Rank-1 update
            y = step / np.linalg.norm(step) if np.linalg.norm(step) > 1e-10 else np.zeros(self.dim)
            rank1 = np.outer(y, y)
            
            # Rank-mu update
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                y_i = (pop_sorted[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + (1 + c_c - 1/c_c) * np.outer(pc, pc)) + c_mu * rank_mu
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            
            mean = new_mean
            
            # Check for convergence or stagnation
            if np.allclose(mean, best_x, atol=1e-6):
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
