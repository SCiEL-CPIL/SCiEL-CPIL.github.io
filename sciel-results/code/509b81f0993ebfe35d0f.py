"""CMA-ES optimizer for BBOB-style Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is a state-of-the-art derivative-free optimization algorithm that
    adapts the sampling distribution to the shape of the fitness landscape.
    It is particularly effective for multimodal functions like Rastrigin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        rng = np.random.default_rng(42)

        dim = self.dim
        n = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        # Recommended: 4 + 3*ln(n)
        lam = int(4 + 3 * np.log(dim))
        if lam < 4:
            lam = 4
        # For dim=5, lam = 4 + 3*1.6 = 8.8 -> 8. Let's use 12 for robustness.
        lam = 12
        
        # Number of parents
        mu = lam // 2
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial step size
        sigma = 2.0
        
        # Initial mean: Random point in [-5, 5]
        m = rng.uniform(-5, 5, dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Bias correction factors
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        calls_used = 0
        x_best = m.copy()
        f_best = func(m)
        calls_used += 1
        
        def check_deadline():
            return time.time() > deadline

        def update_best(x, fx):
            nonlocal x_best, f_best
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        # Main Loop
        while calls_used + lam <= n and not check_deadline():
            # 1. Sample population
            pop = []
            for i in range(lam):
                # Sample from N(m, sigma^2 * C)
                # Use Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    L = np.eye(dim)
                    
                z = rng.standard_normal(dim)
                x = m + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)
                
            # 2. Evaluate
            fitnesses = []
            for x in pop:
                fx = func(x)
                calls_used += 1
                fitnesses.append(fx)
                update_best(x, fx)
                if check_deadline():
                    break
            
            if len(fitnesses) < lam:
                break

            # 3. Sort
            indices = np.argsort(fitnesses)
            pop_sorted = [pop[i] for i in indices]
            fitnesses_sorted = [fitnesses[i] for i in indices]
            
            # 4. Update mean
            new_m = np.zeros(dim)
            for i in range(mu):
                new_m += w[i] * pop_sorted[i]
                
            # 5. Update path lengths
            inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(C))
            # Note: inv_sqrt_C is L^{-1} where C = L L^T. 
            # We need C^{-1/2}. This is (L^{-1}) (L^{-1})^T? 
            # Actually, standard update uses C^{-1/2}.
            # Let's compute C_inv_sqrt properly.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (new_m - m) / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (new_m - m) / sigma)
            
            # 6. Update covariance matrix
            delta_sig = (np.linalg.norm(p_sigma) / chi_n - 1) * c_sigma * (1 + c_sigma)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(calls_used/lam + 1)))) < (1 + 2/(dim+1)) * chi_n
            
            # Rank-one update
            C = (1 + c_1 * delta_sig - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            diff = np.array([pop_sorted[i] - m for i in range(mu)])
            C += c_mu * (diff.T @ (w * diff))
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # 7. Update step size
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            m = new_m
            
            # Convergence check: if sigma is very small or step is negligible
            if sigma < 1e-10:
                break
                
            # Restart logic: If we are stuck (sigma very small) and haven't used all budget
            # For simplicity, we rely on the main loop condition and time.
            # If we want restarts, we could reset m and sigma here.
            # But with 500 calls, a single run of CMA-ES is often sufficient if initialized well.
            # Let's add a simple restart if no improvement in last 50 calls?
            # Actually, CMA-ES adapts well.

        return x_best, f_best
# EVOLVE-BLOCK-END
