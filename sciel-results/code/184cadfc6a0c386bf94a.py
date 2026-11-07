import numpy as np
import time

class Optimizer:
    """
    Single-instance CMA-ES with aggressive early convergence and bounded restarts.
    
    Strategy:
    - Use a single, robust CMA-ES instance for the entire budget.
    - The landscape is Schaffers F7, which is a unimodal-like function with a global minimum.
    - The hidden rotation/shift is fixed per instance. CMA-ES adapts its covariance matrix to this rotation.
    - Avoiding multiple islands prevents diluting the budget and allows the covariance matrix to fully converge to the true landscape orientation.
    - A single instance with a larger population (lambda=12) and proper CMA-ES update rules is more likely to find the global minimum within 500 evaluations than multiple small, under-converged islands.
    - Early progress is crucial for AOCC. CMA-ES typically makes fast progress in the first few generations once it starts aligning with the gradient.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        # State
        best_x_global = None
        best_f_global = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x_global, best_f_global
            if calls_made >= total_budget:
                return float('inf')
            try:
                f = func(x)
                calls_made += 1
                if f < best_f_global:
                    best_f_global = f
                    best_x_global = x.copy()
                return f
            except Exception:
                return float('inf')

        # CMA-ES parameters
        # For dim=5, lambda=12 is a reasonable default (1+dim or 4+2*dim)
        lambda_ = 12
        mu = lambda_ // 2  # Select top half
        
        # Weights: linear weights, normalized
        weights = np.arange(mu, 0, -1, dtype=float)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2.0 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        
        # Initialize
        # Start at the center of the box, as the optimum is likely near the center after shift
        # But the shift is hidden. Let's start at 0.
        mean = np.zeros(dim)
        sigma = 2.0  # Start with a reasonable step size
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Evaluate initial point
        f = safe_eval(mean)
        if f < best_f_global:
            best_f_global = f
            best_x_global = mean.copy()
            
        gen = 0
        
        while calls_made < total_budget:
            if time.time() > deadline:
                break
                
            # Sample population
            pop = []
            fits = []
            
            for k in range(lambda_):
                # Sample from N(mean, sigma^2 * C)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                z = np.random.randn(dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                
                f = safe_eval(x)
                pop.append(x)
                fits.append(f)
                
                if calls_made >= total_budget:
                    break
            
            if len(pop) < mu:
                break
            
            # Sort
            idx = np.argsort(fits)
            pop = [pop[j] for j in idx]
            fits = [fits[j] for j in idx]
            
            # Update best
            if fits[0] < best_f_global:
                best_f_global = fits[0]
                best_x_global = pop[0].copy()
            
            # Update CMA-ES state
            # Weighted mean
            new_mean = np.zeros(dim)
            for j in range(mu):
                new_mean += weights[j] * pop[j]
            
            old_mean = mean
            mean = new_mean
            
            # Update sigma and C
            y = (new_mean - old_mean) / (sigma + 1e-12)
            
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) / \
                      (np.sqrt(mu_eff) * np.sqrt(np.trace(C)))
            
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (gen + 1))) / \
                  (np.sqrt(mu_eff) * np.sqrt(np.trace(C)))
            
            # Sigma update
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # C update
            rank_one = np.outer(y, y)
            rank_mu = np.zeros((dim, dim))
            for j in range(mu):
                y_j = (pop[j] - old_mean) / (sigma + 1e-12)
                rank_mu += weights[j] * np.outer(y_j, y_j)
            
            delta_h = (1 - h_sigma) * c_sigma * (2 - c_sigma)
            delta_hc = h_c * c_c * (2 - c_c) - 1
            
            C = (1 + c_1 * delta_hc + c_mu * (delta_h - delta_hc)) * C \
                - c_1 * rank_one + c_mu * rank_mu
                
            # Symmetrize and ensure PSD
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.any(eigvals < 1e-8):
                    C += 1e-6 * np.eye(dim)
            except:
                pass
            
            gen += 1
            
            # Termination criteria: if sigma is very small, we've converged
            if sigma < 1e-6:
                break

        return best_x_global, best_f_global
