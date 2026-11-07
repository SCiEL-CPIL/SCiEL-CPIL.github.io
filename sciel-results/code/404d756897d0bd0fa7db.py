import scipy
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin for 300s limit
        
        # CMA-ES parameters
        n = self.dim
        lambda_ = 10  # population size
        mu = lambda_ // 2  # recombination weight count
        # weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        sigma = 2.0
        x_mean = np.zeros(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Initial population
        best_x = None
        best_f = np.inf
        calls = 0
        
        # We need to manage the budget carefully.
        # Each generation uses lambda_ evaluations.
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            B = np.random.randn(n, lambda_)
            # Standard normal
            # Update step size (simplified CMA update)
            # For simplicity and robustness in this short budget, 
            # we might skip the full CMA covariance update and just do 
            # a simple (mu+lambda)-ES with adaptive sigma, 
            # OR implement full CMA.
            
            # Let's implement full CMA to be safe, but it's complex.
            # Given 500 evals, a simpler (1+1)-ES with large sigma and restarts 
            # might be comparable, but CMA is better.
            
            # Actually, let's use scipy.optimize.minimize with Nelder-Mead 
            # and multiple restarts. It's less code and less prone to bugs.
            # But CMA is better for Rastrigin.
            
            # Let's stick to a simplified CMA-ES or just use the `cma` logic manually.
            # To avoid bugs in a hand-rolled CMA, I will use 
            # `scipy.optimize.minimize` with `method='Nelder-Mead'` 
            # and **10 random restarts**.
            
            # Wait, 10 restarts * 50 evals = 500.
            # Nelder-Mead converges fast.
            # This is a very safe and effective strategy for low-dim multimodal.
            
            break # Exit CMA loop, switch to restart strategy

        # Strategy: Multi-start Nelder-Mead
        n_restarts = 10
        budget_per_restart = self.budget // n_restarts
        
        best_x = np.zeros(self.dim)
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        for i in range(n_restarts):
            if time.time() > deadline:
                break
            # Random start in [-5, 5]
            x0 = rng.uniform(-5, 5, self.dim)
            # Run Nelder-Mead
            try:
                res = scipy.optimize.minimize(func, x0, method='Nelder-Mead', 
                                              options={'maxiter': budget_per_restart, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
                
        return best_x, best_f
