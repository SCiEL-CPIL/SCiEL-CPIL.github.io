"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using a simplified CMA-ES strategy with multi-start capability.
    CMA-ES adapts the covariance matrix to the landscape geometry, which is
    crucial for sharp ridge functions like BBOB f13.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        
        # We will perform multiple independent CMA-ES runs if budget allows.
        # For a 5D problem with 500 budget, we can afford 2-3 restarts.
        # Let's split the budget: 40% for first run, 30% for second, 30% for third.
        # Or simpler: just run one robust CMA-ES, but with a restart mechanism
        # if it gets stuck. Given the small dim, a single well-tuned CMA-ES
        # often suffices, but multi-start is safer for unknown shifts.
        
        # Strategy: Run CMA-ES with a population size suitable for dim=5.
        # lambda = 4 + floor(3 * log(dim)) = 4 + 3*1.6 = 8.8 -> 9
        # Let's use lambda = 12 for slightly better exploration.
        
        def run_cmaes(func, x0, f0, budget, rng):
            dim = self.dim
            lam = 12
            mu = lam // 2
            
            # Initial sigma
            sigma = 0.5
            x_mean = x0.copy()
            f_mean = f0
            
            # CMA-ES state
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)
            
            # Parameters
            alpha_mu = 1.0 / (2 * (lam + 2) ** 2 + 1)
            cmu = min(1 - 2 * alpha_mu, 2 / ((dim + 1.3) ** 2 + lam))
            c_sigma = (mu + 2) / (dim + mu + 5)
            d_sigma = 1 + 2 * max(0, sqrt((dim - 1) / (lam + 1) - 1)) + c_sigma
            c_c = (4 + lam / mu) / (dim + 4)
            chi_n = sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Weights
            weights = np.log(np.arange(lam, lam - mu, -1)) - np.log(np.arange(lam, lam - mu, -1))
            # Standard weights: w_i = log(lam - i + 1) - log(lam - mu) for i in 1..mu
            # Let's use standard equal weights for simplicity and robustness in low dim
            w = np.ones(mu) / mu
            mu_eff = 1 / np.sum(w ** 2)
            
            steps = 0
            max_steps = budget - 1
            
            while steps < max_steps:
                if time.time() > deadline:
                    break
                    
                # Generate population
                ys = []
                fs = []
                for i in range(lam):
                    # Sample from N(x_mean, sigma^2 * C)
                    # Cholesky decomposition for sampling
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                        
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    
                    fy = func(y)
                    ys.append(y)
                    fs.append(fy)
                    steps += 1
                    
                    if time.time() > deadline:
                        break
                
                if steps >= max_steps or time.time() > deadline:
                    break
                    
                # Sort by fitness
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                fs = [fs[i] for i in idx]
                
                # Update mean
                x_new = np.sum([w[i] * ys[i] for i in range(mu)], axis=0)
                
                # Update covariance matrix
                y_diff = np.array([ys[i] - x_mean for i in range(mu)])
                C = (1 - c_mu) * C + cmu * np.sum([w[i] * np.outer(y_diff[i], y_diff[i]) for i in range(mu)], axis=0)
                
                # Update path variables
                pc = (1 - c_c) * pc + c_c * (x_new - x_mean) / sigma
                hsig = (np.linalg.norm(pc) / sqrt(1 - (1 - c_c) ** (steps + 1)) / chi_n) < (1.4 + 2 / (dim + 1)) * c_sigma
                cs = (1 - c_sigma) * cs + np.sign(hsig) * c_sigma * (np.linalg.norm(pc) / chi_n - 1)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(pc) / chi_n - 1))
                
                # Update mean
                x_mean = x_new
                f_mean = fs[0]
                
                # Check for stagnation
                if steps % 20 == 0 and np.linalg.norm(x_new - x_mean) < 1e-8:
                    break
                    
            return x_mean, f_mean

        # Multi-start strategy
        # Split budget into chunks
        chunks = [int(self.budget * 0.4), int(self.budget * 0.3), int(self.budget * 0.3)]
        
        for i, chunk_budget in enumerate(chunks):
            if time.time() > deadline:
                break
            
            # Initialize new start point
            # Use a different seed for each restart to ensure diversity
            rng = np.random.default_rng(42 + i * 100)
            
            # Start near the center or random
            if i == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-5, 5, self.dim)
                
            f0 = func(x0)
            
            # Update global best
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            # Run CMA-ES
            if chunk_budget > 10:
                x_res, f_res = run_cmaes(func, x0, f0, chunk_budget, rng)
                if f_res < best_f:
                    best_f = f_res
                    best_x = x_res.copy()
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
