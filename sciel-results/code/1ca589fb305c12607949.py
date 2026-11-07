"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin for 300s limit
        
        # State tracking
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget

        # Wrapped function to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return 1e10  # Penalize to stop optimization
            # Clip to box to handle boundary issues
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # CMA-ES parameters
        # For 5D, a population size of 10-20 is typical.
        # Budget is 500. 
        # Initial sigma should be large enough to explore the box [-5, 5]^5.
        # Box width is 10. A sigma of 2.0 covers most of the space initially.
        # However, Katsuura has a global minimum at 0 (before shift).
        # The shift is in [-4, 4]. So the minimum is in [-4, 4].
        # Starting at 0 with sigma 2.0 seems reasonable.
        
        pop_size = 12  # Standard for dim=5 is often 4 + floor(3*log(5)) = 4+4=8, but 12 is safer for robustness
        sigma = 2.0
        
        # Simple CMA-ES implementation to avoid dependency issues if 'cma' lib is not present, 
        # though scipy has some optimizers, cma is often better. 
        # Since we can't guarantee 'cma' package, I will implement a basic CMA-ES logic 
        # or use scipy's Powell/CNOR? No, CMA-ES is specifically requested as a different family.
        # I will implement a lightweight CMA-ES.
        
        # Initialization
        mean = np.zeros(self.dim)
        C = np.eye(self.dim)  # Covariance
        C_sqrt = np.eye(self.dim)  # Square root of C
        step_size = sigma
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # Parameters
        cm_c1 = 1 / (self.dim**1.7 + 9)
        cm_cmu = min(1 - cm_c1, 2 / ((self.dim + 1.3)**2 + pop_size))
        cc = (4 + pop_size/self.dim) / (self.dim + 4 + 2*pop_size/self.dim)
        cs = (pop_size + 2) / (self.dim + pop_size + 5)
        damps = 1 + 2*max(0, np.sqrt((pop_size-1)/(self.dim+1)) - 1) + cs
        
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Initial population
        best_x = mean.copy()
        best_f = wrapped_func(best_x)
        calls_made += 1
        
        # Main loop
        while calls_made < max_calls and time.time() < deadline:
            # Generate population
            # We need to generate pop_size points.
            # If remaining budget < pop_size, generate only as many as we can.
            current_pop_size = min(pop_size, max_calls - calls_made)
            if current_pop_size <= 0:
                break
                
            # Generate random vectors
            # z ~ N(0, I)
            z = np.random.randn(current_pop_size, self.dim)
            # y = C_sqrt @ z.T
            y = z @ C_sqrt.T
            # x = mean + step_size * y
            pop = mean[None, :] + step_size * y
            
            # Evaluate
            f_vals = np.empty(current_pop_size)
            for i in range(current_pop_size):
                f_vals[i] = wrapped_func(pop[i])
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
                
            # Sort by fitness
            order = np.argsort(f_vals)
            pop_sorted = pop[order]
            f_sorted = f_vals[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()
                
            # Update mean
            # Weighted average of top half
            w = np.zeros(pop_size)
            # Standard weights: top half have positive weights
            # mu = pop_size // 2
            mu = pop_size // 2
            # If current_pop_size < pop_size, adjust mu
            eff_mu = min(mu, current_pop_size // 2)
            if eff_mu > 0:
                w[:eff_mu] = np.log(np.arange(1, eff_mu+1))
                w_sum = np.sum(w)
                w /= w_sum
                mu_eff = 2 / np.sum(w**2)
                
                # Update mean
                mean = mean + step_size * np.dot(w[:eff_mu], pop_sorted[:eff_mu])
                
                # Update evolution paths
                y_mean = np.dot(w[:eff_mu], pop_sorted[:eff_mu])
                
                # Update p_sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt.T @ (y_mean / step_size)
                
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**((calls_made + 1) / (cs * self.dim)))) / chi_n < (1.4 + 2 / (self.dim + 1))
                
                # Update p_c
                h_sig = 1 if h_sigma else 0
                p_c = (1 - cc) * p_c + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - mean_prev) / step_size
                
                # Update covariance
                delta_h = (1 - h_sig) * cc * (2 - cc)
                C = (1 + cm_c1 * delta_h - cm_c1 - cm_cmu) * C + cm_c1 * np.outer(p_c, p_c) + cm_cmu * np.dot(w[:eff_mu] * (pop_sorted[:eff_mu] - mean)[None, :], (pop_sorted[:eff_mu] - mean).T)
                
                # Ensure C is positive semi-definite and update C_sqrt
                # Simple regularization if needed
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.sqrt(np.diag(eigvals))
                
                # Adapt step size
                step_size = step_size * np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Prevent step size from becoming too small or too large
                step_size = np.clip(step_size, 1e-12, 100.0)
                
                mean_prev = mean.copy()
            else:
                # If not enough points for weighting, just keep mean
                mean_prev = mean.copy()
                
            # Check convergence
            if step_size < 1e-10:
                break
                
            # Optional: Restart if stuck? For this budget, simple CMA-ES should suffice.
            # If variance is too low, we might want to restart, but with 500 budget,
            # a single well-tuned run is often best.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
