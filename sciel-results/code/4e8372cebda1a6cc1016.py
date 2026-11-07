"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with bounded sampling and stagnation restart."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin of 10s

        rng = np.random.default_rng(42)
        
        # Initial best
        x_best = rng.uniform(-4, 4, self.dim)
        f_best = func(x_best)
        
        # CMA-ES Parameters
        # lambda: number of offspring
        lam = 4 * (1 + int(2 * np.log(self.dim)))  # ~4 * (1 + 2*log(5)) = 4 * 4 = 16
        
        # Initial step size
        sigma = 1.0
        
        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        c_mu = np.sum(weights) ** 2 / (np.sum(weights ** 2))  # effective weight sum
        
        # Learning rates
        c_sigma = (c_mu + 2) / (self.dim + 4)
        c_c = (4 + c_mu / self.dim) / (self.dim + 4 + 2 * c_mu / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + c_mu)
        c_mu = 2 * (c_mu - 2 + 2 * c_mu / self.dim) / ((self.dim + 2) ** 2 + c_mu)
        
        # State variables
        ps = np.zeros(self.dim)
        pc = np.zeros((self.dim, self.dim))
        C = np.eye(self.dim)
        D = 1.0  # Cholesky factor scale (we maintain C directly for simplicity, but track D for normalization)
        # Actually, maintaining C directly is O(N^3) per update which is fine for N=5.
        
        # Initial mean
        mean = x_best.copy()
        
        # Pre-compute Cholesky of C (initially Identity)
        L = np.eye(self.dim)
        
        budget_used = 1
        stall_count = 0
        no_improve_count = 0
        
        while budget_used < self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Adaptation of step size
            # We need to evaluate lam points. 
            # Check if we have enough budget for a full generation
            if self.budget - budget_used < lam:
                # Do a final (1+1)-like step or just exit
                # Try a few more simple steps if budget is small
                while budget_used < self.budget and time.time() < deadline:
                    y = np.clip(mean + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    budget_used += 1
                    if fy < f_best:
                        f_best = fy
                        x_best = y
                        mean = y
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                break

            # Generate offspring
            ys = np.zeros((lam, self.dim))
            fs = np.zeros(lam)
            
            for i in range(lam):
                # Sample from N(0, C)
                z = rng.standard_normal(self.dim)
                # y = mean + sigma * L @ z
                y = mean + sigma * (L @ z)
                # Handle bounds: resample if outside
                # Simple rejection sampling
                while np.any(y < -5) or np.any(y > 5):
                    z = rng.standard_normal(self.dim)
                    y = mean + sigma * (L @ z)
                
                ys[i] = y
                fs[i] = func(y)
                budget_used += 1
                
                if time.time() > deadline:
                    # Emergency break
                    # Update best if we found something better
                    idx_min = np.argmin(fs)
                    if fs[idx_min] < f_best:
                        f_best = fs[idx_min]
                        x_best = ys[idx_min].copy()
                    return x_best, f_best

            # Sort by fitness
            idx_sorted = np.argsort(fs)
            ys = ys[idx_sorted]
            fs = fs[idx_sorted]
            
            # Update best
            if fs[0] < f_best:
                f_best = fs[0]
                x_best = ys[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Check stagnation
            if no_improve_count > 20 or sigma < 1e-10:
                # Restart
                # Keep best point, reset CMA state
                mean = x_best.copy()
                sigma = 1.0
                C = np.eye(self.dim)
                L = np.eye(self.dim)
                ps = np.zeros(self.dim)
                pc = np.zeros((self.dim, self.dim))
                no_improve_count = 0
                # Small penalty to budget to avoid infinite loop on restart if stuck
                # But we are already using budget. Just reset state.
                continue

            # Update mean (recombination)
            y_new = np.zeros(self.dim)
            for i in range(mu):
                y_new += weights[i] * ys[i]
            mean = y_new
            
            # Update path states
            # ps = (1 - c_sigma) ps + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * C^{-1/2} (y_new - mean_old) / sigma
            # Note: The formula uses the difference between the new mean and the old mean.
            # We need the old mean. We overwrote 'mean' in the recombination step? 
            # Let's store old_mean before updating.
            
            # Redo logic to capture old_mean
            # (This block was executed after mean update in previous draft, let's fix structure)
            
            # --- Corrected CMA-ES Step ---
            
            # 1. Sample and Evaluate (Done above, but let's ensure state is correct)
            # We already have ys, fs sorted.
            
            # 2. Update Mean
            old_mean = mean # Wait, 'mean' was updated in the loop above? No, 'mean' is the center.
            # I updated 'mean' in the "Update mean" section above. 
            # Let's trace: 
            # - 'mean' is used for sampling.
            # - After sampling, we compute y_new.
            # - We should update 'mean' to y_new AFTER the path updates, or use old_mean for path updates.
            
            # Let's restart the logic block for clarity and correctness.
            
            pass # Placeholder to prevent code execution of the flawed block above
            
            # --- RESTART OF LOGIC BLOCK ---
            # The code above was messy. I will rewrite the entire loop cleanly below.
            # Since I can't just insert, I will structure the final code carefully.
            
            break # Exit the while loop to prevent using the messy code above.
            # The code below will be the actual implementation.

        # The actual clean implementation starts here.
        # I will replace the entire __call__ content with a clean version.
        return x_best, f_best

    def _run_cma_es(self, func, budget, dim, rng):
        """Clean CMA-ES implementation."""
        start_time = time.time()
        deadline = start_time + 290
        
        x_best = rng.uniform(-4, 4, dim)
        f_best = func(x_best)
        budget_used = 1
        
        lam = 4 * (1 + int(2 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        c_mu = np.sum(weights) ** 2 / (np.sum(weights ** 2))
        
        c_sigma = (c_mu + 2) / (dim + 4)
        c_c = (4 + c_mu / dim) / (dim + 4 + 2 * c_mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + c_mu)
        c_mu = 2 * (c_mu - 2 + 2 * c_mu / dim) / ((dim + 2) ** 2 + c_mu)
        
        sigma = 1.0
        mean = x_best.copy()
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        C = np.eye(dim)
        L = np.linalg.cholesky(C)
        
        no_improve_count = 0
        
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            if budget - budget_used < lam:
                # Tail optimization
                while budget_used < budget and time.time() < deadline:
                    y = np.clip(mean + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    budget_used += 1
                    if fy < f_best:
                        f_best = fy
                        x_best = y
                        mean = y
                    else:
                        sigma *= 0.9
                break

            # Generate offspring
            ys = np.zeros((lam, dim))
            fs = np.zeros(lam)
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                while np.any(y < -5) or np.any(y > 5):
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (L @ z)
                ys[i] = y
                fs[i] = func(y)
                budget_used += 1
                
                if time.time() > deadline:
                    idx_min = np.argmin(fs)
                    if fs[idx_min] < f_best:
                        f_best = fs[idx_min]
                        x_best = ys[idx_min].copy()
                    return x_best, f_best

            idx_sorted = np.argsort(fs)
            ys = ys[idx_sorted]
            fs = fs[idx_sorted]
            
            if fs[0] < f_best:
                f_best = fs[0]
                x_best = ys[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > 20 or sigma < 1e-10:
                mean = x_best.copy()
                sigma = 1.0
                C = np.eye(dim)
                L = np.linalg.cholesky(C)
                ps = np.zeros(dim)
                pc = np.zeros((dim, dim))
                no_improve_count = 0
                continue

            old_mean = mean.copy()
            y_new = np.zeros(dim)
            for i in range(mu):
                y_new += weights[i] * ys[i]
                
            # Update path states
            # ps
            if sigma > 1e-15:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * c_mu) / sigma * np.linalg.solve(L, (y_new - old_mean) / sigma)
            
            # Update covariance matrix
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget_used + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            
            if sigma > 1e-15:
                # pc
                pc = (1 - c_c) * pc + h_sigma * c_c * (1 - c_c) * (y_new - old_mean) / sigma
                
                # Rank-1 and Rank-mu updates
                C = (1 + c_1 * h_sigma * (h_sigma - 1) - c_1 - c_mu) * C \
                    + c_1 * (pc @ pc.T - (1 - h_sigma) * (1 - c_c) * C) \
                    + c_mu * np.sum([weights[i] * np.outer(ys[i] - old_mean, ys[i] - old_mean) / sigma**2 for i in range(mu)], axis=0)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Cholesky
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset
                    C = np.eye(dim)
                    L = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros((dim, dim))
                    sigma = max(sigma, 0.1)
            
            # Update step size
            sigma *= np.exp(c_sigma / (dim + 1.3) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget_used + 1))) - np.sqrt(dim * (1 + 1/dim))))
            
            mean = y_new

        return x_best, f_best

    def __call__(self, func):
        """
        Wrapper to ensure correct interface and error handling.
        """
        try:
            return self._run_cma_es(func, self.budget, self.dim, np.random.default_rng(42))
        except Exception:
            # Fallback to simple random search if CMA-ES fails (unlikely but safe)
            rng = np.random.default_rng(0)
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            for _ in range(self.budget - 1):
                y = rng.uniform(-5, 5, self.dim)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
            return x, fx
# EVOLVE-BLOCK-END
