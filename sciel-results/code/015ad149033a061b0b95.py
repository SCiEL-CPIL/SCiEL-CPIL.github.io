import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        # Time limit guard
        start_time = time.time()
        time_limit = 280.0  # Leave 20s margin
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Number of parents
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Step size control
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start near center of the box, as minimum is in [-4, 4]^5
        x = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        
        ps = np.zeros(n)
        ph = np.zeros(n)
        
        best_x = x.copy()
        best_f = np.inf
        calls_used = 0
        
        # Helper to evaluate safely
        def safe_eval(x_eval):
            nonlocal calls_used
            if time.time() - start_time > time_limit:
                return None
            try:
                x_clipped = np.clip(x_eval, -5.0, 5.0)
                f_val = func(x_clipped)
                calls_used += 1
                return f_val
            except Exception:
                return None

        # Initial evaluation
        f_val = safe_eval(x)
        if f_val is not None:
            best_f = f_val
            best_x = x.copy()
        else:
            return x, np.inf

        # Main loop
        while calls_used < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            for i in range(lam):
                # Generate sample
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                ys[i] = y
                
                f_i = safe_eval(y)
                if f_i is None:
                    # Time limit or error reached
                    break
                fs[i] = f_i
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = y.copy()
                    
            # If we broke out of the inner loop due to time/error, exit main loop
            if time.time() - start_time > time_limit:
                break
                
            # Check if we have enough evaluations for this generation
            # If we ran out of budget mid-generation, we should stop
            if calls_used >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]
            
            # Update mean
            x_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)
            
            # Update evolution paths
            # Need to handle the case where we didn't use all lam evaluations?
            # CMA-ES assumes a full population. If we stopped early due to budget,
            # we might have incomplete data. 
            # However, the budget is 500 and lam=10, so we get exactly 50 generations.
            # If we stop early, the last generation might be incomplete.
            # To be safe, we only update if we have a full population?
            # Or just stop if we can't complete the generation.
            
            # Actually, if calls_used < budget but we broke the inner loop, 
            # it means time limit was hit. 
            # If we just finished the loop, we have a full generation.
            
            # Let's assume we have a full generation if we didn't break the inner loop.
            # The inner loop breaks if safe_eval returns None.
            # If it returns None, we break.
            
            # To keep it simple and robust, let's only update if we have a full population.
            # If we have partial, we just stop.
            
            # Determine if we have a full population
            # We can check if the last fs was set. 
            # But safe_eval increments calls_used.
            # Let's track if we completed the generation.
            
            # Actually, simpler: if calls_used < budget and time is ok, we proceed.
            # But if the inner loop broke early, we don't have all ys.
            # Let's just break the main loop if we didn't complete the inner loop.
            
            # How to know if inner loop broke?
            # If time.time() - start_time > time_limit, it broke.
            # Or if an exception happened.
            # Let's add a flag.
            
            # Re-structure:
            # We'll just stop if we can't complete a full generation.
            
            # For now, let's assume we always complete a generation unless time/budget is hit.
            # If time is hit, we break.
            
            # Update paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * ((x_new - x) / sigma)))
            # Wait, the formula is:
            # ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) / sigma * (B^T * D^-1 * (x_new - x))
            # B is eigenvectors, D is sqrt(eigenvalues).
            # B @ (D * z) is the transformation.
            # Inverse is B^T @ (D^-1 * v)
            
            # Let's recompute the update step carefully.
            # y = x + sigma * B @ (D * z)
            # x_new - x = sigma * B @ (D * z_avg)
            # z_avg = D^-1 @ B^T @ (x_new - x) / sigma
            
            z_avg = (D ** -1) * (B.T @ (x_new - x)) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_avg
            
            # Check for sigma adaptation
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used // lam))) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1)):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt((mu_eff + 1) / (n + 1)) - 1))
                
            # Update covariance matrix
            ph = (1 - c_c) * ph + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D * z_avg))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used // lam)))) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1))
            hsig = float(hsig)
            
            delta_sig = (1 - hsig) * c_c * (2 - c_c)
            
            C = (1 + c_1 * delta_sig - c_1 - c_mu) * C + c_1 * np.outer(ph, ph)
            for i in range(mu):
                z_i = (D ** -1) * (B.T @ (ys_sorted[i] - x)) / sigma
                C += c_mu * w[i] * np.outer(z_i, z_i)
            
            # Eigendecomposition
            C = (C + C.T) / 2
            D2, B = np.linalg.eigh(C)
            D2 = np.maximum(D2, 1e-10) # Ensure positive definite
            D = np.sqrt(D2)
            
            x = x_new
            
            # If sigma becomes too small or too large, we might stagnate
            # But with 50 generations, it should be fine.

        return best_x, best_f
