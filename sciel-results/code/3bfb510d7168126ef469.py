import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 9 + 3 * np.log(n)
        lam = int(np.ceil(lam))
        mu = max(1, lam // 2)
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4.3)
        c_m = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)) / (n + 1 + 2 * mu_eff)
        chi_lambda = np.sqrt(lam) * (1 - 1 / (2 * lam) + 1 / (8 * lam**2))
        
        # Initialize
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x.copy()
        best_f = np.inf
        calls = 0
        
        def safe_eval(x_eval):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            f_val = func(x_eval)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_eval.copy()
            return f_val

        # Initial evaluation
        f_init = safe_eval(x)
        if f_init is None:
            return best_x, best_f
            
        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            pop = np.empty((lam, n))
            fvals = np.empty(lam)
            
            # Compute B and D from C
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Sample
            valid_samples = 0
            for i in range(lam):
                z = rng.standard_normal(n)
                pop[i] = x + sigma * (B @ (D * z))
                f_val = safe_eval(pop[i])
                if f_val is None:
                    break
                fvals[i] = f_val
                valid_samples += 1
                
            if valid_samples == 0:
                break
            
            # If we didn't get a full population, adjust lam for this step?
            # Simpler: just use the valid samples if we have at least mu
            if valid_samples < mu:
                break
                
            # Use only the valid samples
            pop = pop[:valid_samples]
            fvals = fvals[:valid_samples]
            lam_eff = valid_samples
            
            # Sort
            idx = np.argsort(fvals)
            y_selected = pop[idx[:mu]]
            
            # Update mean
            y_mean = np.average(y_selected, axis=0, weights=w[:mu]) # w is length mu
            
            # Update ps
            # C^{-1/2}
            C_inv_sqrt = (B * (1.0 / D)) @ B.T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_mean)
            
            # h_sigma check
            # Approximation for h_sigma
            norm_ps = np.linalg.norm(ps)
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt(n + 1)
            # This formula is complex, often approximated as 1 if norm_ps < threshold
            # More accurate:
            # h_sig = 1 if norm_ps / chi_lambda < (1.4 + 2/(n+1)) * sqrt(n+1) * sqrt(1 - (1-c_sigma)**(2 * (calls/lam))) else 0
            # We can simplify h_sig = 1 for robustness in short runs
            h_sig = 1.0
            
            # Update pc
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Update C
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = y_selected[i]
                rank_mu += w[i] * np.outer(y_i, y_i)
                
            C = (1 + c_m * (mu_eff + 2 - 1) - c_m) * C + c_m * (rank_mu + (1 - h_sig) * (1 - c_c) * c_c * rank_one)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_lambda - 1))
            
            # Update mean
            x = x + y_mean
            
            # Symmetrize C
            C = (C + C.T) / 2
            
        return best_x, best_f
