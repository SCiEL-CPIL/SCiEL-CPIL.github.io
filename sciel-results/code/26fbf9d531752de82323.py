import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

        # Helper to evaluate and track
        def evaluate(x):
            if self.calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            self.calls += 1
            try:
                f = func(x)
                if not np.isfinite(f):
                    f = float('inf')
            except Exception:
                f = float('inf')
            
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Robust CMA-ES implementation
        # Returns: best_x, best_f, calls_made
        def run_cma_es(x_start, sigma_start, max_calls):
            local_best_x = x_start.copy()
            local_best_f = evaluate(x_start)
            local_calls = 1
            
            if local_calls >= max_calls or time.time() > deadline:
                return local_best_x, local_best_f, local_calls

            # Parameters for dim=5
            lam = 9
            mu = 4
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
            
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            while local_calls < max_calls:
                if time.time() > deadline:
                    break
                
                n_off = min(lam, max_calls - local_calls)
                if n_off <= 0:
                    break
                
                # Sample offspring
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    std_eig = np.sqrt(eigvals)
                    sample_mat = (eigvecs * std_eig)
                except:
                    sample_mat = np.eye(dim)

                offspring = []
                for i in range(n_off):
                    z = np.random.randn(dim)
                    y = x_mean + sigma * (sample_mat @ z)
                    fy = evaluate(y)
                    local_calls += 1
                    offspring.append((y, fy))
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()
                        
                if time.time() > deadline or local_calls >= max_calls:
                    break
                
                offspring.sort(key=lambda t: t[1])
                
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * offspring[i][0]
                x_new_mean = y_w
                
                diff = x_new_mean - x_mean
                norm_diff = np.linalg.norm(diff)
                
                if norm_diff < 1e-12:
                    # Stagnation in this local run
                    break
                
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)

                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                h_sigma = 1.0
                if np.linalg.norm(p_sigma) > chi_n * 2.0:
                    h_sigma = 0.0
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                c1_term = c_1 * np.outer(p_c, p_c)
                cmu_term = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = offspring[i][0]
                    diff_i = y_i - x_new_mean
                    cmu_term += w[i] * np.outer(diff_i, diff_i)
                
                C = (1 - c_1 - c_mu + mu_eff * c_mu) * C + c1_term + c_mu * cmu_term
                C = (C + C.T) / 2
                x_mean = x_new_mean

            return local_best_x, local_best_f, local_calls

        # --- Multi-Start Strategy ---
        
        # Phase 1: Explore with 3 diverse starts
        # Start 1: Origin
        # Start 2: Random point in box
        # Start 3: Another random point
        
        # Allocate budget: 3 short runs of ~100 calls, then 1 long run with remainder
        # Total 500. 3*100 = 300. Remainder 200.
        
        starts = [
            np.zeros(dim),
            np.random.uniform(-3, 3, dim),
            np.random.uniform(-4, 4, dim)
        ]
        
        # Set a fixed seed for reproducibility of the multi-start points? 
        # The task says "hidden seeded shift", but our random starts can be random.
        # However, to ensure we don't waste calls if one start is bad, we cap them.
        
        short_budget = 100
        best_x_phase1 = np.zeros(dim)
        best_f_phase1 = float('inf')
        
        for s in starts:
            if self.calls >= budget - 50: # Keep at least 50 for final phase
                break
            if time.time() > deadline:
                break
                
            # Run short CMA-ES
            x_res, f_res, calls_used = run_cma_es(s, 1.5, short_budget)
            
            if f_res < best_f_phase1:
                best_f_phase1 = f_res
                best_x_phase1 = x_res
        
        # Phase 2: Exploit the best point found
        if self.calls < budget - 10 and time.time() < deadline:
            remaining_budget = budget - self.calls
            # Use a smaller sigma for refinement? Or let CMA-ES adapt?
            # CMA-ES adapts sigma. Starting with a moderate sigma is safe.
            # If we found a very good point, sigma might have shrunk. 
            # We restart CMA-ES from best_x_phase1.
            
            # We can also try a few local Nelder-Mead steps if CMA-ES stagnated, 
            # but CMA-ES with small sigma is effectively local search.
            
            x_final, f_final, _ = run_cma_es(best_x_phase1, 0.5, remaining_budget)
            
            # The global best_x and best_f are already updated by 'evaluate' inside run_cma_es
            # So self.best_x and self.best_f are correct.

        return self.best_x, self.best_f
