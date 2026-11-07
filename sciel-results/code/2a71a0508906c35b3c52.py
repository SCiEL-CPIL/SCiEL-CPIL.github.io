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
        self.start_time = 0
        self.deadline = 0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        dim = self.dim
        budget = self.budget

        # Wrap function to count calls, check time, and track best
        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time Limit Exceeded")
            if self.calls >= budget:
                raise Exception("Budget Exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if x.shape != (dim,):
                try:
                    x = x.reshape(dim)
                except:
                    x = np.zeros(dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = float('inf')
            except Exception:
                f = float('inf')
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def run_cma_es(x0, sigma0, max_calls, pop_size=None):
            """
            Robust CMA-ES implementation.
            Returns best_x, best_f found within max_calls.
            """
            if max_calls <= 0:
                return x0.copy(), float('inf')
            
            # Evaluate initial point
            f0 = wrapped_func(x0)
            local_best_x = x0.copy()
            local_best_f = f0
            local_calls = 1
            
            if local_calls >= max_calls:
                return local_best_x, local_best_f

            # Parameters
            if pop_size is None:
                lam = max(4, int(4 + 3 * math.log(dim)))
            else:
                lam = pop_size
            
            mu = max(1, lam // 2)
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
            d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
            c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
            c_1 = 2.0 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1.0 - c_1, 2.0 * ((mu_eff + 2.0) / (dim + 2.0) - 1.0))
            
            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            chi_n = math.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))
            
            while local_calls < max_calls:
                if time.time() > self.deadline:
                    break
                
                n_off = min(lam, max_calls - local_calls)
                if n_off <= 0:
                    break
                
                # Sample offspring
                # Use Cholesky for sampling if possible, else eigen
                try:
                    L = np.linalg.cholesky(C)
                    samples = []
                    for _ in range(n_off):
                        z = np.random.randn(dim)
                        y = x_mean + sigma * (L @ z)
                        y = np.clip(y, -5.0, 5.0)
                        fy = wrapped_func(y)
                        local_calls += 1
                        samples.append((y, fy))
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()
                except np.linalg.LinAlgError:
                    # Fallback to eigen if C is not PD
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    std_eig = np.sqrt(eigvals)
                    samples = []
                    for _ in range(n_off):
                        z = np.random.randn(dim)
                        y = x_mean + sigma * ((eigvecs * std_eig) @ z)
                        y = np.clip(y, -5.0, 5.0)
                        fy = wrapped_func(y)
                        local_calls += 1
                        samples.append((y, fy))
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()
                
                if time.time() > self.deadline or local_calls >= max_calls:
                    break
                
                # Sort offspring
                samples.sort(key=lambda x: x[1])
                
                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * samples[i][0]
                x_new_mean = y_w
                
                diff = x_new_mean - x_mean
                norm_diff = np.linalg.norm(diff)
                
                if norm_diff < 1e-14:
                    # Stagnation: no movement in mean
                    break
                
                # Update evolution paths
                # p_sigma
                # Need C_inv_sqrt
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
                
                p_sigma = (1 - c_sigma) * p_sigma + \
                          math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                
                # h_sigma check
                h_sigma = 1.0
                if norm_p_sigma > 1.4 * chi_n:
                    h_sigma = 0.0
                
                # p_c
                p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                # Update sigma
                sigma = sigma * math.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)
                
                # Update C
                c1_term = c_1 * np.outer(p_c, p_c)
                
                cmu_term = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = samples[i][0]
                    diff_i = y_i - x_new_mean
                    cmu_term += w[i] * np.outer(diff_i, diff_i)
                
                C = (1 - c_1 - c_mu + mu_eff * c_mu) * C + c1_term + c_mu * cmu_term
                
                # Ensure symmetry
                C = (C + C.T) / 2.0
                
                x_mean = x_new_mean
            
            return local_best_x, local_best_f

        # Strategy:
        # 1. Coarse global search: Several random starts with large sigma.
        # 2. Fine local refinement: CMA-ES with smaller sigma on the best found.
        # 3. If time permits, a second pass on the best with even smaller sigma.
        
        # Phase 1: Global Exploration
        # Use 5 random starts.
        # Budget allocation: 20% of budget for global, 80% for local.
        # But we need to be careful. If global finds a good point, local refines.
        # If global is bad, local might still help if we start from a decent point.
        # Let's use 5 starts, each with 10% budget (50 calls).
        # Then refine best with remaining 50% budget (250 calls).
        # Wait, 5*50 = 250. Remaining 250. Total 500.
        
        n_global_starts = 5
        budget_global_each = max(10, budget // (n_global_starts * 2)) # 50
        total_global_budget = n_global_starts * budget_global_each
        
        # Ensure we don't exceed budget in global phase
        if total_global_budget >= budget:
            # Just run one CMA-ES with full budget
            run_cma_es(np.zeros(dim), 2.0, budget)
            return self.best_x, self.best_f
        
        # Run global starts
        for i in range(n_global_starts):
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            # Random start
            if i == 0:
                x_start = np.zeros(dim)
            else:
                x_start = np.random.uniform(-5.0, 5.0, dim)
            
            # Large sigma for global search
            sigma_global = 1.5
            # Use smaller population for global to save calls?
            # Standard CMA-ES pop size is ~9 for dim=5.
            # 50 calls / 9 ~ 5 generations. Might be too few to converge.
            # But we just want a decent starting point for local search.
            run_cma_es(x_start, sigma_global, budget_global_each)
        
        # Phase 2: Local Refinement
        remaining_budget = budget - self.calls
        if remaining_budget > 0 and self.best_x is not None:
            # Start from best found
            x_start = self.best_x.copy()
            # Small sigma for local search
            sigma_local = 0.5
            run_cma_es(x_start, sigma_local, remaining_budget)
            
            # Phase 3: Final Polish
            remaining_budget_2 = budget - self.calls
            if remaining_budget_2 > 5 and self.best_x is not None:
                x_start = self.best_x.copy()
                sigma_polish = 0.1
                run_cma_es(x_start, sigma_polish, remaining_budget_2)
        
        return self.best_x, self.best_f
