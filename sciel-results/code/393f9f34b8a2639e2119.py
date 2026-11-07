import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise Exception("Budget Exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # 1. Initial Global Exploration (Latin Hypercube Sampling)
        # Use 50 evaluations to cover the space reasonably well.
        n_init = 50
        rng = np.random.default_rng(12345)
        # LHS for better coverage than pure random
        # Simple LHS implementation
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                # Sample from a different stratified interval
                # For simplicity and speed, just random uniform is often good enough for dim=5, 
                # but let's use a simple jittered grid or just random.
                # Given dim=5, 50 points is sparse for LHS (10 per dim), let's just use random 
                # with a few specific points like 0.
                x[d] = rng.uniform(-5, 5)
        
        # Also evaluate center
        x_center = np.zeros(dim)
        f_center = wrapped_func(x_center)
        
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_f = wrapped_func(self.best_x)

        # 2. CMA-ES Implementation
        def run_cma_es(x0, sigma0, max_calls):
            local_calls = 0
            local_best_x = x0.copy()
            local_best_f = wrapped_func(x0)
            local_calls += 1
            
            if local_calls >= max_calls:
                return local_best_x, local_best_f, local_calls

            # Parameters
            lam = 9
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            while local_calls < max_calls:
                if time.time() > deadline:
                    break
                
                n_off = min(lam, max_calls - local_calls)
                if n_off <= 0:
                    break
                
                # Sample
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
                except:
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
                
                if time.time() > deadline:
                    break
                
                samples.sort(key=lambda x: x[1])
                
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * samples[i][0]
                x_new_mean = y_w
                
                diff = x_new_mean - x_mean
                norm_diff = np.linalg.norm(diff)
                
                if norm_diff < 1e-14:
                    break
                
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
                
                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = 0.0 if norm_p_sigma > 1.4 * chi_n else 1.0
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                
                c1_term = c_1 * np.outer(p_c, p_c)
                
                cmu_term = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = samples[i][0]
                    diff_i = y_i - x_new_mean
                    cmu_term += w[i] * np.outer(diff_i, diff_i)
                
                C = (1 - c_1 - c_mu + mu_eff * c_mu) * C + c1_term + c_mu * cmu_term
                C = (C + C.T) / 2
                x_mean = x_new_mean
                
                if sigma < 1e-12:
                    break
            
            return local_best_x, local_best_f, local_calls

        # Strategy:
        # We have 500 calls. 50 used for init. 450 remaining.
        # Run CMA-ES from best initial point.
        # Use most of the budget for the first run.
        # If stagnation is detected or budget remains, restart.
        
        remaining = budget - self.calls
        if remaining <= 0:
            return self.best_x, self.best_f

        # First run: 90% of remaining
        budget_run1 = int(remaining * 0.9)
        x_best, f_best, calls_used = run_cma_es(self.best_x.copy(), 1.5, budget_run1)
        
        if f_best < self.best_f:
            self.best_f = f_best
            self.best_x = x_best

        remaining = budget - self.calls
        if remaining > 20 and time.time() < deadline - 10:
            # Check for stagnation: if improvement was minimal, try a new start
            # Or just refine if we improved.
            # Let's try a restart from a random point if we suspect we are stuck.
            # Heuristic: if the last run didn't improve much relative to the range, restart.
            # But we don't know the range. 
            # Let's just refine with a smaller sigma if we improved, or restart if not.
            
            # Simple approach: Run another CMA-ES from the current best with a smaller sigma to refine.
            # Or from a new random point if the first one was bad.
            # Since Discus is unimodal, refining is usually better if we are close.
            
            # Let's try a second run from the best point found.
            budget_run2 = remaining
            # Use smaller sigma for refinement
            x_best2, f_best2, calls_used2 = run_cma_es(self.best_x.copy(), 0.5, budget_run2)
            
            if f_best2 < self.best_f:
                self.best_f = f_best2
                self.best_x = x_best2

        return self.best_x, self.best_f
