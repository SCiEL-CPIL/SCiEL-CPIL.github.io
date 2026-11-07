import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        deadline = start_time + 280  # safety margin before 300s limit

        # Deterministic multi-start CMA-ES with progressively smaller sigmas
        # Sphere is convex, so any local minimum is the global minimum.
        # Start large to find the basin, then refine with smaller sigmas.
        
        best_x = np.zeros(n)
        best_f = float('inf')
        total_calls = 0

        # Define a sequence of initial sigmas to try
        # Large sigma to cover the domain, then progressively smaller
        sigmas = [4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
        
        # For each sigma, run CMA-ES with the remaining budget
        # But we need to be careful about how much budget to allocate to each
        # Strategy: give each run a portion of the budget, or run until stagnation
        # Better: run CMA-ES with a sub-budget, and if it stagnates, restart with smaller sigma
        
        # Let's allocate budget proportionally or use a fixed fraction
        # Since Sphere converges quickly once in the right region, we can afford multiple restarts
        # Let's try: first run gets 60% of budget, subsequent runs get 20% each, etc.
        # Or simpler: run each CMA-ES with a time/call limit, and restart if no improvement
        
        # Actually, for Sphere, CMA-ES with a good initial sigma should converge very fast.
        # The key is to find the right region. Let's do:
        # 1. One CMA-ES with sigma=4.0, budget=budget//2
        # 2. If we have budget left, restart with sigma=1.0 from best_x, budget=budget//4
        # 3. Continue with smaller sigmas if budget remains
        
        # Better approach: run CMA-ES with adaptive restart
        # Monitor progress; if no improvement for K generations, restart with smaller sigma
        
        # Let's implement a simpler strategy:
        # Phase 1: CMA-ES with sigma=4.0, use up to 70% of budget
        # Phase 2: CMA-ES with sigma=1.0 starting from best, use up to 20% of budget  
        # Phase 3: CMA-ES with sigma=0.2 starting from best, use up to 10% of budget
        
        budget_phase1 = int(budget * 0.6)
        budget_phase2 = int(budget * 0.25)
        budget_phase3 = int(budget * 0.15)
        
        # Phase 1: Large sigma to find the basin
        if total_calls < budget and time.time() < deadline:
            best_x, best_f, calls = self._cmaes(func, np.zeros(n), 4.0, budget_phase1, deadline, total_calls)
            total_calls += calls
        
        # Phase 2: Medium sigma to refine
        if total_calls < budget and time.time() < deadline and budget_phase2 > 10:
            best_x, best_f, calls = self._cmaes(func, best_x.copy(), 1.0, budget_phase2, deadline, total_calls)
            total_calls += calls
        
        # Phase 3: Small sigma for final refinement
        if total_calls < budget and time.time() < deadline and budget_phase3 > 5:
            best_x, best_f, calls = self._cmaes(func, best_x.copy(), 0.2, budget_phase3, deadline, total_calls)
            total_calls += calls
        
        # If we still have budget and time, do a final fine-tuning run
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline:
            best_x, best_f, calls = self._cmaes(func, best_x.copy(), 0.05, remaining, deadline, total_calls)
            total_calls += calls

        return best_x, best_f

    def _cmaes(self, func, x0, sigma, budget, deadline, offset_calls=0):
        """
        CMA-ES implementation with call tracking offset.
        """
        n = len(x0)
        calls = 0
        best_x = x0.copy()
        best_f = float('inf')
        
        # Parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))  # lambda = 4 + 3 ln(n)
        mu = max(1, pop_size // 2)  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initialize state
        x_mean = x0.copy()
        C = np.eye(n)
        C_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Evaluate initial point
        f0 = func(x_mean)
        calls += 1
        if f0 < best_f:
            best_f = f0
            best_x = x_mean.copy()
            
        stagnation_count = 0
        best_f_prev = best_f
        max_stagnation = 20  # generations without improvement before stopping
        
        while calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sample y
                z = np.random.randn(n)
                y = C_sqrt @ z
                x_new = x_mean + sigma * y
                
                # Evaluate
                f_new = func(x_new)
                calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
                
            if len(fs) < pop_size:
                break
                
            # Sort
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Check for stagnation
            if fs[0] >= best_f_prev - 1e-15:
                stagnation_count += 1
            else:
                stagnation_count = 0
            best_f_prev = fs[0]
            
            if stagnation_count >= max_stagnation:
                break
                
            # Update mean
            x_mean_new = x_mean + sigma * sum(weights[i] * ys[i] for i in range(mu))
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (x_mean_new - x_mean))
            h_sig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // pop_size))) < (1.4 + 2/(n+1)) * chi_n else 0.0
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            
            # Update covariance
            delta_hsig = (1 - h_sig) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive definite (regularization if needed)
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update x_mean
            x_mean = x_mean_new
            
            # Update C_sqrt (Cholesky decomposition)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: reset C to identity if it becomes singular
                C = np.eye(n)
                C_sqrt = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = max(sigma, 0.01)

            # Stop condition: if sigma is very small
            if sigma < 1e-10:
                break
            if np.max(fs[:mu]) - np.min(fs[:mu]) < 1e-14 * (1 + abs(np.min(fs))):
                break

        return best_x, best_f, calls
