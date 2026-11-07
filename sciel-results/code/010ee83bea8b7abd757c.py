import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0
        rng = np.random.default_rng(42)
        
        # Store evaluated points and their values for potential reuse or analysis
        # Not strictly necessary for CMA-ES if we just track best, but good for debugging
        # We'll just track the global best.
        
        def counting_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Strategy: Multi-Start CMA-ES
        # Katsuura is highly multimodal. A single CMA-ES run might get stuck.
        # We will perform multiple restarts with small populations.
        # Total budget: 500.
        # Let's use population size lambda = 8.
        # Each iteration uses lambda evaluations.
        # If we run for k iterations, we use k*lambda evaluations.
        # We want to spend ~80-90% of budget on optimization.
        # Reserve some for initial global sampling?
        # Actually, CMA-ES with a good initial spread (Halton) works well.
        # Let's use 10 restarts.
        # 500 / 10 = 50 evals per restart.
        # 50 / 8 = ~6 iterations per restart. This is very few iterations for CMA-ES to converge.
        
        # Alternative: Fewer restarts, larger population?
        # Lambda = 20. 500/20 = 25 iterations. 25/5 restarts = 5 restarts.
        # Or Lambda = 10. 500/10 = 50 iterations. 50/10 restarts = 5 restarts.
        
        # Let's try:
        # 1. Initial global scan with Halton (20% budget = 100 evals) to find promising regions.
        # 2. Run CMA-ES from the best points found in the scan.
        
        global_budget = int(budget * 0.2)
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        best_candidates = []
        
        for i in range(1, global_budget + 1):
            if time.time() > deadline - 10:
                break
            pt = np.array([halton(i, base) for base in bases])
            pt = pt * 10.0 - 5.0
            val = counting_func(pt)
            best_candidates.append((val, pt))
        
        # Sort candidates by value
        best_candidates.sort(key=lambda x: x[0])
        
        # Select top K candidates for CMA-ES starts
        K = 5
        starts = [pt for val, pt in best_candidates[:K]]
        
        remaining_budget = budget - eval_count
        
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Allocate remaining budget to CMA-ES restarts
        # We will run CMA-ES sequentially from the best starts.
        # If a CMA-ES run stagnates or uses up its allocated slice, we move to the next start.
        
        # How much budget per start?
        # Let's give each start 20% of the remaining budget?
        # Or just let them run until the global budget is exhausted?
        # CMA-ES is adaptive. Let's set a max number of iterations per run.
        
        # Standard CMA-ES setup:
        # sigma0: initial step size. Since domain is [-5,5], sigma0 = 1.0 is reasonable.
        # lambda: population size. Let's use 10.
        
        lambda_pop = 10
        max_iter_per_run = max(1, remaining_budget // lambda_pop)
        
        # We'll iterate through the starts.
        for start_idx, x0 in enumerate(starts):
            if eval_count >= budget:
                break
            if time.time() > deadline - 5:
                break
            
            # Run CMA-ES
            self._run_cma_es(counting_func, x0, lambda_pop, max_iter_per_run, deadline)
            
            # If we've used a significant chunk of budget, stop?
            # No, let it continue if budget allows, but we only have K starts.
            # If we run out of budget in the first start, we stop.
            
        return best_x, best_f

    def _run_cma_es(self, func, x0, lambda_pop, max_iter, deadline):
        """
        Simple CMA-ES implementation.
        """
        dim = self.dim
        x = x0.copy()
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Weights
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        ys = []
        xs = []
        
        for i in range(lambda_pop):
            z = np.random.normal(0, 1, dim)
            # Transform z using B and D
            y = B @ (D * z)
            x_new = x + sigma * y
            ys.append(y)
            xs.append(x_new)
            
        # Evaluate population
        fs = []
        for x_new in xs:
            if self._check_deadline(deadline):
                break
            f_val = func(x_new)
            fs.append(f_val)
        
        if len(fs) < lambda_pop:
            return
            
        # Sort by fitness
        idx = np.argsort(fs)
        ys = [ys[i] for i in idx]
        xs = [xs[i] for i in idx]
        fs = [fs[i] for i in idx]
        
        # Update mean
        x_new_mean = sum(weights[i] * xs[i] for i in range(mu))
        
        # Update covariance matrix
        y_mean = sum(weights[i] * ys[i] for i in range(mu))
        
        # Update evolution paths
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_mean
        h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / np.sqrt(1 - (1 - c_sigma)**(2 * self._iter_count)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
        
        p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
        
        # Update C
        delta_hsig = (1 - h_sig) * c_c * (2 - c_c)
        C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
        
        # Re-factorize C = B D^2 B^T
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D2, 1e-12))
        
        # Update sigma
        sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
        
        # Update mean
        x = x_new_mean
        
        # Store best for next iteration? 
        # Actually, standard CMA-ES re-samples from the new mean and sigma.
        # We need to loop this.
        
        # I'll implement a proper loop in the calling method or refactor this into a full loop.
        # The current structure is a single step. Let's make _run_cma_es a full loop.
        pass

    def _check_deadline(self, deadline):
        return time.time() > deadline - 5
