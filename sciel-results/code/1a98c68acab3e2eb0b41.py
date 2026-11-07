"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid CMA-ES + L-BFGS-B optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        start_time = time.time()
        deadline = start_time + 320.0

        # Global best tracking
        best_x = rng.uniform(-5, 5, dim)
        best_fx = func(best_x)
        calls_used = 1

        def safe_func(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            val = func(x)
            calls_used += 1
            return val

        # CMA-ES Implementation
        class CMAES:
            def __init__(self, func, x0, sigma0, budget, rng, lower, upper, deadline, best_state):
                self.func = func
                self.x = np.array(x0, dtype=float)
                self.sigma = sigma0
                self.budget = budget
                self.rng = rng
                self.lower = lower
                self.upper = upper
                self.deadline = deadline
                self.best_state = best_state
                
                self.n = len(x0)
                self.popsize = 4 + int(3 * np.log(self.n))
                self.C = np.eye(self.n)
                self.pc = np.zeros(self.n)
                self.cs = 1.0 + 2 * max(0, (np.sqrt(self.n - 1) - 1) / (self.n + 3))
                self.c1 = (2 + 2 / (self.n + 1.3)) / ((self.n + 2) ** 2 + 2)
                self.cmumax = self.c1
                self.cmu = min(self.cmumax, (2 + (mu / self.popsize - 1) / (self.n + 2)))
                self.damps = 1 + 2 * max(0, np.sqrt((self.n - 1) * (self.c1 / self.cmu - 1)) - 1) + self.c1
                self.mumax = int(np.ceil(self.popsize / 2))
                self.weights = np.log(self.mumax + 0.5) - np.log(np.arange(1, self.mumax + 1))
                self.weights /= np.sum(self.weights)
                self.mueff = 1 / np.sum(self.weights ** 2)
                self.mu = self.mumax
                self.chiN = self.n ** 0.5 * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

            def step(self):
                if time.time() > self.deadline or self.budget <= 0:
                    return False

                # Generate population
                # Eigendecomposition for sampling
                evals, evecs = np.linalg.eigh(self.C)
                evals = np.maximum(evals, 1e-20)
                B = evecs
                D = np.sqrt(evals)
                
                ys = []
                for _ in range(self.popsize):
                    z = self.rng.standard_normal(self.n)
                    y = B @ (D * z)
                    ys.append(y)
                
                ys = np.array(ys)
                xs = self.x + self.sigma * ys
                # Clip to bounds (simple reflection or clip? Clip is safer for contract)
                xs = np.clip(xs, self.lower, self.upper)

                # Evaluate
                fs = np.array([self.func(x) for x in xs])
                self.budget -= self.popsize

                # Update best
                min_idx = np.argmin(fs)
                if fs[min_idx] < self.best_state['fx']:
                    self.best_state['fx'] = fs[min_idx]
                    self.best_state['x'] = xs[min_idx].copy()

                # Sort
                order = np.argsort(fs)
                ys_sorted = ys[order]
                xs_sorted = xs[order]
                fs_sorted = fs[order]

                # Update mean
                self.x = np.sum(self.weights[:, None] * xs_sorted[:self.mu], axis=0)

                # Update sigma
                y_sigma = (self.x - self.x_prev) / self.sigma if hasattr(self, 'x_prev') else 0
                # Note: x_prev needs to be stored before update. 
                # Let's refactor step to store x_prev at start.
                
                # Actually, let's implement a simpler, more robust CMA-ES loop inline 
                # or fix the class. Given the complexity and risk of bugs in a custom CMA-ES 
                # within this constraint, a simpler population-based algorithm like 
                # Differential Evolution (DE) or a simple Random Restart with L-BFGS-B 
                # might be more robust and faster to implement correctly.
                
                # Let's pivot to a simpler, highly effective strategy:
                # 1. Use a small population of L-BFGS-B runs.
                # 2. Start them from diverse random points.
                # 3. Run L-BFGS-B with a tight function evaluation limit.
                # 4. If L-BFGS-B fails or is slow, fall back to local random search.
                
                # This avoids the complexity of CMA-ES implementation bugs.
                return True # Dummy return, logic will be replaced

        # Revised Strategy: Multi-start L-BFGS-B with fallback
        # L-BFGS-B is great for smooth functions. For multimodal, we need many starts.
        # Budget is 300. 
        # Strategy:
        # 1. Do a few random evaluations to get a sense of scale? No, just start.
        # 2. Run L-BFGS-B from random starts.
        # 3. Limit evaluations per L-BFGS-B call to prevent getting stuck.
        
        # Let's define a wrapper that counts evaluations
        self_eval_count = 0
        
        def counting_func(x):
            nonlocal self_eval_count
            x = np.clip(x, -5, 5)
            val = func(x)
            self_eval_count += 1
            return val

        # We have 300 calls total.
        # Let's divide the budget into chunks.
        # Chunk size for L-BFGS-B: e.g., 20-30 evaluations.
        # Number of starts: 300 / 25 = 12 starts.
        
        # To improve AOCC, we want early progress.
        # L-BFGS-B usually makes rapid progress initially.
        
        # We will run L-BFGS-B sequentially from different starting points.
        # We will keep the best result found so far.
        
        # Initial random point
        x0 = rng.uniform(-5, 5, dim)
        fx0 = counting_func(x0)
        best_x = x0.copy()
        best_fx = fx0
        calls_used = 1

        # Parameters for L-BFGS-B
        # maxiter is not strictly eval count, but maxfun is.
        # maxfun=25 means it stops after 25 function evaluations.
        
        num_starts = 10
        evals_per_start = 25
        
        for k in range(num_starts):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
                
            # Generate new starting point
            # Mix of random and perturbed best
            if k == 0:
                x_start = best_x.copy()
            elif k % 2 == 0:
                x_start = best_x + rng.standard_normal(dim) * 0.5
            else:
                x_start = rng.uniform(-5, 5, dim)
            
            x_start = np.clip(x_start, -5, 5)
            
            # Check if we have enough budget
            remaining = budget - calls_used
            if remaining < 5:
                break
                
            max_fun = min(evals_per_start, remaining)
            
            try:
                res = minimize(counting_func, x_start, method='L-BFGS-B', 
                               bounds=list(zip(np.full(dim, -5), np.full(dim, 5))),
                               options={'maxfun': max_fun, 'maxiter': max_fun})
                
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If L-BFGS-B fails, do a few local random steps
                x_local = x_start.copy()
                f_local = counting_func(x_local)
                for _ in range(min(5, budget - calls_used)):
                    x_try = x_local + rng.standard_normal(dim) * 0.5
                    x_try = np.clip(x_try, -5, 5)
                    f_try = counting_func(x_try)
                    if f_try < f_local:
                        x_local, f_local = x_try, f_try
                        if f_local < best_fx:
                            best_fx = f_local
                            best_x = x_local.copy()
                    if time.time() > deadline:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
