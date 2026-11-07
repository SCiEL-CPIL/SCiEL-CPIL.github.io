import numpy as np
import time
from scipy.optimize import minimize # Using scipy's Nelder-Mead is robust and fast

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        # Phase 1: Global Exploration
        # 25 restarts, 12 calls each = 300 calls
        num_restarts = 25
        calls_per_restart = 12
        
        candidates = [] # Store (f, x) for all local minima found
        
        for r in range(num_restarts):
            if used_calls >= budget * 0.6: # Use 60% for global
                break
            if time.time() > deadline:
                break

            # Random start
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            used_calls += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            for _ in range(calls_per_restart):
                if used_calls >= budget * 0.6:
                    break
                if time.time() > deadline:
                    break

                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                candidates.append((fy, y.copy()))

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                else:
                    sigma *= 0.85

                sigma = np.clip(sigma, 1e-4, 10.0)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                if sigma < 1e-3:
                    break

        # Phase 2: Local Polishing with Nelder-Mead
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        # Take top 5 unique-ish candidates
        top_candidates = []
        last_x = None
        for f, x in candidates:
            if last_x is None or np.linalg.norm(x - last_x) > 0.1:
                top_candidates.append((f, x))
            if len(top_candidates) >= 5:
                break
            last_x = x

        remaining_budget = budget - used_calls
        if remaining_budget > 0 and top_candidates:
            calls_per_polish = remaining_budget // len(top_candidates)
            
            for f_init, x_init in top_candidates:
                if used_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Use Nelder-Mead
                # We need to wrap func to count calls
                # Since we can't easily wrap the original func for call counting inside scipy without modifying the interface,
                # and the problem says "Every call is recorded", we must ensure we don't exceed the budget.
                # SciPy's minimize might make many calls.
                # It's safer to implement a simple local search or limit Nelder-Mead iterations strictly.
                
                # Let's use a simple adaptive (1+1)-ES for polishing as well, but with smaller initial sigma.
                # Or a coordinate descent.
                # Given the fractal nature, Nelder-Mead is good.
                # Let's implement a lightweight Nelder-Mead or use a fine-grained (1+1)-ES.
                
                # Let's stick to (1+1)-ES with very small sigma for polishing to be safe with call counting.
                x = x_init.copy()
                fx = f_init
                sigma = 0.1
                
                for _ in range(calls_per_polish):
                    if used_calls >= budget:
                        break
                    if time.time() > deadline:
                        break

                    z = rng.standard_normal(d)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    used_calls += 1

                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.1 # Increase slightly
                    else:
                        sigma *= 0.9 # Decrease

                    sigma = np.clip(sigma, 1e-6, 1.0)
                    if sigma < 1e-5:
                        break

        return best_x, best_f
