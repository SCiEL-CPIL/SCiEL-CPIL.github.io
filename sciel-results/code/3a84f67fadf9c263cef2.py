import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with time guard for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial exploration
        # Use 20% of budget for random search to find candidates
        n_init = max(10, self.budget // 5)
        best_x = None
        best_f = np.inf
        candidates = []
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            candidates.append((best_f, best_x))
            
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local optimization using (1+1)-ES
        def run_es(x0, budget_es):
            x = x0.copy()
            fx = func(x)
            sigma = 1.0
            best_x_local = x.copy()
            best_f_local = fx
            no_improve = 0
            for _ in range(budget_es):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f_local:
                        best_f_local = fx
                        best_x_local = x.copy()
                        no_improve = 0
                    sigma *= 1.25
                else:
                    sigma *= 0.8
                    no_improve += 1
                    if no_improve > 20: # Stagnation
                        break
            return best_x_local, best_f_local

        # Allocate budget to top candidates
        remaining_budget = self.budget - n_init
        if remaining_budget <= 0:
            return best_x, best_f

        # Split remaining budget among top 3 candidates
        b1 = remaining_budget // 2
        b2 = remaining_budget // 4
        b3 = remaining_budget - b1 - b2
        
        best_x_final = best_x
        best_f_final = best_f
        
        # Try candidate 1
        x1, f1 = run_es(candidates[0][1], b1)
        if f1 < best_f_final:
            best_f_final = f1
            best_x_final = x1
            
        # Try candidate 2
        if len(candidates) > 1:
            x2, f2 = run_es(candidates[1][1], b2)
            if f2 < best_f_final:
                best_f_final = f2
                best_x_final = x2
        else:
            # If no second candidate, use new random
            x_new = rng.uniform(-5, 5, self.dim)
            x2, f2 = run_es(x_new, b2)
            if f2 < best_f_final:
                best_f_final = f2
                best_x_final = x2

        # Try candidate 3
        if len(candidates) > 2:
            x3, f3 = run_es(candidates[2][1], b3)
            if f3 < best_f_final:
                best_f_final = f3
                best_x_final = x3
        else:
            # If no third candidate, use new random
            x_new = rng.uniform(-5, 5, self.dim)
            x3, f3 = run_es(x_new, b3)
            if f3 < best_f_final:
                best_f_final = f3
                best_x_final = x3

        return best_x_final, best_f_final
