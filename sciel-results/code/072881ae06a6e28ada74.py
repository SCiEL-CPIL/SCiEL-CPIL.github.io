import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        # 1. Global Scan - Reduced to 30 to free budget for refinement
        n_scan = 30
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for x in points:
            if time.time() > deadline - 5.0: break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        if best_x is None:
             x0 = rng.uniform(-5, 5, self.dim)
             return x0, func(x0)

        # 2. Local Refinement
        remaining = self.budget - len(candidates)
        if remaining <= 0:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        top_candidates = candidates[:3]
        
        # Allocate budget: 60% to best, 25% to second, 15% to third
        budget_best = int(remaining * 0.6)
        budget_second = int(remaining * 0.25)
        budget_third = remaining - budget_best - budget_second

        def run_es(start_x, start_f, n_evals):
            x = start_x.copy()
            fx = start_f
            sigma = 1.5  # Moderate initial sigma
            successes = 0
            total = 0
            stagnant = 0
            best_local_f = fx
            best_local_x = x.copy()
            
            for _ in range(n_evals):
                if time.time() > deadline - 2.0:
                    break
                # Generate neighbor
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    stagnant = 0
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local_x = x.copy()
                else:
                    stagnant += 1
                    
                total += 1
                if total >= 15:
                    success_rate = successes / total
                    # CMA-ES-like sigma adaptation
                    if success_rate > 0.35:
                        sigma *= 1.3
                    elif success_rate < 0.15:
                        sigma *= 0.75
                    else:
                        sigma *= 0.95
                    successes = 0
                    total = 0
                    
                # If stuck, try a bigger jump to escape local minima
                if stagnant > 30:
                    # Try multiple random restarts from current best
                    for _ in range(3):
                        y = np.clip(best_local_x + rng.standard_normal(self.dim) * 1.5, -5, 5)
                        fy = func(y)
                        if fy < best_local_f:
                            best_local_f = fy
                            best_local_x = y.copy()
                            x = y.copy()
                            fx = fy
                    sigma = 1.5
                    stagnant = 0

            return best_local_x, best_local_f

        # Refine best
        x_best, f_best = run_es(best_x, best_f, budget_best)
        if f_best < best_f:
             best_x, best_f = x_best, f_best
             
        # Refine second best if exists and budget remains
        if len(top_candidates) > 1:
             x_second, f_second = top_candidates[1]
             if budget_second > 0:
                 x_s, f_s = run_es(x_second, f_second, budget_second)
                 if f_s < best_f:
                     best_x, best_f = x_s, f_s

        # Refine third best if exists and budget remains
        if len(top_candidates) > 2:
             x_third, f_third = top_candidates[2]
             if budget_third > 0:
                 x_t, f_t = run_es(x_third, f_third, budget_third)
                 if f_t < best_f:
                     best_x, best_f = x_t, f_t

        # 3. Final Polish Phase - adaptive small steps to refine on narrow peaks
        polish_budget = min(80, max(20, int(remaining * 0.15)))
        if time.time() < deadline - 5.0:
            polish_sigma = 0.08
            for i in range(polish_budget):
                if time.time() > deadline - 2.0:
                    break
                y = np.clip(best_x + polish_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                    polish_sigma *= 0.98  # Slowly shrink when improving
                else:
                    polish_sigma *= 0.97  # Slightly faster shrink when not improving

        return best_x, best_f
