import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Dense grid + random initialization
        # Katsuura has regular structure; sample densely
        initial_points = []
        
        # Coarse grid in [-5, 5]^dim with spacing ~1.0
        grid_vals = np.linspace(-5, 5, 6)  # 6 points per dim = 6^5 = 7776, too many
        # Use a sparser structured set + random
        for _ in range(20):
            initial_points.append(rng.uniform(-5, 5, dim))
        
        # Focused region where minimum lies: [-4, 4]^5
        for _ in range(30):
            initial_points.append(rng.uniform(-4, 4, dim))
        
        # Center
        initial_points.append(np.zeros(dim))
        
        # Evaluate initial points
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 15.0:
                break
            x0 = clip(x0)
            evaluate(x0)
        
        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # Phase 2: Nelder-Mead refinement from best point
        if calls_made < budget and remaining_time() > 15.0:
            local_budget = min(budget - calls_made, 200)
            local_calls = [0]
            
            def local_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_budget:
                    return best_f + 10.0
                return evaluate(x)
            
            try:
                res = minimize(local_func, best_x, method='Nelder-Mead',
                             options={'maxiter': local_budget, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = clip(res.x)
            except Exception:
                pass

        # Phase 3: Multiple (1+1)-ES restarts from diverse starts
        remaining_calls = budget - calls_made
        if remaining_calls > 30 and remaining_time() > 10.0:
            num_es = min(8, remaining_calls // 20)
            for i in range(num_es):
                if calls_made >= budget or remaining_time() < 5.0:
                    break
                
                if i == 0:
                    x = clip(best_x + 0.5 * rng.standard_normal(dim))
                elif i % 2 == 0:
                    x = clip(best_x + 1.5 * rng.standard_normal(dim))
                else:
                    x = clip(rng.uniform(-4, 4, dim))
                
                fx = evaluate(x)
                sigma = 0.3
                no_improve = 0
                max_steps = 20
                
                for _ in range(max_steps):
                    if calls_made >= budget or remaining_time() < 2.0:
                        break
                    y = clip(x + sigma * rng.standard_normal(dim))
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.05, 2.0)
                        no_improve = 0
                    else:
                        sigma *= 0.9
                        no_improve += 1
                    if no_improve >= 10:
                        break
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 4: Fine-tuning with adaptive step size
        if calls_made < budget and remaining_time() > 3.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            no_improve = 0
            max_no_improve = 15
            while calls_made < budget and remaining_time() > 1.0:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 5: High-precision polish using Nelder-Mead with tight tolerances
        if calls_made < budget - 10 and remaining_time() > 5.0:
            polish_budget = min(budget - calls_made, 50)
            polish_calls = [0]
            
            def polish_func(x):
                polish_calls[0] += 1
                if polish_calls[0] > polish_budget:
                    return best_f + 1e-6
                return evaluate(x)
            
            try:
                res = minimize(polish_func, best_x, method='Nelder-Mead',
                             options={'maxiter': polish_budget, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = clip(res.x)
            except Exception:
                pass

        return best_x, best_f
