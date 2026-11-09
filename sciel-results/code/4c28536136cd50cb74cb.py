import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        time_limit = 325.0
        
        x_best = None
        f_best = np.inf
        eval_count = 0
        
        # 30 restarts, 10 evaluations each
        n_restarts = 30
        evals_per_restart = 10
        
        # Vary sigma across restarts: mix of large (exploration) and small (exploitation)
        # Use a mix: first few restarts with larger sigma, later with smaller
        sigmas = []
        for i in range(n_restarts):
            # Mix of sigmas: some large for exploration, some small for fine-tuning
            if i < 10:
                s = np.random.uniform(2.0, 4.0)  # Large steps
            elif i < 20:
                s = np.random.uniform(0.5, 2.0)  # Medium steps
            else:
                s = np.random.uniform(0.1, 0.5)  # Small steps for exploitation
            sigmas.append(s)
        
        for r in range(n_restarts):
            if time.time() - start_time > time_limit:
                break
            if eval_count >= budget:
                break
            
            # Start from a random point
            x = np.random.uniform(-5, 5, n)
            try:
                fx = func(x)
            except Exception:
                break
            eval_count += 1
            
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            
            sigma = sigmas[r]
            
            # 9 more evaluations for hill-climbing
            for step in range(evals_per_restart - 1):
                if time.time() - start_time > time_limit:
                    break
                if eval_count >= budget:
                    break
                
                # Generate perturbation
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                except Exception:
                    break
                eval_count += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < f_best:
                        f_best = fy
                        x_best = y.copy()
                # If not better, keep x as is (no sigma adaptation within restart)
        
        return x_best, f_best
