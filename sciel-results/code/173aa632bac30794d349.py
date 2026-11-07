import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 285
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if self.calls >= self.budget or time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def nelder_mead(x0, max_calls):
            # Initialize simplex
            n = self.dim
            simplex = [x0.copy()]
            # Add vertices with small perturbation
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += 0.1 if x0[i] < 4.9 else -0.1
                simplex.append(x_i)
            
            f_vals = [wrapped_func(v) for v in simplex]
            
            # If any are inf, we are stuck or out of budget
            if np.any(np.isinf(f_vals)):
                return x0, min(f_vals)

            while self.calls < self.budget and time.time() < self.deadline and (max_calls - (self.calls - calls_before)) > 0:
                # calls_before needs to be tracked or passed
                pass # This logic is getting complex to track "local" budget easily without global state changes.
                # Simpler: Just run NM until global budget is low or local iteration limit is hit.
                # Let's use a local iteration counter.
                break # Placeholder for actual loop below

        # Better approach: Implement NM inline or as a helper that returns when local budget is exhausted.
        
        # Let's rewrite __call__ with a cleaner structure.
        
        def run_nm(x0, local_budget):
            n = self.dim
            simplex = [x0.copy()]
            for i in range(n):
                x_i = x0.copy()
                # Perturb slightly
                delta = 0.1
                if x0[i] + delta > 5:
                    x_i[i] -= delta
                else:
                    x_i[i] += delta
                simplex.append(x_i)
            
            f_vals = [wrapped_func(v) for v in simplex]
            if np.any(np.isinf(f_vals)):
                return x0, min(f_vals)

            iterations = 0
            max_iter = local_budget // (n + 1) # Rough estimate
            
            while iterations < max_iter and self.calls < self.budget and time.time() < self.deadline:
                iterations += 1
                # Sort simplex by f
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check convergence
                if f_vals[-1] - f_vals[0] < 1e-8:
                    break
                    
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_centroid + (x_centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                
                if np.isinf(f_r):
                    break
                    
                if f_r < f_vals[0]:
                    # Expansion
                    x_e = x_centroid + 2 * (x_centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if np.isinf(f_e):
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = x_centroid + 0.5 * (x_worst - x_centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)
                    if np.isinf(f_c):
                        break
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            f_vals[i] = wrapped_func(simplex[i])
                            if np.isinf(f_vals[i]):
                                break
                        if np.isinf(f_vals[i]):
                            break
                            
            # Return best point in simplex
            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx]

        def run_cd(x0, local_budget):
            x = x0.copy()
            f = wrapped_func(x)
            if np.isinf(f):
                return x, f
            
            # Simple coordinate descent with small steps
            # Use a few iterations of checking +/- step
            step = 0.1
            for _ in range(local_budget // 10): # Roughly 10 calls per coord pass
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                for i in range(self.dim):
                    if self.calls >= self.budget or time.time() > self.deadline:
                        break
                    # Try +
                    x_plus = x.copy()
                    x_plus[i] += step
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = wrapped_func(x_plus)
                    if not np.isinf(f_plus) and f_plus < f:
                        x = x_plus
                        f = f_plus
                    # Try -
                    x_minus = x.copy()
                    x_minus[i] -= step
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = wrapped_func(x_minus)
                    if not np.isinf(f_minus) and f_minus < f:
                        x = x_minus
                        f = f_minus
            return x, f

        # Strategy
        starts = [
            np.zeros(self.dim),
            np.random.uniform(-2, 2, self.dim),
            np.random.uniform(-4, 4, self.dim)
        ]
        
        budgets = [
            (150, 50), # NM, CD
            (100, 50),
            (50, 50)
        ]
        
        for i, x0 in enumerate(starts):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            nm_b, cd_b = budgets[i]
            x, f = run_nm(x0, nm_b)
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            x, f = run_cd(x, cd_b)
            
        return self.best_x, self.best_f
