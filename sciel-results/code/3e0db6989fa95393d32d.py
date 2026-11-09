import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        rng = np.random.default_rng(42)
        d = self.dim
        
        # Initialize Nelder-Mead simplex
        # Start at a random point, or use a few random points to pick a good start?
        # Nelder-Mead needs d+1 points.
        # Let's pick a random center and generate simplex around it.
        # To improve robustness, we can spend a few calls to find a good center.
        
        # Spend 10% of budget on initial random search to find a good start point
        n_init = min(10, self.budget // 10)
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            n_init += 1
            
        calls = n_init
        remaining_budget = self.budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f

        # Initialize simplex
        # Center is best_x
        # Initial step size
        step_size = 1.0
        simplex = np.zeros((d + 1, d))
        simplex[0] = best_x
        for i in range(1, d + 1):
            simplex[i] = best_x.copy()
            simplex[i, i-1] += step_size
            
        # Evaluate initial simplex
        f_values = np.zeros(d + 1)
        f_values[0] = best_f
        for i in range(1, d + 1):
            f_values[i] = func(simplex[i])
            calls += 1
            if f_values[i] < best_f:
                best_f = f_values[i]
                best_x = simplex[i].copy()
            
            if calls >= self.budget:
                return best_x, best_f

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_shrink = 0.5 # shrinkage
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Check convergence
            if np.std(f_values) < 1e-10:
                break
            
            # Worst point
            x_worst = simplex[-1]
            f_worst = f_values[-1]
            
            # Centroid of all but worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if calls >= self.budget:
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                break
            
            if f_r < f_values[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
                
            elif f_r < f_values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
            else:
                # Contraction
                # Inside contraction if f_r < f_worst, else outside
                if f_r < f_worst:
                    x_c = x_centroid + rho * (x_r - x_centroid)
                else:
                    x_c = x_centroid - rho * (x_worst - x_centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if calls >= self.budget:
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    break
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                else:
                    # Shrink
                    # Move all points except best towards best
                    x_best = simplex[0]
                    for i in range(1, d + 1):
                        simplex[i] = x_best + sigma_shrink * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls += 1
                        if f_values[i] < best_f:
                            best_f = f_values[i]
                            best_x = simplex[i].copy()
                        
                        if calls >= self.budget:
                            break
                    
                    if calls >= self.budget:
                        break

        return best_x, best_f
