"""Optimizer for BBOB-style f6 (Attractive sector)."""
import numpy as np
import time

class Optimizer:
    """Nelder-Mead optimizer with restarts for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def check_time():
            return time.time() > deadline

        # Initial simplex centered at origin with spread
        # f6 minimum is typically at 0 in canonical form
        # We use a spread of 2.0 to cover a reasonable area initially
        init_spread = 2.0
        
        # Generate initial simplex
        # Simplex has dim+1 points
        simplex = np.zeros((self.dim + 1, self.dim))
        for i in range(self.dim + 1):
            simplex[i] = rng.uniform(-init_spread, init_spread, self.dim)
        
        # Evaluate initial simplex
        simplex_f = np.zeros(self.dim + 1)
        for i in range(self.dim + 1):
            if check_time():
                break
            x_i = np.clip(simplex[i], -5, 5)
            simplex_f[i] = func(x_i)
            calls += 1
            if simplex_f[i] < best_f:
                best_f = simplex_f[i]
                best_x = x_i.copy()
                
        if calls >= self.budget or check_time():
            return best_x, best_f

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrinkage
        
        # Stagnation detection
        prev_best_f = best_f
        no_improve_calls = 0
        restart_threshold = 50 # Restart if no improvement for 50 calls
        
        while calls < self.budget:
            if check_time():
                break
            
            # Sort simplex by function value
            idx = np.argsort(simplex_f)
            simplex = simplex[idx]
            simplex_f = simplex_f[idx]
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = simplex_f[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
                no_improve_calls = 0
            else:
                no_improve_calls += 1

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                    no_improve_calls = 0
                else:
                    no_improve_calls += 1
                    
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < worst_f:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (worst - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                    no_improve_calls = 0
                else:
                    no_improve_calls += 1
                    
                if f_c < min(f_r, worst_f):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()
                            no_improve_calls = 0
                        else:
                            no_improve_calls += 1
                        
                        if check_time():
                            break
                    if check_time():
                        break

            # Check for restart
            if no_improve_calls >= restart_threshold:
                # Restart with a new simplex around the best point found
                # Use a smaller spread to explore locally, or a larger one to escape
                # Let's try a mix: 50% chance local, 50% chance global-ish
                if rng.random() < 0.5:
                    new_spread = 0.5
                else:
                    new_spread = 2.0
                    
                simplex = np.zeros((self.dim + 1, self.dim))
                for i in range(self.dim + 1):
                    simplex[i] = best_x + new_spread * rng.standard_normal(self.dim)
                    simplex[i] = np.clip(simplex[i], -5, 5)
                
                simplex_f = np.zeros(self.dim + 1)
                for i in range(self.dim + 1):
                    if check_time():
                        break
                    simplex_f[i] = func(simplex[i])
                    calls += 1
                    if simplex_f[i] < best_f:
                        best_f = simplex_f[i]
                        best_x = simplex[i].copy()
                
                no_improve_calls = 0
                if calls >= self.budget or check_time():
                    break

        return best_x, best_f
