import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Deadline guard
        deadline = self.start_time + 280  # 20s margin
        
        # State
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # Parameters for Nelder-Mead
        # We will implement a simple Nelder-Mead or use scipy?
        # Using scipy.optimize.minimize is safer and faster to write.
        # But we need to count function calls strictly.
        # SciPy's minimize doesn't easily allow strict call counting in all versions without a wrapper.
        # Let's write a custom Nelder-Mead to be safe and control the budget.
        
        def nelder_mead(func, x0, budget, rng, limit_calls):
            # Simplex initialization
            # x0 is the center
            # Standard deviation for initialization
            sigma_init = 0.5 
            simplex = [x0]
            for i in range(self.dim):
                x_i = x0.copy()
                x_i[i] += sigma_init
                simplex.append(x_i)
            
            f_values = [func(x) for x in simplex]
            # Track calls
            # The wrapper func should increment a counter.
            
            # Nelder-Mead steps
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Sort simplex
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            while limit_calls > 0:
                # Check time
                if time.time() > limit_calls[0]: # passing deadline in limit_calls? No, pass separately
                    break
                
                # Reflection
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                
                f_r = func(x_r)
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Accept Reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_values[i] = func(simplex[i])
                
                # Re-sort
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                limit_calls[0] -= 1 # This logic is messy. Let's use a counter object.
            
            return simplex[0], f_values[0]

        # This custom NM is prone to bugs. 
        # Let's use a simpler, robust approach: 
        # **Multi-start (1+1)-ES** is actually quite strong if tuned.
        # The previous observation said multi-start ES scored 0.67. 
        # Maybe the baseline (1+1)-ES is just better because it spends all budget on one good trajectory?
        # But (1+1)-ES is very sensitive to initialization.
        
        # Let's try **Powell's Method**. It is robust and derivative-free.
        # Or **Nelder-Mead** from SciPy with a strict call wrapper.
        
        # I will implement a strict wrapper around `scipy.optimize.minimize` with `method='Nelder-Mead'`.
        
        call_count = 0
        best_x_global = None
        best_f_global = float('inf')
        
        def wrapped_func(x):
            nonlocal call_count, best_x_global, best_f_global
            if call_count >= self.budget:
                return best_f_global
            call_count += 1
            fx = func(x)
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
            return fx
        
        # Strategy: 5 Restarts of Nelder-Mead
        # Budget per restart: 100
        # If a restart stagnates, stop early and start a new one.
        
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        for r in range(n_restarts):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Random start
            x0 = rng.uniform(-4, 4, self.dim) # Minimizer is in [-4, 4]
            
            # Run Nelder-Mead
            # We need to limit calls to budget_per_restart
            remaining_budget = self.budget - call_count
            current_budget = min(budget_per_restart, remaining_budget)
            
            # SciPy Nelder-Mead options
            # maxiter is number of iterations, not function calls.
            # In 5D, 1 iteration is ~6 calls.
            # 100 calls / 6 = 16 iterations.
            # This might be too few for NM to converge.
            # NM is slow.
            
            # Let's stick to a custom, simple **(1+1)-ES** but with **Multiple Restarts** and **Stagnation**.
            # The baseline is (1+1)-ES.
            # If I do 5 restarts of (1+1)-ES, I get 5 chances to find the global min.
            # Each restart gets 100 calls.
            # (1+1)-ES is very fast. 100 calls is enough to converge to a local min.
            
            x = x0
            fx = wrapped_func(x)
            sigma = 1.0
            
            stagnation_counter = 0
            local_best_f = fx
            local_best_x = x.copy()
            
            for i in range(current_budget):
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1 # Slight increase
                    stagnation_counter = 0
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()
                else:
                    sigma *= 0.9
                    stagnation_counter += 1
                    
                # Stagnation check: if no improvement in 30 steps, restart
                if stagnation_counter > 30:
                    break
                    
            # Update global best (already done in wrapped_func)
            
        return best_x_global, best_f_global
