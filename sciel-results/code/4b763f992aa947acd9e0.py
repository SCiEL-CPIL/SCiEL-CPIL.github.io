import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Track calls to ensure we don't exceed budget
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        
        # Time guard
        start_time = time.time()
        time_limit = 290.0 # Safety margin
        
        def safe_eval(x):
            if self.calls >= self.budget:
                return self.best_f # Should not happen if logic is correct
            if time.time() - start_time > time_limit:
                # Return a bad value to signal stop? 
                # Actually, we should just stop the loops.
                # This is tricky to handle cleanly inside a helper.
                # Better to check time in the main loop.
                pass
            self.calls += 1
            return func(x)

        # 1. Generate initial diverse points
        # Use a few random points to start
        n_starts = 5
        rng = np.random.default_rng(42)
        
        # Allocate budget: 
        # Phase 1: Scout with n_starts points, ~10% budget each? 
        # Total 500. 5 starts. 50 calls each = 250 calls.
        # Phase 2: Best point gets remaining 250 calls.
        
        scout_budget = max(10, self.budget // (2 * n_starts)) # e.g., 50
        refine_budget = self.budget - (n_starts * scout_budget)
        
        best_f_overall = np.inf
        best_x_overall = None
        second_best_f = np.inf
        second_best_x = None
        
        for i in range(n_starts):
            # Initialize simplex around a random point
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Create initial simplex
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x0
            for j in range(1, self.dim + 1):
                simplex[j] = x0.copy()
                simplex[j][j-1] += 0.5 # Small perturbation
            
            # Evaluate initial simplex
            f_values = np.zeros(self.dim + 1)
            for j in range(self.dim + 1):
                # Check time and budget
                if time.time() - start_time > time_limit or self.calls >= self.budget:
                    break
                f_values[j] = safe_eval(simplex[j])
                
                if f_values[j] < self.best_f:
                    self.best_f = f_values[j]
                    self.best_x = simplex[j].copy()
                
            # Run Nelder-Mead for scout_budget iterations
            for k in range(scout_budget):
                if time.time() - start_time > time_limit or self.calls >= self.budget:
                    break
                    
                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Check convergence
                if f_values[0] - f_values[-1] < 1e-8:
                    break
                    
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fr = safe_eval(np.clip(xr, -5, 5))
                
                if f_values[0] <= fr < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = np.clip(xr, -5, 5)
                    f_values[-1] = fr
                elif fr < f_values[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = safe_eval(np.clip(xe, -5, 5))
                    if fe < fr:
                        simplex[-1] = np.clip(xe, -5, 5)
                        f_values[-1] = fe
                    else:
                        simplex[-1] = np.clip(xr, -5, 5)
                        f_values[-1] = fr
                else:
                    # Contraction
                    # Outside contraction if fr < f_values[-1]
                    if fr < f_values[-1]:
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fc = safe_eval(np.clip(xc, -5, 5))
                        if fc <= fr:
                            simplex[-1] = np.clip(xc, -5, 5)
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, self.dim + 1):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                f_values[j] = safe_eval(simplex[j])
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        fc = safe_eval(np.clip(xc, -5, 5))
                        if fc < f_values[-1]:
                            simplex[-1] = np.clip(xc, -5, 5)
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, self.dim + 1):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                f_values[j] = safe_eval(simplex[j])
                
                # Update global best
                if f_values[0] < self.best_f:
                    self.best_f = f_values[0]
                    self.best_x = simplex[0].copy()
            
            # Record best from this start
            local_best_f = np.min(f_values)
            local_best_x = simplex[np.argmin(f_values)].copy()
            
            if local_best_f < best_f_overall:
                second_best_f = best_f_overall
                second_best_x = best_x_overall
                best_f_overall = local_best_f
                best_x_overall = local_best_x
            elif local_best_f < second_best_f:
                second_best_f = local_best_f
                second_best_x = local_best_x

        # Phase 2: Refine the best point found
        if best_x_overall is not None and self.calls < self.budget and time.time() - start_time < time_limit:
            # Initialize simplex around best_x_overall
            # Use a slightly larger simplex if we want to explore around it, 
            # or smaller if we think it's close to optimum.
            # Let's use a moderate size.
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = best_x_overall
            for j in range(1, self.dim + 1):
                simplex[j] = best_x_overall.copy()
                simplex[j][j-1] += 0.1 # Smaller perturbation for refinement
            
            f_values = np.zeros(self.dim + 1)
            for j in range(self.dim + 1):
                if self.calls >= self.budget or time.time() - start_time > time_limit:
                    break
                f_values[j] = safe_eval(simplex[j])
                
                if f_values[j] < self.best_f:
                    self.best_f = f_values[j]
                    self.best_x = simplex[j].copy()
            
            # Run Nelder-Mead for remaining budget
            remaining_budget = self.budget - self.calls
            for k in range(remaining_budget):
                if time.time() - start_time > time_limit or self.calls >= self.budget:
                    break
                    
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                if f_values[0] - f_values[-1] < 1e-8:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                xr = centroid + (centroid - simplex[-1])
                fr = safe_eval(np.clip(xr, -5, 5))
                
                if f_values[0] <= fr < f_values[-2]:
                    simplex[-1] = np.clip(xr, -5, 5)
                    f_values[-1] = fr
                elif fr < f_values[0]:
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = safe_eval(np.clip(xe, -5, 5))
                    if fe < fr:
                        simplex[-1] = np.clip(xe, -5, 5)
                        f_values[-1] = fe
                    else:
                        simplex[-1] = np.clip(xr, -5, 5)
                        f_values[-1] = fr
                else:
                    if fr < f_values[-1]:
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fc = safe_eval(np.clip(xc, -5, 5))
                        if fc <= fr:
                            simplex[-1] = np.clip(xc, -5, 5)
                            f_values[-1] = fc
                        else:
                            for j in range(1, self.dim + 1):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                f_values[j] = safe_eval(simplex[j])
                    else:
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        fc = safe_eval(np.clip(xc, -5, 5))
                        if fc < f_values[-1]:
                            simplex[-1] = np.clip(xc, -5, 5)
                            f_values[-1] = fc
                        else:
                            for j in range(1, self.dim + 1):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                f_values[j] = safe_eval(simplex[j])
                
                if f_values[0] < self.best_f:
                    self.best_f = f_values[0]
                    self.best_x = simplex[0].copy()

        return self.best_x, self.best_f
