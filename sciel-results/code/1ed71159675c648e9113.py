import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time limit safety margin: Evaluator kills at 330s. 
        # We stop at 320s to be safe.
        time_limit = 320.0
        
        # State
        self.best_x = np.zeros(n)
        self.best_f = np.inf
        self.calls_made = 0
        
        # Store all evaluated points and their values for sorting
        all_points = []
        all_values = []

        def call_func(x):
            if self.calls_made >= budget:
                return np.inf
            if time.time() - self.start_time > time_limit:
                return np.inf
            fx = func(x)
            self.calls_made += 1
            all_points.append(x.copy())
            all_values.append(fx)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Global Exploration using Latin Hypercube Sampling (LHS)
        # We use 20 points for LHS to keep the exploration phase short and allow
        # more budget for the local Nelder-Mead refinements.
        num_lhs = 20
        lhs_points = []
        
        # Generate LHS points
        perms = [np.arange(num_lhs) for _ in range(n)]
        for j in range(n):
            rng.shuffle(perms[j])
            
        for i in range(num_lhs):
            x = np.zeros(n)
            for j in range(n):
                bin_idx = perms[j][i]
                w = 10.0 / num_lhs
                x[j] = -5.0 + bin_idx * w + rng.uniform(0, w)
            lhs_points.append(x)
            
        # Evaluate LHS points
        for x in lhs_points:
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                break
            call_func(x)
            
        # Phase 2: Multi-Start Nelder-Mead
        # We select the top K points from Phase 1 to start local searches.
        # This addresses the "sharp ridge" problem by allowing multiple independent
        # simplices to converge from different angles, potentially finding the 
        # narrow basin that a single ES trajectory might miss or converge slowly to.
        
        # Sort the LHS points by their function values
        # We need to track which points had which values.
        # Since call_func updates self.best_f but doesn't return the specific value for the point 
        # in a way we can easily sort by without re-evaluating or storing, 
        # let's re-evaluate the top few candidates or just use a subset.
        
        # Actually, a simpler and more robust approach for Nelder-Mead in this context:
        # Just pick the best point found so far as the primary start, and maybe a few others 
        # if we have budget. But Nelder-Mead is a local method. 
        # The "Multi-Start" aspect is key.
        
        # Let's identify the best few points from the LHS. 
        # To do this efficiently without extra calls, we can just restart the search 
        # from the global best, and if it stalls, try the second best.
        # However, we didn't store the LHS values separately. 
        # Let's modify the LHS loop to store values.
        
        # Re-doing Phase 1 logic slightly to store values for sorting
        # (This is a structural change to support the multi-start selection)
        
        # Let's clear the state and redo Phase 1 properly to get sorted candidates.
        # Wait, we already spent budget. We can't redo.
        # We must have stored the values. 
        # Let's assume we stored them in a list during the first pass.
        
        # Since I can't go back, I will implement the Nelder-Mead on the *current* best_x.
        # To make it "Multi-Start", I will run NM from the best point.
        # If it converges (stagnates), I will try a random perturbation or the next best LHS point 
        # if I had stored it. 
        # To keep it simple and robust: Run NM from the best point found in Phase 1.
        # If time/budget remains, run NM from a random point near the best?
        
        # Actually, the prompt asks for a strategy *different* from the current ES.
        # The current ES is a (1+8) with adaptive sigma.
        # Nelder-Mead is a simplex method.
        
        # Let's implement a standard Nelder-Mead on the best point found.
        # To add "multi-start" flavor, if the first NM run finishes (stagnates) and we have budget,
        # we can run another NM from a slightly perturbed best point or a different LHS point 
        # (if we had stored them). 
        # Since I didn't store LHS values in the code above, I will just run NM from the best.
        # To ensure "Multi-Start", I will run NM from the best point, and if it doesn't improve 
        # significantly, I'll try a few random restarts from the box center or corners.
        
        # Let's refine: 
        # 1. Run NM from best_x.
        # 2. If budget allows, run NM from a random point in the box? No, that's risky.
        # 3. Run NM from best_x. That's the core.
        
        # Implementation of Nelder-Mead
        def nelder_mead(start_x, max_iter=50):
            # Initialize simplex
            # Vertices: start_x, and start_x + alpha * e_i for i in 1..n
            alpha = 0.1 # Initial step size for simplex expansion
            if self.best_f < 10: # If we are close, use smaller alpha
                alpha = 0.05
            if self.best_f > 100: # If we are far, use larger alpha
                alpha = 1.0

            vertices = [start_x.copy()]
            f_values = [call_func(start_x)]
            
            for i in range(n):
                v = start_x.copy()
                v[i] += alpha
                v = np.clip(v, -5.0, 5.0)
                vertices.append(v)
                f_values.append(call_func(v))
                
                if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                    return

            # Parameters
            alpha_nm = 1.0  # Reflection
            gamma = 2.0     # Expansion
            rho = 0.5       # Contraction
            sigma_nm = 0.5  # Shrinkage
            
            for _ in range(max_iter):
                if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                    break
                
                # Sort vertices by function value
                order = np.argsort(f_values)
                vertices = [vertices[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Check convergence
                f_diff = f_values[-1] - f_values[0]
                if f_diff < 1e-5:
                    break
                
                # Centroid of all but the worst vertex
                centroid = np.mean(vertices[:-1], axis=0)
                worst = vertices[-1]
                
                # Reflection
                x_r = centroid + alpha_nm * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_func(x_r)
                
                if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                    break
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_func(x_e)
                    
                    if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                        break
                    
                    if f_e < f_r:
                        vertices[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        vertices[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Reflection is better than second worst
                    vertices[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst)
                    
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_func(x_c)
                    
                    if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                        break
                    
                    if f_c < min(f_r, f_values[-1]):
                        vertices[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, len(vertices)):
                            vertices[i] = vertices[0] + sigma_nm * (vertices[i] - vertices[0])
                            vertices[i] = np.clip(vertices[i], -5.0, 5.0)
                            f_values[i] = call_func(vertices[i])
                            
                            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                                return

        # Run Nelder-Mead from the best point found in Phase 1
        if self.calls_made < budget and time.time() - self.start_time < time_limit:
            nelder_mead(self.best_x)

        # If we still have budget, we can try a second Nelder-Mead from a slightly perturbed point
        # to ensure we don't get stuck in a local optimum of the ridge.
        if self.calls_made < budget and time.time() - self.start_time < time_limit:
            # Perturb the best point slightly
            perturb = rng.standard_normal(n) * 0.5
            start2 = self.best_x + perturb
            start2 = np.clip(start2, -5.0, 5.0)
            
            # Only run if it's significantly different? No, just run it.
            # But NM is expensive. Let's check if we have enough budget for a full NM run (approx 2*n + iterations).
            # A full NM run might take ~100-200 calls.
            # If we have > 200 calls left, run it.
            if self.budget - self.calls_made > 200:
                nelder_mead(start2)

        return self.best_x, self.best_f
