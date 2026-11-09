"""Nelder-Mead Simplex Optimization for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Initial simplex setup
        # Start at the center of the box
        x0 = np.zeros(d)
        
        # Initial step size for simplex vertices
        # The box is [-5, 5]. A step of 1.0 is reasonable.
        # If the function is ill-conditioned, a smaller step might be safer initially,
        # but we need to explore.
        step_size = 1.0
        
        calls = 0
        best_x = x0.copy()
        best_f = float('inf')
        
        # Helper to create simplex around a center
        def create_simplex(center, size):
            vertices = [center.copy()]
            for i in range(d):
                v = center.copy()
                v[i] += size
                vertices.append(v)
            return vertices
        
        # Evaluate a point, respecting budget and time
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                return None
            if calls >= budget:
                return None
            
            # func clips x to [-5, 5]
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx
        
        # Main Loop
        # We will run Nelder-Mead, and if it stalls, restart locally.
        # Since Nelder-Mead is local, and the landscape is shifted/rotated,
        # a single run from 0 might miss the global minimum if it's far away.
        # However, the task says "minimizer lies in [-4, 4]^5".
        # 0 is in [-4, 4]^5.
        
        # To improve robustness, we perform a "Multi-Restart Nelder-Mead"
        # but keeping the "local search" spirit by restarting near the best found so far.
        
        # Initial Simplex
        simplex = create_simplex(x0, step_size)
        
        # Evaluate initial simplex
        fvals = []
        for v in simplex:
            fv = evaluate(v)
            if fv is None:
                break
            fvals.append(fv)
        
        if len(fvals) < d + 1:
            return best_x, best_f
            
        # Sort initial simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        stall_count = 0
        max_stall = 50
        restart_count = 0
        max_restarts = 10 # Limit restarts to avoid wasting budget
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check for convergence/stall
            if fvals[0] == fvals[-1]:
                break
                
            # If no improvement in last 'max_stall' iterations, restart
            if stall_count >= max_stall:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                
                # Restart around the current best point with a smaller step
                # Or a random perturbation?
                # Let's try a smaller step first to refine, then if it fails again, maybe larger?
                # For ill-conditioned, smaller is often better once close.
                new_step = step_size * 0.5
                if new_step < 1e-6:
                    break
                step_size = new_step
                
                simplex = create_simplex(best_x, step_size)
                fvals = []
                for v in simplex:
                    fv = evaluate(v)
                    if fv is None:
                        break
                    fvals.append(fv)
                if len(fvals) < d + 1:
                    break
                stall_count = 0
                continue
            
            # Nelder-Mead Step
            x_worst = simplex[-1]
            x_second_worst = simplex[-2]
            x_best = simplex[0]
            
            # Centroid of all points except worst
            centroid = np.sum(simplex[:-1], axis=0) / d
            
            # 1. Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            f_ref = evaluate(x_ref)
            
            if f_ref is None:
                break
                
            if f_ref < fvals[0]:
                # 2. Expansion
                x_exp = centroid + gamma * (centroid - x_worst)
                f_exp = evaluate(x_exp)
                
                if f_exp is None:
                    break
                    
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                stall_count = 0 # Progress made
            elif f_ref < fvals[-2]:
                # Accept Reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
                stall_count = 0
            else:
                # 3. Contraction
                # Outside contraction if f_ref < f_worst
                if f_ref < fvals[-1]:
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (centroid - x_worst)
                
                f_con = evaluate(x_con)
                
                if f_con is None:
                    break
                
                if f_con < min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                    stall_count = 0
                else:
                    # 4. Shrink
                    for i in range(1, d + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        fvals[i] = evaluate(simplex[i])
                        if fvals[i] is None:
                            break
                    if fvals[-1] is None:
                        break
                    stall_count += 1
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
