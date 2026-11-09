"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex Method with adaptive step size and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # 1. Initialize Simplex
        # Center at origin, as minimizer is in [-4, 4]^5
        x0 = np.zeros(dim)
        
        # Simplex points: x0 and x0 + delta * e_i
        # Start with larger delta for faster initial convergence
        delta = 1.0
        simplex = [x0]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
            
        # Ensure points are within bounds (though func clips, better to start valid)
        simplex = [np.clip(x, -5.0, 5.0) for x in simplex]
        
        # Evaluate initial simplex
        # We need dim + 1 calls
        if budget < dim + 1:
            # Fallback: just return center if no budget for simplex
            fx = func(x0)
            return x0, fx

        fx_values = []
        call_count = 0
        best_x = x0.copy()
        best_f = np.inf
        for x in simplex:
            fx = func(x)
            fx_values.append(fx)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort simplex by function value
        def sort_simplex(simplex, fx_values):
            order = np.argsort(fx_values)
            return [simplex[i] for i in order], [fx_values[i] for i in order]
            
        simplex, fx_values = sort_simplex(simplex, fx_values)
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        # Adaptive step size tracking
        step_scale = 1.0
        stagnant_count = 0
        
        # Main Loop
        while call_count < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Identify best, second best, and worst
            x_best = simplex[0]
            f_best = fx_values[0]
            x_second = simplex[1]
            f_second = fx_values[1]
            x_worst = simplex[-1]
            f_worst = fx_values[-1]
            
            # Track best globally
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            
            # Check convergence: if function values are very close
            # Use relative tolerance for Sphere
            if f_worst - f_best < 1e-12 * max(1.0, abs(f_best)):
                break
            
            # Check if simplex is too small in space
            simplex_span = np.max(np.linalg.norm(simplex[0] - x, axis=-1) for x in simplex[1:])
            if simplex_span < 1e-8:
                break
            
            # Centroid of all points except the worst
            centroid = np.sum(simplex[:-1], axis=0) / dim
            
            # 1. Reflection
            x_reflected = centroid + alpha * (centroid - x_worst)
            x_reflected = np.clip(x_reflected, -5.0, 5.0)
            
            f_reflected = func(x_reflected)
            call_count += 1
            
            # Update best if reflected is better
            if f_reflected < f_best:
                f_best = f_reflected
                x_best = x_reflected
                if f_reflected < best_f:
                    best_f = f_reflected
                    best_x = x_reflected.copy()
            
            if f_reflected < f_second:
                # Reflection is better than second worst
                # Replace worst with reflected
                simplex[-1] = x_reflected
                fx_values[-1] = f_reflected
                simplex, fx_values = sort_simplex(simplex, fx_values)
                stagnant_count = 0
                continue
            
            elif f_reflected < f_worst:
                # Reflection is better than worst, but worse than second
                # Replace worst with reflected
                simplex[-1] = x_reflected
                fx_values[-1] = f_reflected
                simplex, fx_values = sort_simplex(simplex, fx_values)
                stagnant_count += 1
                continue
            
            else:
                # Reflection is worse than or equal to worst
                # 2. Contraction
                x_contracted = centroid + rho * (x_worst - centroid)
                x_contracted = np.clip(x_contracted, -5.0, 5.0)
                
                f_contracted = func(x_contracted)
                call_count += 1
                
                if f_contracted < f_worst:
                    # Contraction is better than worst
                    simplex[-1] = x_contracted
                    fx_values[-1] = f_contracted
                    simplex, fx_values = sort_simplex(simplex, fx_values)
                    stagnant_count += 1
                    continue
                
                else:
                    # 3. Shrinkage
                    # Move all points (except best) closer to best
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        
                        # If budget allows, evaluate new positions
                        if call_count < budget:
                            fx_new = func(simplex[i])
                            fx_values[i] = fx_new
                            call_count += 1
                            if fx_new < best_f:
                                best_f = fx_new
                                best_x = simplex[i].copy()
                        else:
                            break
                    
                    # Sort after shrinkage
                    simplex, fx_values = sort_simplex(simplex, fx_values)
                    stagnant_count += 1
                    
                    # If we broke out of inner loop due to budget, outer loop will also break
                    if call_count >= budget:
                        break
            
            # Adaptive step: if stagnating, reduce step size
            if stagnant_count >= 5:
                step_scale *= 0.5
                stagnant_count = 0
                # Re-center simplex around best with smaller steps
                for i in range(1, dim + 1):
                    simplex[i] = x_best + step_scale * 0.1 * (simplex[i] - x_best)
                    simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                    if call_count < budget:
                        fx_new = func(simplex[i])
                        fx_values[i] = fx_new
                        call_count += 1
                        if fx_new < best_f:
                            best_f = fx_new
                            best_x = simplex[i].copy()
                    else:
                        break
                simplex, fx_values = sort_simplex(simplex, fx_values)
                if call_count >= budget:
                    break

        # Final polish: use remaining budget for fine-tuning around best_x
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            # Coordinate descent polish
            step = 0.01
            for _ in range(min(remaining, 50)):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    # Try positive step
                    x_try = best_x.copy()
                    x_try[i] += step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = func(x_try)
                    call_count += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        improved = True
                        continue
                    # Try negative step
                    if call_count >= budget or time.time() > deadline:
                        break
                    x_try = best_x.copy()
                    x_try[i] -= step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = func(x_try)
                    call_count += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        improved = True
                
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
        
        # Return the best point found
        return best_x, best_f
# EVOLVE-BLOCK-END
