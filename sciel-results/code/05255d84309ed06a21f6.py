"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with time and budget guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink

        # Initial simplex
        # Start at 0, or random? 0 is often good for centered functions.
        # f11 (Discus) is usually centered but shifted/rotated.
        # Random start within box might be safer if shift is large, but 0 is in [-4,4] usually.
        # Let's use a small random perturbation from 0 to avoid symmetry issues if any.
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-1, 1, dim)
        
        # Build simplex
        simplex = []
        f_values = []
        
        # First point
        x_curr = x0.copy()
        f_curr = func(x_curr)
        simplex.append(x_curr)
        f_values.append(f_curr)
        calls = 1
        
        # Other points
        for i in range(dim):
            x_new = x0.copy()
            # Standard Nelder-Mead initial step size is 0.05 or 0.25
            # Box is [-5, 5]. Let's use 0.5 step.
            step = 0.5
            if i == 0:
                x_new[i] += step
            else:
                x_new[i] += step
            # Clip to box
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            simplex.append(x_new)
            f_values.append(f_new)
            calls += 1
            
            if calls >= budget:
                break

        if calls >= budget:
            # Return best found
            idx = np.argmin(f_values)
            return simplex[idx], f_values[idx]

        # Nelder-Mead Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            # We maintain simplex and f_values sorted
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Check convergence (optional, but good for stopping)
            # If spread is small, stop
            if np.std(f_values) < 1e-8:
                break
                
            # Centroid of all but worst
            worst_idx = dim
            centroid = np.mean(simplex[:dim], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[worst_idx])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if calls >= budget:
                break
                
            if f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[worst_idx])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if calls >= budget:
                    break
                    
                if f_e < f_r:
                    simplex[worst_idx] = x_e
                    f_values[worst_idx] = f_e
                else:
                    simplex[worst_idx] = x_r
                    f_values[worst_idx] = f_r
            elif f_r < f_values[-2]:
                # Simple Reflection
                simplex[worst_idx] = x_r
                f_values[worst_idx] = f_r
            else:
                # Contraction
                # Outside contraction
                x_c = centroid + rho * (simplex[worst_idx] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if calls >= budget:
                    break
                    
                if f_c < f_values[worst_idx]:
                    simplex[worst_idx] = x_c
                    f_values[worst_idx] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

        # Find best
        idx = np.argmin(f_values)
        return simplex[idx], f_values[idx]
# EVOLVE-BLOCK-END
