"""Nelder-Mead optimizer for Linear Slope (f5)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Nelder-Mead direct search.
    This approach avoids explicit gradient estimation and instead uses
    a simplex that moves toward the optimum. For linear functions, this
    should provide steady improvements in the objective value, which is
    beneficial for the AOCC score.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink

        # Initialize simplex
        # Start at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        
        simplex = [x0]
        f_simplex = [f0]
        
        best_x = x0.copy()
        best_fx = f0

        # Create initial simplex points
        # Use a small step size for initial simplex to stay within bounds
        # and to allow for local exploration
        step = 0.5 
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            x_i = x0.copy()
            x_i[i] += step
            x_i = np.clip(x_i, -5, 5)
            f_i = func(x_i)
            calls += 1
            
            simplex.append(x_i)
            f_simplex.append(f_i)
            
            if f_i < best_fx:
                best_fx = f_i
                best_x = x_i.copy()

        # Nelder-Mead loop
        while calls < budget and time.time() < deadline:
            # Sort simplex by function value
            indices = np.argsort(f_simplex)
            simplex = [simplex[i] for i in indices]
            f_simplex = [f_simplex[i] for i in indices]
            
            # Check for convergence (optional, but good for early exit if stuck)
            # If the spread is very small, we might be at the optimum (or a local one)
            # For linear functions, we should reach a corner.
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                    if f_e < best_fx:
                        best_fx = f_e
                        best_x = x_e.copy()
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                    if f_r < best_fx:
                        best_fx = f_r
                        best_x = x_r.copy()
            elif f_r < f_simplex[-2]:
                # Reflection is good
                simplex[-1] = x_r
                f_simplex[-1] = f_r
                if f_r < best_fx:
                    best_fx = f_r
                    best_x = x_r.copy()
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < f_simplex[-1]:
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                    if f_c < best_fx:
                        best_fx = f_c
                        best_x = x_c.copy()
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = func(simplex[i])
                        calls += 1
                        if f_simplex[i] < best_fx:
                            best_fx = f_simplex[i]
                            best_x = simplex[i].copy()
                        
                        if calls >= budget or time.time() > deadline:
                            break
                    
                    if calls >= budget or time.time() > deadline:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
