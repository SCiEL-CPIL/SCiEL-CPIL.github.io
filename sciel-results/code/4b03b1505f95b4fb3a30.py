"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        calls = 0
        
        # Initialize simplex around origin
        x0 = np.zeros(self.dim)
        alpha = 0.5
        
        simplex = [x0.copy()]
        f_simplex = [func(x0)]
        calls += 1
        
        for i in range(self.dim):
            x_i = x0.copy()
            x_i[i] += alpha
            f_i = func(x_i)
            simplex.append(x_i)
            f_simplex.append(f_i)
            calls += 1
            if time.time() > deadline:
                break

        best_x = simplex[0]
        best_f = f_simplex[0]
        
        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma = 0.5
        shrink = 0.5
        
        while calls < self.budget and time.time() < deadline:
            # Sort simplex by function value
            indices = np.argsort(f_simplex)
            simplex = [simplex[i] for i in indices]
            f_simplex = [f_simplex[i] for i in indices]
            
            x_best = simplex[0]
            f_best = f_simplex[0]
            x_worst = simplex[-1]
            f_worst = f_simplex[-1]
            x_second_worst = simplex[-2]
            f_second_worst = f_simplex[-2]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + gamma * (centroid - x_worst)
            f_r = func(x_r)
            calls += 1
            if time.time() > deadline:
                break
                
            if f_r < f_best:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                f_e = func(x_e)
                calls += 1
                if time.time() > deadline:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_second_worst:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + sigma * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - sigma * (x_worst - centroid)
                    
                f_c = func(x_c)
                calls += 1
                if time.time() > deadline:
                    break
                    
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = x_best + shrink * (simplex[i] - x_best)
                        f_simplex[i] = func(simplex[i])
                        calls += 1
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
            
            # Update best
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = simplex[0]
                
        return best_x, best_f
# EVOLVE-BLOCK-END
