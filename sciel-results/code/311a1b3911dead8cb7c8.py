"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        # Initialize Nelder-Mead simplex at center of box
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        
        # Create simplex: x0 plus small perturbations along each axis
        simplex = [x0.copy()]
        func_vals = [f0]
        
        step_size = 0.5
        for i in range(self.dim):
            xi = x0.copy()
            xi[i] += step_size
            simplex.append(xi)
            fi = func(xi)
            func_vals.append(fi)
        
        best_x = x0.copy()
        best_f = f0
        
        # Nelder-Mead parameters
        alpha = 0.5   # reflection
        gamma = 1.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        calls_made = self.dim + 1  # initial simplex evaluations
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(func_vals)
            simplex = [simplex[i].copy() for i in order]
            func_vals = [func_vals[i] for i in order]
            
            # Check convergence
            if abs(func_vals[-1] - func_vals[0]) < 1e-12:
                break
            
            # Best and worst points
            x_best = simplex[0]
            f_best = func_vals[0]
            x_worst = simplex[-1]
            f_worst = func_vals[-1]
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            
            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls_made += 1
            
            if f_ref < func_vals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls_made += 1
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    func_vals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    func_vals[-1] = f_ref
            elif f_ref < func_vals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                func_vals[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_worst:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_ref - centroid)
                
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls_made += 1
                
                if f_con <= min(f_ref, f_worst):
                    simplex[-1] = x_con
                    func_vals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        func_vals[i] = func(simplex[i])
                        calls_made += 1
            
            # Check if we're still making progress
            if calls_made >= self.budget:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
