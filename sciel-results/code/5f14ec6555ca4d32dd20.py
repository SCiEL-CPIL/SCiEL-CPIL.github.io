"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Nelder-Mead optimizer for anisotropic functions like Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        budget = self.budget
        dim = self.dim

        calls = 0
        best_fx = np.inf
        best_x = np.zeros(dim)

        def safe_func(x):
            nonlocal calls, best_fx, best_x
            if calls >= budget:
                return np.inf
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        def nelder_mead(simplex, f_values, max_calls, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, 
                        tol=1e-12, min_step=1e-14):
            """Run Nelder-Mead with adaptive step sizing."""
            nonlocal calls, best_fx, best_x
            n = len(simplex)
            iterations = 0
            max_iterations = max_calls // 2
            step_scale = 1.0
            
            while calls < budget - n and iterations < max_iterations:
                if time.time() > deadline:
                    break
                
                iterations += 1

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check convergence - use relative tolerance
                f_range = abs(f_values[-1] - f_values[0])
                f_scale = max(abs(f_values[0]), 1e-10)
                if f_range < tol * f_scale:
                    break

                x_worst = simplex[-1]
                x_best = simplex[0]
                
                # Centroid of all points except the worst
                x_centroid = np.mean(simplex[:-1], axis=0)

                # 1. Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = safe_func(x_ref)

                if f_values[0] <= f_ref < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                    step_scale = max(0.5, step_scale * 0.9)
                elif f_ref < f_values[0]:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    f_exp = safe_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                    step_scale = min(2.0, step_scale * 1.2)
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        x_con = x_centroid + rho * (x_ref - x_centroid)
                    else:
                        # Inside contraction
                        x_con = x_centroid - rho * (x_worst - x_centroid)
                    
                    f_con = safe_func(x_con)
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                        step_scale = max(0.3, step_scale * 0.7)
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_values[i] = safe_func(simplex[i])
                        step_scale = max(0.1, step_scale * 0.5)
                        
                        # If simplex is too small, break
                        simplex_range = np.max(np.array(simplex), axis=0) - np.min(np.array(simplex), axis=0)
                        if np.max(simplex_range) < min_step:
                            break

            return simplex, f_values

        # Phase 1: Coarse search with larger initial steps
        simplex = []
        f_values = []
        
        # Better initialization: sample multiple points to find a good starting region
        # Start with a grid-like initialization
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        simplex.append(x0.copy())
        f_values.append(f0)

        # Create vertices with larger initial step for Bent Cigar (anisotropic)
        initial_step = 2.0
        for i in range(dim):
            xi = np.zeros(dim)
            xi[i] = initial_step
            fi = safe_func(xi)
            simplex.append(xi)
            f_values.append(fi)

        # Run coarse Nelder-Mead
        simplex, f_values = nelder_mead(simplex, f_values, budget // 2, tol=1e-8, min_step=1e-10)
        
        # Phase 2: Fine polish with tighter tolerances
        if calls < budget - (dim + 1) and time.time() < deadline - 10:
            # Reinitialize simplex around best point found
            x_best = best_x.copy()
            f_best = best_fx
            
            # Create small simplex around best point
            polish_step = 0.1
            simplex2 = [x_best.copy()]
            f_values2 = [f_best]
            
            for i in range(dim):
                xi = x_best.copy()
                xi[i] += polish_step
                fi = safe_func(xi)
                simplex2.append(xi)
                f_values2.append(fi)
            
            # Run fine Nelder-Mead
            simplex2, f_values2 = nelder_mead(simplex2, f_values2, budget - calls, tol=1e-14, min_step=1e-15)
            
            # Update best if improved
            if f_values2[0] < best_fx:
                order = np.argsort(f_values2)
                best_x = simplex2[order[0]].copy()
                best_fx = f_values2[order[0]]
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
