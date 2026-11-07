"""Combined optimizer for BBOB f12 (Bent cigar) using Nelder-Mead Simplex."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin

        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma_shrink = 0.5   # Shrink
        
        # Initial step size for simplex vertices
        initial_step = 1.0
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initial point: Origin is a good guess for BBOB
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        if f0 is None:
            return best_x, best_f

        # Construct initial simplex
        # x0 is the first vertex
        # Other vertices are x0 + initial_step * e_i
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += initial_step
            x_i = np.clip(x_i, -5, 5)
            f_i = evaluate(x_i)
            if f_i is None:
                break
            simplex.append(x_i)
            simplex_f.append(f_i)
            
        if calls >= budget:
            return best_x, best_f

        # Nelder-Mead Loop
        no_improve_count = 0
        max_no_improve = 40 # Restart if stuck for 40 iterations
        
        while calls < budget and time.time() < deadline:
            # Sort simplex by function value
            # We want lowest f first
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_point = simplex[-1]
            worst_f = simplex_f[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst_point)
            xr = np.clip(xr, -5, 5)
            fr = evaluate(xr)
            if fr is None:
                break
            
            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst_point)
                xe = np.clip(xe, -5, 5)
                fe = evaluate(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                no_improve_count = 0
            elif fr < simplex_f[-2]:
                # Reflection was good enough
                simplex[-1] = xr
                simplex_f[-1] = fr
                no_improve_count = 0
            else:
                # Contraction
                # Outside contraction if worst_f < fr, else inside
                if worst_f < fr:
                    # Outside contraction
                    xc = centroid + rho * (worst_point - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - worst_point)
                
                xc = np.clip(xc, -5, 5)
                fc = evaluate(xc)
                if fc is None:
                    break
                
                if fc < min(worst_f, fr):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                    no_improve_count = 0
                else:
                    # Shrink
                    # Move all points except best towards best
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_shrink = evaluate(simplex[i])
                        if f_shrink is None:
                            break
                        simplex_f[i] = f_shrink
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Restart around best point with smaller step
                        # This helps escape narrow valleys or local minima
                        best_x = simplex[0].copy()
                        best_f = simplex_f[0]
                        
                        # Reduce step size for polishing
                        initial_step *= 0.5
                        if initial_step < 1e-6:
                             break
                             
                        # Rebuild simplex
                        simplex = [best_x.copy()]
                        simplex_f = [best_f]
                        for i in range(dim):
                            x_i = best_x.copy()
                            x_i[i] += initial_step
                            x_i = np.clip(x_i, -5, 5)
                            f_i = evaluate(x_i)
                            if f_i is None:
                                break
                            simplex.append(x_i)
                            simplex_f.append(f_i)
                        no_improve_count = 0
                        if calls >= budget:
                            break

        return best_x, best_f
