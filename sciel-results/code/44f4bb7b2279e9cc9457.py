import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        dim = self.dim
        budget = self.budget
        calls_used = 0

        # Phase 1: Determine the optimal vertex using axis probes
        # Evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_used += 1
        update_best(x0, f0)

        signs = np.zeros(dim)
        step = 5.0

        for i in range(dim):
            if time.time() > deadline:
                break
            
            # Probe positive direction
            x_plus = np.zeros(dim)
            x_plus[i] = step
            f_plus = func(x_plus)
            calls_used += 1
            update_best(x_plus, f_plus)
            
            # Probe negative direction
            x_minus = np.zeros(dim)
            x_minus[i] = -step
            f_minus = func(x_minus)
            calls_used += 1
            update_best(x_minus, f_minus)
            
            # Determine sign of gradient component w_i
            # f(x) = w^T x + b
            # f_plus = w_i * step + b
            # f_minus = w_i * (-step) + b
            # f_plus - f_minus = 2 * w_i * step
            # If f_plus < f_minus, w_i < 0 -> min at +step
            # If f_plus > f_minus, w_i > 0 -> min at -step
            diff = f_plus - f_minus
            if diff < -1e-12:
                signs[i] = 1.0  # w_i < 0, go to +5
            elif diff > 1e-12:
                signs[i] = -1.0 # w_i > 0, go to -5
            else:
                signs[i] = 0.0  # w_i approx 0

        # Construct optimal vertex
        x_opt = signs * step
        f_opt = func(x_opt)
        calls_used += 1
        update_best(x_opt, f_opt)

        # Phase 2: Refinement with Nelder-Mead
        # In case the function is not perfectly linear or to ensure we are at the local min
        # We have plenty of budget left.
        if calls_used < budget and time.time() < deadline:
            # Simple Nelder-Mead implementation
            # Initial simplex
            x_best = best_x.copy()
            f_best = best_f
            
            # Create initial simplex around x_best
            # Use a small step for the simplex
            simplex_step = 0.1
            simplex = []
            f_vals = []
            
            # Vertex
            simplex.append(x_best.copy())
            f_vals.append(f_best)
            
            # Other vertices
            for i in range(dim):
                x_v = x_best.copy()
                x_v[i] += simplex_step
                simplex.append(x_v)
                f_v = func(x_v)
                calls_used += 1
                update_best(x_v, f_v)
                f_vals.append(f_v)
                
                if time.time() > deadline or calls_used >= budget:
                    break
            
            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            max_iter = 1000
            for _ in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                x_best_nm = simplex[0]
                f_best_nm = f_vals[0]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = centroid + alpha * (centroid - x_worst)
                f_reflect = func(x_reflect)
                calls_used += 1
                update_best(x_reflect, f_reflect)
                
                if f_reflect < f_best_nm:
                    # Expansion
                    x_expand = centroid + gamma * (centroid - x_worst)
                    f_expand = func(x_expand)
                    calls_used += 1
                    update_best(x_expand, f_expand)
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_vals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_vals[-1] = f_reflect
                elif f_reflect < f_vals[-2]:
                    simplex[-1] = x_reflect
                    f_vals[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = centroid + rho * (x_worst - centroid)
                    f_contract = func(x_contract)
                    calls_used += 1
                    update_best(x_contract, f_contract)
                    
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        f_vals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best_nm + sigma * (simplex[i] - x_best_nm)
                            f_vals[i] = func(simplex[i])
                            calls_used += 1
                            update_best(simplex[i], f_vals[i])
                            
                            if time.time() > deadline or calls_used >= budget:
                                break
                        if time.time() > deadline or calls_used >= budget:
                            break

        return best_x, best_f
