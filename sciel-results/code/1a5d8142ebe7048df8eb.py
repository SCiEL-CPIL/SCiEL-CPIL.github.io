"""Hybrid Nelder-Mead + (1+1)-ES optimizer for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex followed by (1+1)-ES for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Nelder-Mead Simplex Method
        # Initialize simplex
        x0 = np.zeros(dim)
        f0 = try_eval(x0)
        if f0 is None:
            return best_x, best_f

        simplex = [x0.copy()]
        # Create initial simplex vertices
        # Standard Nelder-Mead initialization
        delta = 0.5
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            fi = try_eval(xi)
            if fi is None:
                break
            simplex.append(xi)
        
        # Evaluate initial simplex
        # We already have f0 for x0. We need to evaluate the others.
        # Wait, the loop above evaluated them and stored in 'simplex' but not their values in a structured way for the algorithm.
        # Let's restart the Nelder-Mead loop properly.
        
        # Reset state for clean Nelder-Mead run
        call_count_nm = 0 # Local counter for NM logic, but we use global try_eval
        
        # Re-initialize simplex to be safe
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.5 # Initial step size
            fi = try_eval(xi)
            if fi is None:
                break
            simplex.append(xi)
            simplex_f.append(fi)
        
        if len(simplex) < dim + 1:
            # Not enough calls to form a full simplex, fallback to random or just return
            pass
        else:
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            max_nm_iter = budget - call_count - 50 # Leave some for ES
            for _ in range(max_nm_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline - 20:
                    break
                
                # Sort simplex by function value
                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = try_eval(xr)
                if fr is None:
                    break
                
                if fr < simplex_f[0]:
                    # Check for expansion
                    if fr < simplex_f[-2]: # Better than second worst
                        xe = centroid + gamma * (centroid - simplex[-1])
                        fe = try_eval(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Between best and second worst
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    # Outer contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = try_eval(xc)
                    if fc is None:
                        break
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fi = try_eval(simplex[i])
                            if fi is None:
                                break
                            simplex_f[i] = fi
                        if call_count >= budget:
                            break

        # Phase 2: (1+1)-ES polish
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(50, budget - call_count)):
                if time.time() > deadline - 5:
                    break
                if call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
