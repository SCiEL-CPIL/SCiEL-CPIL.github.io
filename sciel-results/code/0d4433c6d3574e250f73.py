"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Sobol + Adaptive Nelder-Mead."""
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Hybrid Sobol + Adaptive Nelder-Mead optimizer for Lunacek bi-Rastrigin.
    
    Strategy:
    1. Global Phase: Use a deterministic Sobol sequence to sample the space uniformly.
       This avoids random clustering and ensures good coverage to identify the global basin
       vs the shallow basin of the bi-Rastrigin landscape.
    2. Local Phase: Use an adaptive Nelder-Mead simplex starting from the best global point.
       The simplex step size adapts based on progress, allowing it to converge quickly
       without the overhead of covariance matrix estimation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We aim to finish well before.
        # 300s limit per task contract, so 290s deadline is safe.
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # Call counter
        calls_ref = [0]
        
        def count_call():
            calls_ref[0] += 1

        def adaptive_nelder_mead(x0, f0, max_calls, func, calls_ref):
            """
            Adaptive Nelder-Mead.
            - Initializes simplex with a reasonable step size.
            - Adjusts step size if simplex shrinks or stagnates.
            """
            n = dim
            simplex = [x0.copy()]
            f_values = [f0]
            
            # Initial step size: 10% of the box width (10.0) -> 1.0
            # This is large enough to escape shallow local minima but small enough for convergence.
            current_step = 1.0
            
            for i in range(n):
                new_x = x0.copy()
                new_x[i] += current_step
                new_x = np.clip(new_x, -5.0, 5.0)
                simplex.append(new_x)
                f_val = func(new_x)
                calls_ref[0] += 1
                f_values.append(f_val)
                
                if calls_ref[0] >= max_calls:
                    return x0, f0

            best_x = x0.copy()
            best_f = f0
            last_best_f = f0
            no_improve_count = 0
            shrink_count = 0
            
            while calls_ref[0] < max_calls and time_ok():
                # Sort simplex
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Update best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Adaptive step logic
                # If we haven't improved in a while, try shrinking the step size
                # to refine the search, or expanding if we are stuck in a tiny region?
                # For bi-Rastrigin, we want to converge to the bottom of the basin.
                # Shrink if no improvement.
                if no_improve_count > 20:
                    current_step *= 0.5
                    shrink_count += 1
                    # Re-initialize simplex around current best with smaller step
                    # This effectively restarts the local search with finer granularity
                    simplex = [best_x.copy()]
                    f_values = [best_f]
                    for i in range(n):
                        new_x = best_x.copy()
                        new_x[i] += current_step
                        new_x = np.clip(new_x, -5.0, 5.0)
                        simplex.append(new_x)
                        f_val = func(new_x)
                        calls_ref[0] += 1
                        f_values.append(f_val)
                        if calls_ref[0] >= max_calls:
                            return best_x, best_f
                    no_improve_count = 0
                    if shrink_count > 10: # Step size too small, break
                        break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func(xr)
                calls_ref[0] += 1
                
                if calls_ref[0] >= max_calls:
                    break

                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func(xe)
                    calls_ref[0] += 1
                    
                    if calls_ref[0] >= max_calls:
                        break

                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = func(xc)
                    calls_ref[0] += 1
                    
                    if calls_ref[0] >= max_calls:
                        break

                    if fc < f_values[-1]:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_values[i] = func(simplex[i])
                            calls_ref[0] += 1
                            if calls_ref[0] >= max_calls:
                                return best_x, best_f
                            
            return best_x, best_f

        # Phase 1: Global Exploration with Sobol Sequence
        # Sobol sequences are quasi-random, providing low-discrepancy sampling.
        # We use 200 points (40% of budget) to ensure we hit the global basin.
        n_global = 200
        
        # Generate Sobol samples in [0, 1]^dim
        sampler = qmc.Sobol(d=dim, scramble=True, seed=123)
        # Next power of 2 >= n_global for better properties, or just n_global
        # Sobol requires power of 2 for standard implementation in scipy
        n_pow2 = 2 ** int(np.ceil(np.log2(n_global)))
        sobol_samples = sampler.random(n_pow2)
        
        # Map [0, 1] to [-5, 5]
        sobol_samples = sobol_samples * 10.0 - 5.0
        
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Evaluate global samples
        for i in range(n_global):
            if not time_ok() or calls_ref[0] >= budget:
                break
            x = sobol_samples[i]
            f = func(x)
            calls_ref[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Local Refinement with Adaptive Nelder-Mead
        if calls_ref[0] < budget and time_ok():
            x0 = best_x.copy()
            f0 = best_f
            # Allocate remaining budget to NM
            max_nm_calls = budget - calls_ref[0]
            best_x, best_f = adaptive_nelder_mead(x0, f0, max_nm_calls, func, calls_ref)
            
        # If we have time and budget left, and the result is still "high" (poor),
        # we might try a second local refinement from the second best global point?
        # But usually the best global point is in the correct basin.
        # Let's check if we have significant budget left.
        if calls_ref[0] < budget - 10 and time_ok():
             # Check if we converged well. If f is still high, maybe try another start?
             # For simplicity and robustness, one strong local run is usually sufficient
             # if the global phase found the right basin.
             pass

        return best_x, best_f
