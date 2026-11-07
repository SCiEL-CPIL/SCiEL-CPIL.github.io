"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using LHS + Nelder-Mead."""
import numpy as np
import time

class Optimizer:
    """Hybrid LHS + Nelder-Mead optimizer for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        def nelder_mead(x0, f0, max_calls, func, calls_ref):
            """
            Nelder-Mead simplex method.
            Returns (best_x, best_f).
            """
            n = dim
            # Initialize simplex
            simplex = [x0]
            f_values = [f0]
            
            # Create initial simplex vertices
            # Standard NM uses a small perturbation. 
            # Given the box [-5, 5] and the nature of BBOB, a step of 0.1 is reasonable for local refinement.
            # However, to be safe against scaling, we might use a slightly larger step if we are sure we are in the basin.
            # Let's use 0.1 as a standard local search step.
            step = 0.1
            for i in range(n):
                new_x = x0.copy()
                new_x[i] += step
                new_x = np.clip(new_x, -5, 5)
                simplex.append(new_x)
                f_val = func(new_x)
                calls_ref[0] += 1
                f_values.append(f_val)
                
                if calls_ref[0] >= max_calls:
                    return x0, f0

            best_x = x0.copy()
            best_f = f0

            # Track stagnation to break early if NM gets stuck
            stagnant = 0
            max_stagnant = 50

            while calls_ref[0] < max_calls and time_ok():
                # Sort simplex by function value
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Update best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                    stagnant = 0
                else:
                    stagnant += 1
                
                if stagnant > max_stagnant:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_ref[0] += 1
                
                if calls_ref[0] >= max_calls:
                    break

                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
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
                    
                    xc = np.clip(xc, -5, 5)
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
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
                            calls_ref[0] += 1
                            if calls_ref[0] >= max_calls:
                                return best_x, best_f
            
            return best_x, best_f

        calls_ref = [0]
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Latin Hypercube Sampling
        # Use a significant portion of the budget for global exploration.
        # LHS provides uniform coverage.
        # Let's use 100 points for LHS. This is 20% of budget.
        n_lhs = 100
        # Generate LHS points
        # For each dimension, divide [-5, 5] into n_lhs intervals and pick one random point in each.
        # This ensures coverage.
        
        # To make it deterministic but varied per instance, we rely on np.random state which is reset per instance? 
        # No, np.random is global. But the task says "hidden seeded shift". 
        # The evaluator creates Optimizer(budget, dim) and calls it. 
        # We should use a local random generator to ensure reproducibility within the run if needed, 
        # but here we just need good sampling.
        
        rng = np.random.RandomState(42) # Fixed seed for LHS structure? Or use global? 
        # Using a fixed seed for the LHS pattern is good for reproducibility of the search strategy itself.
        # The "hidden seed" affects the function, not our RNG.
        
        # Create LHS
        # n_lhs points in dim dimensions
        # For dim=5, n_lhs=100
        # Permutation for each dimension
        points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            # Map permutation to [-5, 5]
            # Interval width: 10 / n_lhs
            width = 10.0 / n_lhs
            points[:, i] = (perm + rng.uniform(size=n_lhs)) * width - 5.0
            
        # Evaluate LHS points
        for i in range(n_lhs):
            if not time_ok() or calls_ref[0] >= budget:
                break
            x = points[i]
            f = func(x)
            calls_ref[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Local Refinement with Nelder-Mead
        # Start NM from the best point found in LHS
        if calls_ref[0] < budget and time_ok():
            x0 = best_x.copy()
            f0 = best_f
            # Allocate remaining budget to NM
            # NM is fast, but we cap it to avoid wasting time if it stagnates quickly.
            # Let's give it up to 300 calls (60% of budget).
            max_nm_calls = min(300, budget - calls_ref[0])
            best_x, best_f = nelder_mead(x0, f0, calls_ref[0] + max_nm_calls, func, calls_ref)

        # If we still have budget and time, and the result isn't great, maybe do a second round of LHS + NM?
        # Or just stick to one robust pass. 
        # Given the "bi-Rastrigin" has a deep and a shallow basin, LHS with 100 points in 5D should likely hit the deep basin.
        # 100 points in 5D: volume per point is (10/100)^5 = 1e-5 * 100000 = 1. 
        # So we have 100 points spread out.
        
        # If best_f is still high (e.g., > 50), it might be in the shallow basin or far away.
        # We could try a second NM from a different LHS point if the first one didn't converge well.
        # But for simplicity and speed, let's stick to the best one.
        
        return best_x, best_f
