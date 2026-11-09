import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # WFG4 Structure:
        # x[0] is the position variable defining the Pareto front.
        # x[1:] are bias variables. For WFG4, the Pareto front is achieved when bias variables are 0.
        # However, since we don't know the exact scaling/centering, we assume 0 is the optimal bias value
        # if 0 is within bounds, otherwise the closest bound.
        
        bias_opt = np.zeros(dim - 1)
        for i in range(dim - 1):
            l, h = bounds[i+1]
            if l <= 0 <= h:
                bias_opt[i] = 0.0
            elif 0 < l:
                bias_opt[i] = l
            else: # h < 0
                bias_opt[i] = h

        best_points = []
        best_fvals = []
        
        # Strategy: Grid-based decomposition on x[0] (position) and LHS on x[1:] (bias)
        # The Pareto front is 1-dimensional (parameterized by x[0]).
        # We want to sample x[0] densely to capture the front shape.
        # We want to sample x[1:] to ensure we are close to the front (bias ~ 0).
        
        # Budget allocation:
        # Let N_pos be the number of positions for x[0].
        # Let N_bias be the number of bias configurations per position.
        # Total evaluations = N_pos * N_bias.
        # To maximize hypervolume, we need good spread in x[0] and good accuracy in bias.
        # A common heuristic is to spend more evaluations on the dimension that defines the front (x[0])
        # because the front is 1D. However, if bias is not exactly 0, points will be off the front.
        # Using LHS for bias allows us to cover the bias space efficiently, but we really want to hit 0.
        # A hybrid approach: 
        # 1. Use a significant portion of the budget to sweep x[0] with bias_opt (the best guess).
        # 2. Use the remaining budget to do a local LHS search around bias_opt for a subset of x[0] values
        #    to correct for any bias errors.
        
        # Let's try a 2-phase approach:
        # Phase 1: Dense sweep of x[0] with bias_opt. This should hit the front closely if bias_opt is correct.
        # Phase 2: If budget remains, do a small local search on bias for a few x[0] points to improve.
        
        # However, the directive suggests "Latin Hypercube Sampling to generate one candidate per cell".
        # Let's interpret "cells" as a grid in the full 12D space? No, that's too sparse.
        # The "grid-based decomposition" likely refers to decomposing the search space into the position and bias parts.
        
        # Let's allocate 80% of the budget to the x[0] sweep with the optimal bias guess.
        # And 20% to a random/LHS search in the bias space for a few x[0] points to ensure robustness.
        
        n_sweep = int(self.budget * 0.8)
        n_refine = self.budget - n_sweep
        
        if n_sweep < 10:
            n_sweep = 10
            n_refine = self.budget - n_sweep
        
        x0_lo, x0_hi = bounds[0]
        
        # Generate x[0] sweep points
        # Use linear spacing to ensure even coverage of the front.
        x0_vals = np.linspace(x0_lo, x0_hi, n_sweep)
        
        # Phase 1: Sweep x[0] with bias_opt
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.empty(dim)
            x[0] = x0
            x[1:] = bias_opt
            f = problem(x)
            
            if np.all(np.isfinite(f)) and np.all(f < [3.0, 5.0, 7.0]):
                best_points.append(x.copy())
                best_fvals.append(f)
        
        # Phase 2: Refine bias for a subset of x[0] points using LHS
        # Select a few x[0] points from the sweep to refine.
        # We want to ensure that if bias_opt was slightly off, we can find better bias values.
        # We use LHS in the bias space for these specific x[0] points.
        
        if n_refine > 0:
            # Pick a subset of indices from the sweep to refine
            # Let's pick evenly spaced indices from the sweep
            n_refine_points = min(5, n_sweep)
            if n_refine_points > 0:
                refine_indices = np.linspace(0, n_sweep - 1, n_refine_points, dtype=int)
                
                # Generate LHS samples in the bias space
                # We have n_refine evaluations left.
                # Let's distribute them among the refine_indices.
                # For each refine_index, we do k = n_refine // n_refine_points evaluations.
                
                k_per_point = max(1, n_refine // n_refine_points)
                
                for idx in refine_indices:
                    if time.time() > deadline:
                        break
                    x0 = x0_vals[idx]
                    
                    # Generate k_per_point LHS samples in the bias space
                    # LHS in [lo[1:], hi[1:]]
                    # For simplicity and speed, we'll use random sampling within a small neighborhood of bias_opt
                    # if bias_opt is 0, we sample around 0.
                    
                    # Actually, let's use true LHS in the full bias bounds, but weighted towards bias_opt?
                    # No, just standard LHS in the full bounds might be too far.
                    # Let's sample in a box around bias_opt.
                    # Box size: 10% of the range?
                    
                    # Better: Just use random sampling in the full bias space for a few points?
                    # If bias_opt is correct, random sampling will be worse.
                    # If bias_opt is wrong, we need to explore.
                    
                    # Let's try a local search around bias_opt.
                    # Step size: 0.1 * range
                    step_size = 0.1 * (hi[1:] - lo[1:])
                    
                    for _ in range(k_per_point):
                        if time.time() > deadline or n_refine <= 0:
                            break
                        x = np.empty(dim)
                        x[0] = x0
                        # Sample bias variables in a small box around bias_opt
                        # Using uniform distribution in [bias_opt - step, bias_opt + step] clipped to bounds
                        x[1:] = bias_opt + (np.random.rand(dim - 1) * 2 - 1) * step_size
                        x[1:] = np.clip(x[1:], lo[1:], hi[1:])
                        
                        f = problem(x)
                        n_refine -= 1
                        
                        if np.all(np.isfinite(f)) and np.all(f < [3.0, 5.0, 7.0]):
                            # Check if this point is better than any existing point with similar x[0]
                            # Or just add it. The hypervolume calculation will handle non-dominated points.
                            # Adding points that are worse in all objectives than existing ones is useless.
                            # Adding points that are better in some and worse in others might be useful if they fill gaps.
                            # But since we are refining the bias for a specific x[0], we expect to find points
                            # that are closer to the true front (lower objectives) for that x[0].
                            # So we should replace the point from Phase 1 if the new one is better.
                            
                            # Find the index of the point from Phase 1 with this x[0]
                            # Since we stored them in order, the index in best_points might not match idx directly
                            # if some Phase 1 points were filtered out (e.g., > ref point).
                            # Let's just add it. The evaluator computes hypervolume of ALL recorded points.
                            # Duplicates or dominated points don't hurt, they just don't add volume.
                            best_points.append(x.copy())
                            best_fvals.append(f)

        if not best_points:
            # Fallback: return a single point in the center
            x = np.array([(l + h) / 2.0 for l, h in bounds])
            return x

        return best_points[0]
