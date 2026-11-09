"""
Optimizer for Bent Cigar (f12) with anisotropy-aware adaptive coordinate descent.
"""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Bent Cigar is unimodal but shifted/rotated. 
        # We need to find the basin of attraction.
        n_global = 100
        sobol = qmc.Sobol(d=dim, scramble=True, seed=123)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break
        
        if best_x is None:
            x0 = np.zeros(dim)
            wrapped_obj(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Anisotropy-Aware Adaptive Coordinate Descent ---
        # The Bent Cigar function has one flat dimension and n-1 steep dimensions.
        # We use adaptive step sizes for each dimension.
        # If a step is successful, we expand the step size (exploration).
        # If a step is unsuccessful, we shrink the step size (exploitation).
        
        # Initialize step sizes. 
        # Start with a medium size. The algorithm will adapt.
        h = np.ones(dim) * 0.5
        
        # Shrink and expand factors
        shrink_factor = 0.5
        expand_factor = 1.5
        
        # Minimum step size to consider converged
        min_h = 1e-6
        
        current_x = x0.copy()
        current_fx = best_fx
        
        # We allow for multiple passes over dimensions until no improvement is made
        # or budget/time is exhausted.
        
        max_passes = 500
        pass_count = 0
        
        while calls_used < budget and time.time() < deadline:
            improved_in_pass = False
            pass_count += 1
            
            for i in range(dim):
                if time.time() >= deadline or calls_used >= budget:
                    break
                
                # Skip if step is too small
                if h[i] < min_h:
                    continue
                
                # Try positive step
                x_plus = current_x.copy()
                x_plus[i] += h[i]
                x_plus = np.clip(x_plus, -5.0, 5.0)
                
                fx_plus = wrapped_obj(x_plus)
                
                if fx_plus < current_fx:
                    current_x = x_plus
                    current_fx = fx_plus
                    improved_in_pass = True
                    # Success: Expand step to explore further in this direction
                    h[i] *= expand_factor
                    continue
                
                # Try negative step
                x_minus = current_x.copy()
                x_minus[i] -= h[i]
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                fx_minus = wrapped_obj(x_minus)
                
                if fx_minus < current_fx:
                    current_x = x_minus
                    current_fx = fx_minus
                    improved_in_pass = True
                    # Success: Expand step
                    h[i] *= expand_factor
                    continue
                
                # No improvement: Shrink step size for this dimension
                h[i] *= shrink_factor
            
            # If no dimension improved, we might be stuck or converged.
            # In Bent Cigar, if we are at the bottom of the steep dimensions,
            # only the flat dimension might still improve.
            # If h[i] becomes very small for all i, we stop.
            
            if not improved_in_pass:
                # Check if we can still improve by trying larger steps?
                # Usually, if we shrink too much, we miss the valley.
                # But if we expanded too much, we might oscillate.
                # Let's check if all steps are small.
                if np.all(h < min_h):
                    break
                
                # If we are stuck, try to "kick" the dimensions with the largest steps?
                # Or just break if we've done enough passes without improvement.
                # For Bent Cigar, once we are in the valley, we mostly move along the flat axis.
                # The adaptive step size should naturally find this.
                # If we made no improvement in a full pass, it's likely we are at a local minimum
                # or the steps are too small/large.
                # Given the unimodal nature, breaking is safe if steps are small.
                # If steps are large, we might have oscillated.
                # Let's break after a few passes of no improvement to save budget for other instances?
                # No, we only have one instance per call.
                # Let's break if no improvement and steps are small.
                if np.max(h) < 1e-3:
                    break
                else:
                    # If steps are large but no improvement, we might be oscillating.
                    # Reduce all steps slightly to dampen oscillation?
                    # Or just break.
                    # Let's break to avoid wasting budget on oscillations.
                    break
            
            # Safety break if we've done too many passes
            if pass_count > 200:
                break

        return best_x, best_fx
