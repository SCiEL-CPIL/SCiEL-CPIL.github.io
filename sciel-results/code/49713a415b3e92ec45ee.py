"""Two-phase hybrid: dense radial/axis sweep to find the needle, then aggressive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase hybrid: dense radial/axis sweep to find the needle, then aggressive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def try_eval(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Dense radial/axis sweep to find the narrow anisotropic axis
        # The Discus function has extreme anisotropy - one very narrow axis
        # We sweep along axis-aligned and random directions at multiple radii
        
        n_axes = dim
        n_random_dirs = 80
        n_radii = 10
        
        # Generate directions
        directions = []
        # Axis-aligned directions (positive and negative)
        for i in range(dim):
            for sign in [-1, 1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        
        # Random directions
        while len(directions) < n_axes * 2 + n_random_dirs:
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                directions.append(d / norm)
        
        # Radii: geometric spacing from small to large
        radii = np.geomspace(0.05, 4.5, n_radii)
        
        # Evaluate all combinations
        for d in directions:
            for r in radii:
                if calls >= budget:
                    break
                x = d * r
                try_eval(x)
            if calls >= budget:
                break
        
        if best_x is None:
            # Fallback: evaluate origin
            x0 = np.zeros(dim)
            f0 = try_eval(x0)
            if f0 is not None:
                best_x = x0.copy()
                best_f = f0

        if best_x is None or calls >= budget:
            return best_x, best_f

        # Phase 2: Aggressive (1+1)-ES with adaptive step size
        # Start from the best point found in Phase 1
        x = best_x.copy()
        f = best_f
        
        # Initial step size: estimate from Phase 1 results
        # Use a moderate starting step
        sigma = 0.5
        
        # (1+1)-ES parameters
        success_rate_target = 0.2
        c_sigma = 0.05
        c_sigma_max = 0.5
        c_sigma_min = 0.005
        d_sigma = 1.0
        
        # Adaptive step size control
        sigma_state = sigma
        
        max_iter = budget - calls
        stagnant_count = 0
        improvement_threshold = 1e-10
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Generate mutant
            z = rng.standard_normal(dim)
            x_mut = x + sigma_state * z
            x_mut = np.clip(x_mut, -5.0, 5.0)
            
            f_mut = try_eval(x_mut)
            if f_mut is None:
                break
            
            # Update best if improved
            if f_mut < f - improvement_threshold:
                # Success
                x = x_mut
                f = f_mut
                best_x = x.copy()
                best_f = f
                stagnant_count = 0
                
                # Increase step size on success
                sigma_state *= (1 + c_sigma)
                sigma_state = min(sigma_state, 10.0)
            else:
                # Failure
                stagnant_count += 1
                
                # Decrease step size on failure
                sigma_state *= (1 - c_sigma)
                sigma_state = max(sigma_state, 1e-6)
            
            # If stagnating for too long, try a larger jump
            if stagnant_count > 50:
                # Random restart around current best with larger step
                jump_sigma = max(sigma_state * 5, 1.0)
                z_jump = rng.standard_normal(dim)
                x_jump = x + jump_sigma * z_jump
                x_jump = np.clip(x_jump, -5.0, 5.0)
                
                f_jump = try_eval(x_jump)
                if f_jump is not None:
                    if f_jump < f - improvement_threshold:
                        x = x_jump
                        f = f_jump
                        best_x = x.copy()
                        best_f = f
                        stagnant_count = 0
                        sigma_state = jump_sigma * 0.5
                    else:
                        # Reset stagnant count and try smaller steps
                        stagnant_count = 0
                        sigma_state = max(sigma_state * 0.5, 1e-6)
                if calls >= budget:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
