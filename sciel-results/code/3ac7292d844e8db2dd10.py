"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive coordinate descent with bisection for step-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        # Track calls and best
        call_count = [0]
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal best_x, best_f
            call_count[0] += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse random search to get into the right basin
        rng = np.random.default_rng(42)
        
        # Try several random starting points
        n_initial = min(30, budget // 10)
        for i in range(n_initial):
            if call_count[0] >= budget - 1:
                break
            if time.time() > deadline - 60:
                break
            x = rng.uniform(-5, 5, dim)
            fx = counted_func(x)
        
        if call_count[0] >= budget - 1 or time.time() > deadline - 60:
            return best_x, best_f

        # Phase 2: Adaptive coordinate descent with bisection
        # For each coordinate, try to find the step boundary using bisection
        x = best_x.copy()
        f = best_f
        
        # Estimate local scale from initial search
        # Try to determine the "step size" by probing nearby points
        step_sizes = []
        for i in range(dim):
            if call_count[0] >= budget - 1:
                break
            if time.time() > deadline - 60:
                break
            # Probe a few distances to estimate local structure
            for dist in [0.5, 1.0, 2.0]:
                if call_count[0] >= budget - 1:
                    break
                x_probe = x.copy()
                x_probe[i] = x[i] + dist
                if x_probe[i] > 5:
                    x_probe[i] = x[i] - dist
                if x_probe[i] < -5:
                    x_probe[i] = x[i] + dist
                f_probe = counted_func(x_probe)
                step_sizes.append(abs(f_probe - f))
        
        # Use median step size as a guide for bisection range
        if step_sizes:
            median_step = np.median(step_sizes)
            bisection_range = max(0.1, median_step * 2)
        else:
            bisection_range = 1.0

        # Phase 3: Coordinate descent with bisection for each axis
        # Iterate multiple times to converge
        max_cd_iterations = 3
        bisection_iters = 8
        
        for cd_iter in range(max_cd_iterations):
            if call_count[0] >= budget - 1 or time.time() > deadline - 30:
                break
            
            improved = False
            for i in range(dim):
                if call_count[0] >= budget - 1 or time.time() > deadline - 30:
                    break
                
                # Try bisection to find better value along axis i
                # First, determine search direction by probing
                x_plus = x.copy()
                x_minus = x.copy()
                probe_dist = bisection_range
                
                x_plus[i] = np.clip(x[i] + probe_dist, -5, 5)
                x_minus[i] = np.clip(x[i] - probe_dist, -5, 5)
                
                f_plus = counted_func(x_plus)
                f_minus = counted_func(x_minus)
                
                if call_count[0] >= budget - 1 or time.time() > deadline - 30:
                    break
                
                # Determine which direction is better
                if f_plus < f_minus:
                    lo, hi = x[i], x_plus[i]
                    f_lo, f_hi = f, f_plus
                else:
                    lo, hi = x_minus[i], x[i]
                    f_lo, f_hi = f_minus, f
                
                # Bisection to find the step boundary
                for b_iter in range(bisection_iters):
                    if call_count[0] >= budget - 1 or time.time() > deadline - 30:
                        break
                    
                    mid = (lo + hi) / 2.0
                    x_mid = x.copy()
                    x_mid[i] = mid
                    f_mid = counted_func(x_mid)
                    
                    if f_mid < f_lo:
                        # Move toward mid
                        lo, f_lo = mid, f_mid
                        hi = x[i] if x[i] > mid else x[i]
                        # Actually, we want to keep searching toward the better side
                        if f_mid < f_hi:
                            hi, f_hi = mid, f_mid
                        else:
                            lo, f_lo = mid, f_mid
                    else:
                        # The step is between mid and the better endpoint
                        if f_mid < f_hi:
                            lo, f_lo = mid, f_mid
                        else:
                            hi, f_hi = mid, f_mid
                
                # After bisection, take the best point found
                if f_lo < f:
                    x[i] = lo
                    f = f_lo
                    improved = True
                elif f_hi < f:
                    x[i] = hi
                    f = f_hi
                    improved = True
            
            if not improved:
                break
        
        # Phase 4: Final local refinement with small steps
        if call_count[0] < budget - 1 and time.time() < deadline - 10:
            local_sigma = 0.1
            for _ in range(min(20, budget - call_count[0] - 1)):
                if time.time() > deadline - 10:
                    break
                x_trial = x + local_sigma * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = counted_func(x_trial)
                if f_trial < f:
                    x, f = x_trial, f_trial
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.8
        
        return best_x, best_f
# EVOLVE-BLOCK-END
