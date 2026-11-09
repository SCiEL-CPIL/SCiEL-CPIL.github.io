"""Optimizer for BBOB-style f13 (Sharp ridge) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with adaptive step sizes and random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget or time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        def coordinate_descent(x_start, max_sweeps=100):
            nonlocal calls_made, best_x, best_f, deadline
            
            x = x_start.copy()
            f_current = safe_eval(x)
            
            if calls_made >= total_budget or time.time() > deadline:
                return

            # Adaptive step sizes per dimension
            step_sizes = np.ones(n) * 0.5
            min_step = 1e-6
            max_step = 10.0
            shrink_factor = 0.5
            grow_factor = 2.0

            for sweep in range(max_sweeps):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                improved = False
                for i in range(n):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    
                    # Try moving in positive and negative direction
                    best_local_f = f_current
                    best_local_x = x.copy()
                    
                    # Try several step multipliers to find good direction
                    for mult in [1.0, 0.5, 2.0, 0.25, 4.0]:
                        step = step_sizes[i] * mult
                        if step < min_step or step > max_step:
                            continue
                        
                        # Positive direction
                        x_try = x.copy()
                        x_try[i] += step
                        f_try = safe_eval(x_try)
                        if f_try < best_local_f:
                            best_local_f = f_try
                            best_local_x = x_try.copy()
                        
                        # Negative direction
                        x_try = x.copy()
                        x_try[i] -= step
                        f_try = safe_eval(x_try)
                        if f_try < best_local_f:
                            best_local_f = f_try
                            best_local_x = x_try.copy()
                    
                    if best_local_f < f_current:
                        x = best_local_x
                        f_current = best_local_f
                        improved = True
                        # Grow step size if we made progress
                        step_sizes[i] = min(step_sizes[i] * grow_factor, max_step)
                    else:
                        # Shrink step size if no improvement
                        step_sizes[i] = max(step_sizes[i] * shrink_factor, min_step)
                
                # If no dimension improved this sweep, we may be converged
                if not improved:
                    break

        # Phase 1: Sparse Halton-like grid to locate basin
        # Use quasi-random points for better coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_initial = 30
        for i in range(1, n_initial + 1):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x_point = np.array([halton(i, base) * 10.0 - 5.0 for base in [2, 3, 5, 7, 11][:n]])
            safe_eval(x_point)
        
        # Phase 2: Coordinate descent from best point found
        if best_x is not None:
            coordinate_descent(best_x.copy(), max_sweeps=80)
        
        # Phase 3: Multiple restarts with perturbations
        max_restarts = 5
        for i in range(max_restarts):
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # Perturb best_x with decreasing scale
            scale = 2.0 / (i + 1)
            if best_x is not None:
                x_start = best_x + scale * rng.standard_normal(n)
            else:
                x_start = rng.uniform(-5, 5, n)
            
            x_start = np.clip(x_start, -5, 5)
            coordinate_descent(x_start, max_sweeps=40)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
