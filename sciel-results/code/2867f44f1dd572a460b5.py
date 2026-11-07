"""Two-phase hybrid: radial sweep + aggressive (1+1)-ES for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Radial sweep followed by aggressive (1+1)-ES for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill
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

        # Phase 1: Dense radial sweep from origin to find the narrow axis
        # The Discus function has extreme anisotropy - one narrow axis
        # Try many directions densely
        n_dirs = 300
        n_radii = 15
        
        # Generate directions: mix of axis-aligned and random
        directions = []
        # Axis-aligned directions (positive and negative)
        for i in range(dim):
            for sign in [-1, 1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        # Random directions
        for _ in range(n_dirs - 2 * dim):
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                directions.append(d / norm)
        
        # Radii: logarithmically spaced from small to large
        radii = np.logspace(-1, np.log10(4.5), n_radii)
        
        # Evaluate radial sweep
        for d in directions:
            for r in radii:
                if call_count >= budget:
                    break
                x = r * d
                if np.any(x < -5.0) or np.any(x > 5.0):
                    continue
                fx = try_eval(x)
                if fx is None:
                    break
            if call_count >= budget:
                break
        
        if best_x is None:
            # Fallback: evaluate origin
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: Aggressive (1+1)-ES with high mutation rate
        # Start from best point found, use large initial step
        x_curr = best_x.copy()
        f_curr = best_f
        sigma = 0.5  # Start with moderate step, will adapt
        
        # Success rate target for step size adaptation
        target_success = 0.2  # Aim for 20% success rate (aggressive)
        success_count = 0
        total_trials = 0
        step_size_history = []
        
        # (1+1)-ES loop
        while call_count < budget:
            if time.time() > deadline - 5:
                break
            
            # Generate mutant
            z = rng.standard_normal(dim)
            x_trial = x_curr + sigma * z
            
            # Evaluate
            f_trial = try_eval(x_trial)
            if f_trial is None:
                break
            
            total_trials += 1
            
            # Accept if better
            if f_trial < f_curr:
                success_count += 1
                x_curr = x_trial.copy()
                f_curr = f_trial
            
            # Adapt step size every 10 trials
            if total_trials % 10 == 0:
                recent_success = success_count / 10.0
                if recent_success > target_success:
                    sigma *= 1.5  # Increase step if too many successes
                elif recent_success < target_success:
                    sigma *= 0.7  # Decrease step if too few successes
                sigma = min(sigma, 5.0)
                sigma = max(sigma, 1e-4)
                success_count = 0  # Reset counter
        
        return best_x, best_f
# EVOLVE-BLOCK-END
