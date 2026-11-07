import numpy as np
import time


class Optimizer:
    """Deterministic hierarchical coordinate-wise descent exploiting Katsuura separability."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(d)
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or check_time():
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid search to find the global basin
        # Katsuura has periodic structure with period ~1 in each dimension.
        # The hidden shift moves the minimum, so we need to search broadly.
        # Use a coarse grid with step size ~1.0 to locate the basin.
        
        # First, try origin and axis-aligned points
        try_point(np.zeros(d))
        for i in range(d):
            for val in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]:
                if used_calls >= budget or check_time():
                    break
                x = np.zeros(d)
                x[i] = val
                try_point(x)
            if used_calls >= budget or check_time():
                break

        # Random samples to get a good initial point
        num_random = min(40, budget - used_calls)
        for _ in range(num_random):
            if used_calls >= budget or check_time():
                break
            x = rng.uniform(-4, 4, d)
            try_point(x)

        if best_x is None or used_calls >= budget or check_time():
            return best_x, best_f

        # Phase 2: Hierarchical coordinate-wise descent
        # For each dimension, do a 1D search around the current best point
        # with decreasing step sizes. This exploits the separable structure.
        
        x = best_x.copy()
        f = best_f

        # Coarse-to-fine step sizes for coordinate-wise search
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]

        for step in step_sizes:
            if used_calls >= budget or check_time():
                break
            
            # For each dimension, try moving left and right by step
            improved = False
            for i in range(d):
                if used_calls >= budget or check_time():
                    break
                
                for delta in [-step, step]:
                    if used_calls >= budget or check_time():
                        break
                    x_trial = x.copy()
                    x_trial[i] += delta
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    fx_trial = try_point(x_trial)
                    if fx_trial is not None and fx_trial < f:
                        x = x_trial.copy()
                        f = fx_trial
                        best_x = x_trial.copy()
                        best_f = fx_trial
                        improved = True
            
            # If no improvement at this scale, we can stop early
            # but continue to next (smaller) scale to be safe
            if not improved and step < 0.01:
                break

        # Phase 3: Fine-tuning with random perturbations at very small scales
        if used_calls < budget and not check_time():
            for scale in [0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                if used_calls >= budget or check_time():
                    break
                num_tries = min(20, budget - used_calls)
                for _ in range(num_tries):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    x_new = np.clip(x + scale * z, -5.0, 5.0)
                    fx_new = try_point(x_new)
                    if fx_new is not None and fx_new < best_f:
                        x = x_new.copy()
                        best_x = x_new.copy()
                        best_f = fx_new

        return best_x, best_f
