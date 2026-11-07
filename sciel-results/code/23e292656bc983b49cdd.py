"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Rastrigin-aware adaptive local search with Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        dim = self.dim

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Lattice-aware sampling
        # Büche-Rastrigin has minima near integer multiples of 2*pi (shifted)
        # Sample near these lattice points plus some random points
        candidates = []
        
        # Try points near the origin (likely global minimum region)
        if call_count < max_calls:
            x_origin = np.zeros(dim)
            f_origin = counted_func(x_origin)
            candidates.append((f_origin, x_origin.copy()))
        
        # Sample near 2*pi lattice points
        period = 2.0 * np.pi
        for k in [-1, 0, 1]:
            if call_count >= max_calls or time.time() > deadline:
                break
            for _ in range(3):
                if call_count >= max_calls or time.time() > deadline:
                    break
                x = k * period + rng.uniform(-0.5, 0.5, dim)
                x = np.clip(x, -5, 5)
                f = counted_func(x)
                candidates.append((f, x.copy()))
        
        # Add some random points for exploration
        n_random = min(15, max(5, max_calls // 20))
        for i in range(n_random):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = counted_func(x)
            candidates.append((f, x.copy()))

        # Sort by fitness and keep top candidates
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Adaptive multi-scale local search on top candidates
        n_local = min(6, len(candidates))
        base_steps = [5.0, 2.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        for f_init, x_init in candidates[:n_local]:
            if call_count >= max_calls or time.time() > deadline:
                break

            x = x_init.copy()
            f_curr = f_init
            max_iter_local = 25

            for iter_local in range(max_iter_local):
                if call_count >= max_calls or time.time() > deadline:
                    break

                best_dim_f = f_curr
                best_dim_x = None
                found_improvement = False

                # Adaptive step sizing: start large, shrink as we converge
                step_idx = min(iter_local // 3, len(base_steps) - 1)
                steps_to_try = base_steps[step_idx:]

                for d in range(dim):
                    for step in steps_to_try:
                        for direction in [-1, 1]:
                            if call_count >= max_calls or time.time() > deadline:
                                break
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            if x_trial[d] < -5 or x_trial[d] > 5:
                                continue
                            f_trial = counted_func(x_trial)
                            if f_trial < best_dim_f:
                                best_dim_f = f_trial
                                best_dim_x = x_trial.copy()
                                found_improvement = True

                if not found_improvement:
                    break

                x = best_dim_x
                f_curr = best_dim_f

                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x.copy()

        # Phase 3: Nelder-Mead refinement on the best point found
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 20:
                try:
                    nm_res = minimize(
                        counted_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10, 'adaptive': True}
                    )
                    if nm_res.fun < best_f:
                        best_f = nm_res.fun
                        best_x = np.clip(nm_res.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 4: Final fine coordinate search if budget remains
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 10:
                x = best_x.copy()
                f_curr = best_f
                fine_steps = [0.01, 0.005, 0.002, 0.001, 0.0005]
                for step in fine_steps:
                    if call_count >= max_calls or time.time() > deadline:
                        break
                    for d in range(dim):
                        for direction in [-1, 1]:
                            if call_count >= max_calls or time.time() > deadline:
                                break
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            if x_trial[d] < -5 or x_trial[d] > 5:
                                continue
                            f_trial = counted_func(x_trial)
                            if f_trial < f_curr:
                                x = x_trial
                                f_curr = f_trial
                                if f_curr < best_f:
                                    best_f = f_curr
                                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
