"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    """Hybrid Differential Evolution and Rastrigin-aware coordinate local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

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

        # Phase 1: Global Exploration with Differential Evolution
        # Allocate ~40% of budget to DE for robust global search
        de_calls_expected = int(max_calls * 0.40)
        
        try:
            # popsize=10, maxiter=15 => ~150 calls
            # This leaves ~350 calls for local refinement
            de_res = differential_evolution(
                counted_func, 
                bounds=[(-5, 5)] * self.dim, 
                maxiter=15, 
                popsize=10, 
                tol=1e-8,
                seed=42,
                init='latinhypercube'
            )
            if de_res.fun < best_f:
                best_f = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass

        # Phase 2: Rastrigin-aware coordinate local search
        # The Büche-Rastrigin function has periodic ridges with period 2*pi
        # Local minima lie at integer multiples of 2*pi (shifted by the hidden offset)
        # We use coordinate-wise line search with ridge-jumping capability

        # Try multiple starting points from the random sampling
        # Sort by fitness and take top candidates
        candidates = [(best_f, best_x.copy())]
        
        # Re-sample to get more candidates for the local search
        n_candidates = min(10, (max_calls - call_count) // 30)
        for i in range(n_candidates):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = counted_func(x)
            candidates.append((f, x.copy()))

        candidates.sort(key=lambda t: t[0])

        # Apply local search to top candidates
        for f_init, x_init in candidates:
            if call_count >= max_calls or time.time() > deadline:
                break

            x = x_init.copy()
            f_curr = f_init
            improved = True
            max_iter_local = 50

            for iter_local in range(max_iter_local):
                if call_count >= max_calls or time.time() > deadline:
                    break

                best_dim_improvement = None
                best_dim_x = None
                best_dim_f = f_curr

                for d in range(dim):
                    # Try coordinate-wise moves with different step sizes
                    # Include ridge-jumping steps (multiples of ~2*pi)
                    step_sizes = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
                    for step in step_sizes:
                        for direction in [-1, 1]:
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            if x_trial[d] < -5 or x_trial[d] > 5:
                                continue
                            f_trial = counted_func(x_trial)
                            if f_trial < best_dim_f:
                                best_dim_f = f_trial
                                best_dim_x = x_trial.copy()
                                best_dim_improvement = d

                    if call_count >= max_calls or time.time() > deadline:
                        break

                if best_dim_improvement is None:
                    # No improvement found; try smaller steps
                    for d in range(dim):
                        for step in [0.1, 0.2, 0.05]:
                            for direction in [-1, 1]:
                                x_trial = x.copy()
                                x_trial[d] += direction * step
                                if x_trial[d] < -5 or x_trial[d] > 5:
                                    continue
                                f_trial = counted_func(x_trial)
                                if f_trial < best_dim_f:
                                    best_dim_f = f_trial
                                    best_dim_x = x_trial.copy()
                                    best_dim_improvement = d
                        if call_count >= max_calls or time.time() > deadline:
                            break
                    if best_dim_improvement is None:
                        break

                x = best_dim_x
                f_curr = best_dim_f

                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x.copy()

        # Phase 3: Final Nelder-Mead refinement if budget remains
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 20:
                try:
                    nm_res = minimize(
                        counted_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10}
                    )
                    if nm_res.fun < best_f:
                        best_f = nm_res.fun
                        best_x = nm_res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
