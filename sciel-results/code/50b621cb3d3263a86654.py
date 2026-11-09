"""Combined (1+1)-ES + Nelder-Mead optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES for global search + Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def run_es(x0, sigma0, max_calls):
            """Run (1+1)-ES with adaptive step size."""
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget or time.time() > deadline:
                return

            x = np.array(x0, dtype=np.float64)
            f = f_best
            sigma = sigma0
            rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
            success_count = 0
            total_steps = 0

            while calls_made < max_calls and time.time() < deadline:
                # Generate candidate
                direction = rng.standard_normal(dim)
                trial = x + sigma * direction
                trial = np.clip(trial, -5, 5)
                f_trial = evaluate(trial)

                total_steps += 1
                if f_trial < f:
                    # Success: accept and increase sigma slightly
                    x = trial.copy()
                    f = f_trial
                    success_count += 1
                    sigma *= 1.1
                    if f < f_best:
                        f_best = f
                        x_best = x.copy()
                else:
                    # Failure: decrease sigma
                    sigma *= 0.9

                # Adapt sigma based on recent success rate
                if total_steps % 10 == 0 and total_steps >= 20:
                    recent_success_rate = success_count / total_steps
                    if recent_success_rate > 0.5:
                        sigma *= 1.2
                    elif recent_success_rate < 0.2:
                        sigma *= 0.8
                    success_count = 0
                    total_steps = 0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 1e-6, 10.0)

        def run_nm(x0, max_calls):
            """Run Nelder-Mead local refinement."""
            nonlocal calls_made, x_best, f_best
            if x0 is None or max_calls < 5 or time.time() > deadline:
                return

            local_x = np.array(x0, dtype=np.float64)
            local_f = f_best

            def wrapped(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return local_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
                fx = float(func(x))
                calls_made += 1
                if fx < local_f:
                    local_f = fx
                    x_best = x.copy()
                return fx

            try:
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': 1e-7, 'fatol': 1e-9,
                                        'adaptive': True})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        # Strategy: Multi-restart (1+1)-ES with Nelder-Mead polishing
        # Büche-Rastrigin has a rugged landscape with many local minima,
        # so we use multiple restarts to explore different basins.

        # Phase 1: Initial exploration with (1+1)-ES from origin
        es_calls = int(budget * 0.4)
        if time.time() < deadline:
            run_es(np.zeros(dim), 2.0, es_calls)

        # Phase 2: Nelder-Mead refinement on best point
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls = min(remaining, 40)
            run_nm(x_best.copy(), nm_calls)

        # Phase 3: Second (1+1)-ES restart from a different location
        remaining = budget - calls_made
        if remaining >= 20 and time.time() < deadline:
            rng = np.random.default_rng(int(time.time() * 2000) % (2**31))
            x_start2 = rng.uniform(-4, 4, dim)
            es_calls2 = min(remaining, int(remaining * 0.5))
            run_es(x_start2, 1.5, es_calls2)

        # Phase 4: Another Nelder-Mead refinement
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls2 = min(remaining, 30)
            run_nm(x_best.copy(), nm_calls2)

        # Phase 5: Third (1+1)-ES restart if budget remains
        remaining = budget - calls_made
        if remaining >= 15 and time.time() < deadline:
            rng = np.random.default_rng(int(time.time() * 3000) % (2**31))
            x_start3 = rng.uniform(-3, 3, dim)
            es_calls3 = min(remaining, int(remaining * 0.6))
            run_es(x_start3, 1.0, es_calls3)

        # Phase 6: Final Nelder-Mead polish
        remaining = budget - calls_made
        if remaining >= 5 and x_best is not None and time.time() < deadline:
            nm_calls3 = min(remaining, 20)
            run_nm(x_best.copy(), nm_calls3)

        # Final safety: ensure we have a result
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
