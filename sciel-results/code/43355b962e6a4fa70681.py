"""Multi-scale coordinate descent + Nelder-Mead hybrid optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-scale coordinate descent for early progress + Nelder-Mead for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
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

        def run_coordinate_descent(x0, max_calls):
            """
            Multi-scale coordinate descent.
            Strategy:
            - Start with large steps to find the basin.
            - Shrink steps geometrically.
            - For each dimension, try +step. If it improves, accept. Else try -step.
            - This saves calls compared to always trying both.
            """
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget or time.time() > deadline:
                return

            x = np.array(x0, dtype=np.float64)
            f = evaluate(x)
            
            if f < f_best:
                f_best = f
                x_best = x.copy()

            # Step sizes: Large to small
            # 4.0 covers half the domain [-5, 5]
            # We cycle through these scales.
            step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 
                          0.015625, 0.0078125, 0.00390625, 0.001953125,
                          0.0009765625, 0.00048828125, 0.000244140625]

            # We will iterate through step sizes. If we run out of calls, we stop.
            # To maximize efficiency, we prioritize larger steps first.
            for step in step_sizes:
                if calls_made >= max_calls or time.time() > deadline:
                    break
                
                for d in range(dim):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    trial = x.copy()
                    trial[d] += step
                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial < f:
                        x = trial
                        f = f_trial
                    else:
                        # Try negative direction only if positive failed
                        # This conserves budget for more dimensions/steps
                        trial = x.copy()
                        trial[d] -= step
                        trial = np.clip(trial, -5, 5)
                        f_trial = evaluate(trial)
                        
                        if f_trial < f:
                            x = trial
                            f = f_trial

            if f < f_best:
                f_best = f
                x_best = x.copy()

        def run_nm(x0, max_calls):
            """
            Nelder-Mead refinement.
            Uses scipy's adaptive Nelder-Mead.
            """
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
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                return fx

            try:
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        # --- Phase 1: Initial Exploration from Origin ---
        # Allocate ~35% of budget to find the main basin
        cd_calls_1 = int(budget * 0.35)
        if time.time() < deadline:
            run_coordinate_descent(np.zeros(dim), cd_calls_1)

        # --- Phase 2: Refinement ---
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls_1 = min(remaining, 30)
            run_nm(x_best.copy(), nm_calls_1)

        # --- Phase 3: Restart 1 ---
        remaining = budget - calls_made
        if remaining >= 20 and time.time() < deadline:
            rng = np.random.default_rng(42)
            x_start2 = rng.uniform(-3, 3, dim)
            cd_calls_2 = min(remaining, int(remaining * 0.3))
            run_coordinate_descent(x_start2, cd_calls_2)

        # --- Phase 4: Refinement 2 ---
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls_2 = min(remaining, 25)
            run_nm(x_best.copy(), nm_calls_2)

        # --- Phase 5: Restart 2 ---
        remaining = budget - calls_made
        if remaining >= 20 and time.time() < deadline:
            rng = np.random.default_rng(123)
            x_start3 = rng.uniform(-4, 4, dim)
            cd_calls_3 = min(remaining, int(remaining * 0.3))
            run_coordinate_descent(x_start3, cd_calls_3)

        # --- Phase 6: Refinement 3 ---
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            nm_calls_3 = min(remaining, 25)
            run_nm(x_best.copy(), nm_calls_3)

        # --- Phase 7: Final Polish ---
        # Use remaining budget for a final aggressive NM run if available
        remaining = budget - calls_made
        if remaining >= 5 and x_best is not None and time.time() < deadline:
            nm_calls_4 = min(remaining, 20)
            run_nm(x_best.copy(), nm_calls_4)

        # Final safety
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
