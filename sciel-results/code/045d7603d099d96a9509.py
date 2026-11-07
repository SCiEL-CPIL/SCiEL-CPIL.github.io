import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with adaptive sigma and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        best_x = x.copy()
        best_fx = fx
        sigma = 3.0

        # Phase 1: (1+1)-ES for most of the budget
        # Reserve last 50 calls for Nelder-Mead polish
        polish_budget = 50
        es_budget = budget - polish_budget

        # Track success rate for adaptive sigma
        successes = 0
        total_trials = 0
        window_size = 10  # Look at last N trials

        trial_history = []  # (success, step_norm)

        for _ in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            step_norm = np.linalg.norm(y - x)
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                trial_history.append((True, step_norm))
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                trial_history.append((False, step_norm))

            total_trials += 1

            # Adaptive sigma based on success rate over recent window
            if len(trial_history) > window_size:
                trial_history.pop(0)
            
            recent_successes = sum(1 for s, _ in trial_history if s)
            success_rate = recent_successes / len(trial_history) if trial_history else 0.5

            # Target success rate ~0.33 for (1+1)-ES
            if success_rate > 0.4:
                sigma *= 1.15
            elif success_rate < 0.2:
                sigma *= 0.85
            else:
                sigma *= 1.0

            # Clamp sigma
            if sigma > 15.0:
                sigma = 15.0
            elif sigma < 1e-14:
                sigma = 1e-14

        # Phase 2: Nelder-Mead polish using remaining budget
        remaining = budget - calls
        if remaining > 2 and time.time() < deadline:
            # Wrap func to track calls and enforce deadline
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(z)
            
            # Use Nelder-Mead from best point
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
