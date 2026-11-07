"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(0)

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Phase 1: Multi-restart (1+1)-ES
        # Use ~75% of budget for exploration
        es_budget = int(budget * 0.75)
        calls_in_phase1 = 0

        # Try multiple restarts with different initial points
        num_restarts = 3
        budget_per_restart = es_budget // num_restarts

        for restart in range(num_restarts):
            if not time_ok() or calls_in_phase1 >= es_budget:
                break

            # Initialize with a new random point (or near best for later restarts)
            if restart == 0:
                x = rng.uniform(-5, 5, d)
            else:
                # Start near the current best with some perturbation
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(d) * 2.0, -5, 5)
                else:
                    x = rng.uniform(-5, 5, d)

            fx = func(x)
            calls_in_phase1 += 1
            total_calls += 1

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            sigma = 2.0
            no_improve = 0
            max_no_improve = 50

            # Run (1+1)-ES for this restart
            for _ in range(budget_per_restart - 1):
                if not time_ok() or calls_in_phase1 >= es_budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls_in_phase1 += 1
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    no_improve = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1

                # If stuck for too long, break out of this restart
                if no_improve >= max_no_improve:
                    break

        # Phase 2: Nelder-Mead polish using remaining budget
        if time_ok() and best_x is not None:
            try:
                from scipy.optimize import minimize
                polish_budget = budget - total_calls
                if polish_budget > 10:
                    # Use a local search around the best point
                    res = minimize(
                        lambda v: func(np.clip(v, -5, 5)),
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': polish_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'initial_simplex': np.array([best_x + 0.1 * np.eye(d).T[i] for i in range(d + 1)])
                        }
                    )
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(d)
            best_fx = func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
