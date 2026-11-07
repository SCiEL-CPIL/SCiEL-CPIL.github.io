"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES-like optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # We'll use a simple but effective strategy:
        # 1. Start with a few diverse initial points
        # 2. Use a (1+1)-ES with adaptive sigma, but restart if stuck
        # 3. Track best point and return it

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if len(x) != dim:
                x = np.resize(x, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate initial diverse points
        n_starts = 5
        initial_points = []
        # Center point
        initial_points.append(np.zeros(dim))
        # Random points
        for _ in range(n_starts - 1):
            initial_points.append(rng.uniform(-5, 5, dim))

        # Evaluate initial points
        for p in initial_points:
            if time.time() > deadline or calls_used >= self.budget:
                break
            evaluate(p)

        # Now do local search from the best point using (1+1)-ES with restarts
        # We'll do multiple rounds of (1+1)-ES from different starting points

        # Sort initial points by fitness
        # But we only tracked best, so let's just use the best and a few random ones

        # Strategy: do (1+1)-ES from best point, and also from a few other points
        # with limited budget each

        # Let's do a series of (1+1)-ES runs from different starts
        # Each run gets a fraction of the remaining budget

        remaining_budget = self.budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Number of (1+1)-ES runs
        n_es_runs = 3
        budget_per_run = remaining_budget // n_es_runs
        if budget_per_run < 10:
            budget_per_run = remaining_budget
            n_es_runs = 1

        # Starting points for ES runs: best point + 2 random
        es_starts = [best_x.copy()]
        for _ in range(min(n_es_runs - 1, 2)):
            es_starts.append(rng.uniform(-5, 5, dim))

        for start_idx, es_start in enumerate(es_starts):
            if time.time() > deadline or calls_used >= self.budget:
                break

            # If this isn't the first run, evaluate the start point
            if start_idx > 0:
                if time.time() > deadline or calls_used >= self.budget:
                    break
                evaluate(es_start)

            x = es_start.copy()
            # Get current fitness
            # We need to know fx; if it's the best, we have it, otherwise we just evaluated
            # Let's just re-evaluate if needed, but that wastes a call.
            # Instead, track it:
            # For start_idx == 0, x is best_x, fx is best_f
            # For others, we just evaluated, so fx is the last evaluated value
            # This is getting complex. Let's simplify: just use a local (1+1)-ES
            # that starts from the given point and tracks its own best.

            # Simplified: each ES run is independent, tracking its own best
            # and updating global best.

            # Actually, let's just do one long (1+1)-ES from the global best,
            # but with periodic restarts if no improvement for a while.

            # Let me restructure: single (1+1)-ES with restart mechanism.

            break  # Break out, we'll do a different approach below

        # Better approach: single (1+1)-ES from best point with adaptive restart
        if time.time() > deadline or calls_used >= self.budget:
            return best_x, best_f

        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        no_improve = 0
        max_no_improve = 50

        while calls_used < self.budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                no_improve = 0
            else:
                sigma *= 0.9
                no_improve += 1

            # Restart if stuck
            if no_improve > max_no_improve:
                # Jump to a random point
                x = rng.uniform(-5, 5, dim)
                fx = evaluate(x)
                sigma = 2.0
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
