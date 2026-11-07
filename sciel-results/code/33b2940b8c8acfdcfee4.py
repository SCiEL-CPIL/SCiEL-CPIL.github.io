"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with deadline guard for f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # LHS-style diverse initial points
        def lhs_points(n, dim, rng):
            pts = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                for i in range(n):
                    pts[perm[i], d] = -5.0 + 10.0 * (i + 0.5) / n
            return pts

        # First batch: diverse LHS points to explore landscape
        n_lhs = min(30, budget // 10)
        lhs_pts = lhs_points(n_lhs, dim, rng)
        best_x = None
        best_f = float('inf')
        call_count = 0

        for pt in lhs_pts:
            if time.time() > deadline or call_count >= budget:
                break
            f_val = func(pt)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        # Multi-restart (1+1)-ES from best point and a few others
        remaining = budget - call_count
        if remaining <= 0:
            return best_x, best_f

        # Adjust restarts and steps to fit exactly within remaining budget
        # Each restart costs 1 (init) + per_restart (steps)
        # Let n_restarts be k. Total cost = k * (1 + per_restart) <= remaining
        # We want to maximize exploration. Let's fix per_restart to a reasonable value
        # and calculate how many restarts fit.
        
        # Strategy: Use a fixed step count for ES to keep logic simple and predictable.
        # If we use per_restart = 40, cost per restart = 41.
        # 470 remaining / 41 = 11 restarts.
        # This is better than the previous 8 restarts with 58 steps.
        # More restarts can help escape local optima on f21 (many peaks).
        
        per_restart = 40
        cost_per_restart = per_restart + 1
        n_restarts = max(1, remaining // cost_per_restart)
        
        # If we have leftover budget, we can do one more partial restart or just stop.
        # To be safe and simple, we just do n_restarts full restarts.
        
        # Start points: best so far + some random
        start_points = [best_x.copy()]
        for _ in range(n_restarts - 1):
            start_points.append(rng.uniform(-5, 5, dim))

        for sp in start_points:
            if time.time() > deadline or call_count >= budget:
                break
            
            x = sp.copy()
            # Check budget before initial eval
            if call_count >= budget:
                break
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # (1+1)-ES with 1/5 rule adaptation
            successes = 0
            window = 10
            
            for _ in range(per_restart):
                if time.time() > deadline or call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    successes = max(0, successes - 1)
                # Adapt sigma every 10 steps
                if call_count % window == 0:
                    if successes > window / 2:
                        sigma *= 1.3
                    elif successes < window / 4:
                        sigma *= 0.7
                    successes = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
