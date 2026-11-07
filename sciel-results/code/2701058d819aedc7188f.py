"""Optimizer for BBOB f13 (Sharp ridge) using Powell's method with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Powell's method for sharp ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin

        # Track total function calls
        calls = [0]

        def safe_func(x):
            calls[0] += 1
            if calls[0] > budget:
                return float('inf')
            return func(np.clip(x, -5, 5))

        def powell(func_eval, x0, budget_remaining, dim):
            """Powell's method with adaptive directions."""
            n = dim
            # Initial directions: standard basis
            directions = np.eye(n)
            x = np.array(x0, dtype=float)
            fx = func_eval(x)
            budget_used = 1
            best_x = x.copy()
            best_fx = fx

            # Powell's method iterations
            while budget_used < budget_remaining:
                x_min = x.copy()
                fx_min = fx
                # For each direction, do a line search
                for i in range(n):
                    # Line search along directions[i]
                    # Use a small step and refine
                    d = directions[i]
                    # Initial step size
                    step = 0.5
                    x_trial = x + step * d
                    fx_trial = func_eval(x_trial)
                    budget_used += 1

                    if fx_trial >= fx:
                        # Try negative direction
                        x_trial = x - step * d
                        fx_trial = func_eval(x_trial)
                        budget_used += 1
                        if fx_trial >= fx:
                            continue
                    else:
                        # Go in positive direction
                        pass

                    # Now we have a better point in some direction
                    # Do a proper 1D minimization
                    # Find the interval containing the minimum
                    if fx_trial < fx:
                        # Determine direction sign
                        if x_trial[0] > x[0]:
                            sign = 1.0
                        else:
                            sign = -1.0
                        # Expand to find bracket
                        a = 0.0
                        b = sign * step
                        fa = fx
                        fb = fx_trial
                        # Expand b
                        while budget_used < budget_remaining:
                            b_new = b + sign * step
                            x_b = x + b_new * d
                            fb = func_eval(x_b)
                            budget_used += 1
                            if fb > fb and b_new * sign > 0:
                                break
                            if abs(b_new) > 20:
                                break
                            a, b = b, b_new
                            fa, fb = fb, fb
                            step *= 0.5
                            if step < 1e-6:
                                break
                    else:
                        continue

                    # Golden section search in [a, b]
                    if budget_used >= budget_remaining:
                        break
                    golden = 0.6180339887498949
                    a_val = a
                    b_val = b
                    c_val = b_val - golden * (b_val - a_val)
                    d_val = a_val + golden * (b_val - a_val)
                    x_c = x + c_val * d
                    x_d = x + d_val * d
                    fc = func_eval(x_c)
                    fd = func_eval(x_d)
                    budget_used += 2

                    for _ in range(20):
                        if budget_used >= budget_remaining:
                            break
                        if fc < fd:
                            b_val = d_val
                            d_val = c_val
                            fd = fc
                            c_val = b_val - golden * (b_val - a_val)
                            fc = func_eval(x + c_val * d)
                            budget_used += 1
                        else:
                            a_val = c_val
                            c_val = d_val
                            fc = fd
                            d_val = a_val + golden * (b_val - a_val)
                            fd = func_eval(x + d_val * d)
                            budget_used += 1

                    # Best in this direction
                    if fc < fd:
                        t_best = c_val
                        fx_best_dir = fc
                    else:
                        t_best = d_val
                        fx_best_dir = fd

                    if fx_best_dir < fx:
                        x = x + t_best * d
                        fx = fx_best_dir
                        if fx < fx_min:
                            fx_min = fx
                            x_min = x.copy()

                # Check for convergence
                if abs(fx - fx_min) < 1e-10:
                    break

                # Update directions
                # The new direction is x_min - x_old
                x_old = x - (x_min - x)  # approximate
                # Actually, in Powell's method, the new direction replaces the one with the largest increase
                # For simplicity, add the direction from x to x_min
                new_dir = x_min - x
                norm_new = np.linalg.norm(new_dir)
                if norm_new > 1e-8:
                    new_dir /= norm_new
                    # Replace the direction with the largest step increase
                    # For simplicity, replace the last direction or find the one with max change
                    # We'll replace the direction that contributed least
                    # Simple approach: replace the direction most similar to new_dir
                    # Or just append and keep n directions
                    # Let's replace the direction with smallest magnitude change
                    # For now, replace direction 0
                    directions[0] = new_dir

                # Reorthogonalize if needed (skip for simplicity)
                if budget_used >= budget_remaining:
                    break

            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
            return best_x, best_fx, budget_used

        # Multi-start strategy
        # First, try center and a few random points
        rng = np.random.default_rng(42)

        # Start points: center, and random points
        start_points = [np.zeros(dim)]
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, dim))

        # Allocate budget: give more to later starts if earlier ones don't converge fast
        # Strategy: try each start with a portion of the budget
        # First start gets 50%, then split remaining among others
        n_starts = len(start_points)
        # Budget allocation: first start gets 60%, rest split evenly
        allocs = [int(budget * 0.6)]
        remaining = budget - allocs[0]
        for i in range(1, n_starts):
            allocs.append(remaining // (n_starts - 1))

        best_x = None
        best_fx = float('inf')

        for i, sp in enumerate(start_points):
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
            alloc = min(allocs[i], budget - calls[0])
            if alloc < 10:
                break
            x_res, fx_res, used = powell(safe_func, sp, alloc, dim)
            if fx_res < best_fx:
                best_fx = fx_res
                best_x = x_res

        # If we have budget left, do a final local refinement from best_x
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining_budget = budget - calls[0]
            if remaining_budget > 20:
                x_res, fx_res, used = powell(safe_func, best_x, remaining_budget, dim)
                if fx_res < best_fx:
                    best_fx = fx_res
                    best_x = x_res

        return best_x, best_fx
# EVOLVE-BLOCK-END
