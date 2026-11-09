"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using random initialization, (1+1)-ES for global exploration,
    followed by Nelder-Mead for local refinement.
    Designed for BBOB f14 (Different Powers) in 5 dimensions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0
        rng = np.random.default_rng(0)

        # Wrapper to count calls and clip to box
        def _eval(x):
            nonlocal calls
            if calls >= budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(x, -5.0, 5.0)
            calls += 1
            return func(x_clipped)

        best_x = None
        best_f = np.inf

        # 1. Phase 0: Random Initialization
        # Sample a few random points to find a good starting basin.
        # This is crucial for rotated/shifted functions where origin might be far.
        n_init = 10
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = _eval(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(dim)
            best_f = _eval(x)
            best_x = x.copy()

        # 2. Phase 1: (1+1)-ES with 1/5 success rule
        # This provides robust global exploration and adaptive step sizing.
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        success_count = 0
        window_size = 10
        target_success_rate = 0.2

        # Run (1+1)-ES for the first 50% of the budget
        phase1_budget = budget // 2

        for i in range(phase1_budget - calls):
            if time.time() > deadline or calls >= budget - 10:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = _eval(y)

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                pass

            # Adapt sigma every window_size steps
            if (i + 1) % window_size == 0:
                success_rate = success_count / window_size
                if success_rate > target_success_rate:
                    sigma *= 1.25
                elif success_rate < target_success_rate:
                    sigma *= 0.8
                success_count = 0

            if calls >= budget:
                break

        # 3. Phase 2: Local refinement with Nelder-Mead
        # Use the remaining budget to polish the best point found.
        # We implement a custom Nelder-Mead to strictly control the number of function calls.

        remaining = budget - calls
        if remaining > 10:
            # Initialize simplex around the best point
            center = best_x.copy()
            simplex = [(center.copy(), best_f)]

            # Add vertices with a step size appropriate for NM
            # Start with a moderate step size
            step = 0.5
            
            for i in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                pt = center.copy()
                pt[i] += step
                fval = _eval(pt)
                simplex.append((pt.copy(), fval))

            # Sort simplex
            simplex.sort(key=lambda p: p[1])

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5

            # Run NM for the remaining budget
            nm_iterations = 0
            max_nm_iter = 200  # Cap iterations to avoid infinite loops, budget is the real limit

            while nm_iterations < max_nm_iter:
                if time.time() > deadline or calls >= budget:
                    break

                nm_iterations += 1

                # Get best, worst, second worst
                x_worst, f_worst = simplex[-1]
                x_second, f_second = simplex[-2]

                # Centroid of all but worst
                x_centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)

                # Reflection
                if calls + 1 > budget:
                    break
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = _eval(x_ref)

                if f_ref < simplex[0][1]:  # Better than best
                    # Expansion
                    if calls + 1 > budget:
                        break
                    x_exp = x_centroid + gamma * (x_centroid - x_worst)
                    f_exp = _eval(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = (x_exp, f_exp)
                    else:
                        simplex[-1] = (x_ref, f_ref)
                elif f_ref < f_second:  # Between best and second worst
                    simplex[-1] = (x_ref, f_ref)
                else:  # Worse than second worst
                    # Contraction
                    if calls + 1 > budget:
                        break
                    if f_ref < f_worst:
                        # Outer contraction
                        x_con = x_centroid + 0.5 * (x_ref - x_centroid)
                    else:
                        # Inner contraction
                        x_con = x_centroid - 0.5 * (x_worst - x_centroid)

                    f_con = _eval(x_con)

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = (x_con, f_con)
                    else:
                        # Shrink
                        # Shrink requires d evaluations. Check if we have budget.
                        if calls + dim > budget:
                            break
                        
                        x_best, f_best_cur = simplex[0]
                        new_simplex = [(x_best, f_best_cur)]
                        shrink_completed = True
                        for i, (x_i, f_i) in enumerate(simplex):
                            if i == 0:
                                continue
                            if calls >= budget:
                                shrink_completed = False
                                break
                            x_shrink = x_best + rho * (x_i - x_best)
                            f_shrink = _eval(x_shrink)
                            new_simplex.append((x_shrink, f_shrink))
                        
                        if shrink_completed:
                            simplex = new_simplex
                        else:
                            break

                # Update best if improved
                if simplex[0][1] < best_f:
                    best_f = simplex[0][1]
                    best_x = simplex[0][0].copy()

                # Re-sort
                simplex.sort(key=lambda p: p[1])

                # Convergence check: if simplex is very small, break
                x_min = np.min([p[0] for p in simplex], axis=0)
                x_max = np.max([p[0] for p in simplex], axis=0)
                if np.max(x_max - x_min) < 1e-8:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
