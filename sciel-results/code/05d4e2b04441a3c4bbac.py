import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def powell(x_start, max_calls=150):
            """Powell's method with quadratic interpolation along conjugate directions."""
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() >= deadline:
                return x_start, np.inf

            x = np.clip(np.asarray(x_start, dtype=np.float64).copy(), -5.0, 5.0)
            f = try_call(x)
            if f is None:
                return x, np.inf

            # Initial directions: coordinate axes
            directions = [np.eye(dim)[i] for i in range(dim)]

            tol = 1e-12
            max_iter = 30

            for _ in range(max_iter):
                if calls >= max_calls or time.time() >= deadline:
                    break

                x_min = x.copy()
                f_min = f
                max_delta = 0.0

                for i in range(dim):
                    if calls >= max_calls or time.time() >= deadline:
                        break
                    # Line search along direction i using quadratic interpolation
                    d = directions[i]
                    # Find a good step size by trying a few candidates
                    # First, find the minimum along this line via bisection/quadratic
                    # Try initial step
                    step = 0.1
                    # Evaluate at x, x+d*step, x-d*step to build quadratic
                    x0_pt = x.copy()
                    f0 = f

                    # Try positive direction
                    x1_pt = x0_pt + d * step
                    x1_pt = np.clip(x1_pt, -5.0, 5.0)
                    f1 = try_call(x1_pt)
                    if f1 is None:
                        break

                    if f1 < f0:
                        # Try negative direction too
                        x2_pt = x0_pt - d * step
                        x2_pt = np.clip(x2_pt, -5.0, 5.0)
                        f2 = try_call(x2_pt)
                        if f2 is None:
                            break
                        # Quadratic interpolation: minimize f(t) ≈ a*t^2 + b*t + c
                        # Using points at t=-step, 0, step with values f2, f0, f1
                        # f(t) = a*t^2 + b*t + c
                        # f(0) = c = f0
                        # f(step) = a*step^2 + b*step + f0 = f1
                        # f(-step) = a*step^2 - b*step + f0 = f2
                        # => a = (f1 + f2 - 2*f0) / (2*step^2)
                        #    b = (f1 - f2) / (2*step)
                        # min at t* = -b/(2a)
                        denom = 2.0 * step * step
                        a_q = (f1 + f2 - 2.0 * f0) / denom
                        b_q = (f1 - f2) / (2.0 * step)
                        if abs(a_q) > 1e-30:
                            t_star = -b_q / (2.0 * a_q)
                        else:
                            t_star = -step if f1 < f2 else step
                        # Clamp t_star to reasonable range
                        t_star = np.clip(t_star, -5.0 * step, 5.0 * step)
                        if abs(t_star) > 1e-14:
                            x_new = x0_pt + d * t_star
                            x_new = np.clip(x_new, -5.0, 5.0)
                            f_new = try_call(x_new)
                            if f_new is not None and f_new < f_min:
                                # Check if we should take this step
                                delta = f_min - f_new
                                if delta > max_delta:
                                    max_delta = delta
                                x_min = x_new.copy()
                                f_min = f_new
                                x = x_new
                                f = f_new
                        elif f1 < f2:
                            # Linear: take positive step
                            if f1 < f_min:
                                delta = f_min - f1
                                if delta > max_delta:
                                    max_delta = delta
                                x = x1_pt.copy()
                                f = f1
                                x_min = x.copy()
                                f_min = f
                        else:
                            if f2 < f_min:
                                delta = f_min - f2
                                if delta > max_delta:
                                    max_delta = delta
                                x = x2_pt.copy()
                                f = f2
                                x_min = x.copy()
                                f_min = f
                    else:
                        # f1 >= f0, try negative
                        x2_pt = x0_pt - d * step
                        x2_pt = np.clip(x2_pt, -5.0, 5.0)
                        f2 = try_call(x2_pt)
                        if f2 is None:
                            break
                        if f2 < f0:
                            # Try further negative
                            x3_pt = x0_pt - d * 2.0 * step
                            x3_pt = np.clip(x3_pt, -5.0, 5.0)
                            f3 = try_call(x3_pt)
                            if f3 is None:
                                break
                            # Quadratic with points at t=0, -step, -2*step
                            # f(0)=f0, f(-step)=f2, f(-2*step)=f3
                            # a = (f3 - 2*f2 + f0) / (2*step^2)
                            # b = (f3 - f0) / (2*step)  ... let's be careful
                            # Actually: f(t) = a*t^2 + b*t + c, c=f0
                            # f(-step) = a*step^2 - b*step + f0 = f2
                            # f(-2*step) = 4*a*step^2 - 2*b*step + f0 = f3
                            # From eq1: a*step^2 - b*step = f2 - f0
                            # From eq2: 4*a*step^2 - 2*b*step = f3 - f0
                            # Multiply eq1 by 2: 2*a*step^2 - 2*b*step = 2*(f2-f0)
                            # Subtract from eq2: 2*a*step^2 = f3 - f0 - 2*f2 + 2*f0 = f3 + f0 - 2*f2
                            # a = (f3 + f0 - 2*f2) / (2*step^2)
                            # b*step = a*step^2 - (f2 - f0) = (f3+f0-2*f2)/2 - f2 + f0 = (f3+f0-2*f2-2*f2+2*f0)/2 = (f3+3*f0-4*f2)/2
                            # b = (f3+3*f0-4*f2)/(2*step)
                            # t* = -b/(2a)
                            a_q = (f3 + f0 - 2.0 * f2) / (2.0 * step * step)
                            b_q = (f3 + 3.0 * f0 - 4.0 * f2) / (2.0 * step)
                            if abs(a_q) > 1e-30:
                                t_star = -b_q / (2.0 * a_q)
                            else:
                                t_star = -step
                            t_star = np.clip(t_star, -5.0 * step, 0.0)
                            if abs(t_star) > 1e-14:
                                x_new = x0_pt + d * t_star
                                x_new = np.clip(x_new, -5.0, 5.0)
                                f_new = try_call(x_new)
                                if f_new is not None and f_new < f_min:
                                    delta = f_min - f_new
                                    if delta > max_delta:
                                        max_delta = delta
                                    x_min = x_new.copy()
                                    f_min = f_new
                                    x = x_new
                                    f = f_new
                            elif f2 < f3:
                                if f2 < f_min:
                                    delta = f_min - f2
                                    if delta > max_delta:
                                        max_delta = delta
                                    x = x2_pt.copy()
                                    f = f2
                                    x_min = x.copy()
                                    f_min = f
                            else:
                                if f3 < f_min:
                                    delta = f_min - f3
                                    if delta > max_delta:
                                        max_delta = delta
                                    x = x3_pt.copy()
                                    f = f3
                                    x_min = x.copy()
                                    f_min = f

                # Check convergence
                if max_delta < tol:
                    break

                # Update directions: replace the direction with largest delta
                # The new direction is x_new - x_old (the overall movement)
                # Actually in Powell's method, the new direction is the displacement
                # from the start of this iteration to the end
                # Let's track the original x at start of iteration
                # We need to save x at start
                # For simplicity, use the net displacement
                # Find which direction gave max_delta - we didn't track it explicitly
                # Let's just replace the last direction with the net displacement
                # This is a simplified version
                # Better: track which direction had the largest improvement
                # For now, use a heuristic: replace direction dim-1 with net displacement
                # Actually, let's track it properly by re-running with tracking

                # Simplified: replace one direction with the net movement
                # Net movement from start of iteration
                # We didn't save start, so use x_min - (x before this iteration)
                # This is getting complex; let's use a simpler update
                # Replace direction with max improvement - we'll track it

                # For a cleaner implementation, let's restart with proper tracking
                pass

            return x, f

        def powell_proper(x_start, max_calls=150):
            """Powell's method with proper direction set update."""
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() >= deadline:
                return x_start, np.inf

            x = np.clip(np.asarray(x_start, dtype=np.float64).copy(), -5.0, 5.0)
            f = try_call(x)
            if f is None:
                return x, np.inf

            # Initial directions
            directions = [np.eye(dim)[i].copy() for i in range(dim)]

            tol = 1e-14
            max_iter = 40

            for _ in range(max_iter):
                if calls >= max_calls or time.time() >= deadline:
                    break

                x_start_iter = x.copy()
                f_start_iter = f
                x_min = x.copy()
                f_min = f
                max_delta = 0.0
                max_dir_idx = -1

                for i in range(dim):
                    if calls >= max_calls or time.time() >= deadline:
                        break
                    d = directions[i]
                    x_line_start = x.copy()
                    f_line_start = f

                    # Line search: find min along x + t*d
                    # Try t = 0.1, 0.01, 0.5, 1.0, -0.1, -0.5, -1.0
                    best_t = 0.0
                    best_f_line = f_line_start

                    # First try t=0.1
                    step = 0.5
                    for t_candidate in [0.5, 0.1, 1.0, 0.2, -0.5, -0.1, -1.0, 2.0, -2.0]:
                        if calls >= max_calls or time.time() >= deadline:
                            break
                        x_test = x_line_start + d * t_candidate
                        x_test = np.clip(x_test, -5.0, 5.0)
                        f_test = try_call(x_test)
                        if f_test is None:
                            break
                        if f_test < best_f_line:
                            best_f_line = f_test
                            best_t = t_candidate

                    # Refine with quadratic interpolation if we found a better point
                    if best_t != 0.0 and best_f_line < f_line_start - 1e-15:
                        # We have f(0) = f_line_start, f(best_t) = best_f_line
                        # Try another point to get quadratic
                        t2 = best_t * 2.0
                        if calls < max_calls and time.time() < deadline:
                            x_test2 = x_line_start + d * t2
                            x_test2 = np.clip(x_test2, -5.0, 5.0)
                            f_test2 = try_call(x_test2)
                            if f_test2 is not None:
                                # Quadratic through (0, f0), (t1, f1), (t2, f2)
                                f0 = f_line_start
                                f1 = best_f_line
                                f2 = f_test2
                                t1 = best_t
                                # f(t) = a*t^2 + b*t + c, c = f0
                                # a*t1^2 + b*t1 = f1 - f0
                                # a*t2^2 + b*t2 = f2 - f0
                                # Solve:
                                # a = ((f1-f0)/t1 - (f2-f0)/t2) / (t1 - t2)
                                # b = (f1-f0)/t1 - a*t1
                                if abs(t2 - t1) > 1e-15 and abs(t1) > 1e-15 and abs(t2) > 1e-15:
                                    a_q = ((f1 - f0) / t1 - (f2 - f0) / t2) / (t2 - t1)
                                    b_q = (f1 - f0) / t1 - a_q * t1
                                    if abs(a_q) > 1e-30:
                                        t_star = -b_q / (2.0 * a_q)
                                        # Clamp
                                        t_star = np.clip(t_star, -5.0, 5.0)
                                        if calls < max_calls and time.time() < deadline:
                                            x_star = x_line_start + d * t_star
                                            x_star = np.clip(x_star, -5.0, 5.0)
                                            f_star = try_call(x_star)
                                            if f_star is not None and f_star < best_f_line:
                                                best_f_line = f_star
                                                best_t = t_star

                        # Apply the best step
                        if best_t != 0.0:
                            x = x_line_start + d * best_t
                            x = np.clip(x, -5.0, 5.0)
                            f = best_f_line
                            delta = f_start_iter - f
                            if delta > max_delta:
                                max_delta = delta
                                max_dir_idx = i
                                x_min = x.copy()
                                f_min = f

                if max_delta < tol:
                    break

                # Update direction set
                if max_dir_idx >= 0:
                    # The new direction is the net displacement from start of iteration
                    # to the current best
                    new_dir = x_min - x_start_iter
                    norm_new = np.linalg.norm(new_dir)
                    if norm_new > 1e-14:
                        directions[max_dir_idx] = new_dir / norm_new

            return x, f

        # ---- Phase 1: Coarse global grid/random search ----
        # Evaluate origin
        x0 = np.zeros(dim)
        try_call(x0)

        # Random points in the box
        n_random = 20
        for _ in range(n_random):
            if calls >= budget:
                break
            try_call(rng.uniform(-4, 4, dim))

        # Also try a coarse grid near the boundaries and center
        # Simple: a few structured points
        for val in [-3, -1, 0, 1, 3]:
            if calls >= budget:
                break
            x_grid = np.full(dim, val)
            try_call(x_grid)

        # ---- Phase 2: Powell's method from multiple starting points ----
        # Collect candidate starting points: best found + nearby perturbations
        candidates = [best_x.copy()]
        # Add perturbations around the best
        for _ in range(4):
            if calls >= budget:
                break
            x_pert = best_x + rng.normal(0, 1.0, dim)
            x_pert = np.clip(x_pert, -5.0, 5.0)
            candidates.append(x_pert)

        # Run Powell from each candidate with a budget allocation
        remaining = budget - calls
        n_powell = len(candidates)
        budget_per_powell = max(10, remaining // n_powell)

        for i, cand in enumerate(candidates):
            if calls >= budget or time.time() >= deadline:
                break
            # Allocate more budget to the first (best) candidate
            if i == 0:
                bp = min(budget_per_powell * 2, budget - calls)
            else:
                bp = min(budget_per_powell, budget - calls)
            if bp < 10:
                break
            powell_proper(cand, max_calls=budget - calls)

        # ---- Phase 3: Final polish with tight Powell from the best ----
        if calls < budget and time.time() < deadline:
            powell_proper(best_x.copy(), max_calls=budget - calls)

        return best_x, best_f
