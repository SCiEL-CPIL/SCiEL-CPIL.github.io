"""Optimizer for BBOB-style f22 (Gallagher 21 peaks) using Powell's method with restarts."""
import numpy as np
import time


class Optimizer:
    """Powell's method with multiple restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Track best solution found so far
        best_x = None
        best_f = np.inf

        # Call counter
        call_count = 0

        # Wrap function to count calls and handle edge cases
        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=np.float64))
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return float(val)
            except Exception:
                return 1e10

        # Powell's method implementation
        def powell_method(x0, max_calls, rng):
            """
            Powell's method for minimization.
            Returns (best_x, best_f, calls_used).
            """
            nonlocal best_x, best_f
            dim = self.dim
            x = np.array(x0, dtype=np.float64)
            fx = safe_func(x)
            calls_used = 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Initial directions: coordinate axes
            directions = [np.zeros(dim) for _ in range(dim)]
            for i in range(dim):
                directions[i][i] = 1.0

            # Powell's algorithm
            max_iterations = 50  # limit iterations to avoid excessive calls
            for iteration in range(max_iterations):
                if calls_used >= max_calls:
                    break
                if time.time() > deadline:
                    break

                # Step along each direction
                x_prev = x.copy()
                fx_prev = fx
                min_delta = -np.inf
                min_idx = -1

                for i in range(dim):
                    if calls_used >= max_calls:
                        break
                    if time.time() > deadline:
                        break

                    # Find minimum along direction[i] using golden section or bisection
                    d = directions[i]

                    # Determine search interval by expanding from x
                    # Try small step first
                    step = 0.1
                    # Expand to find bracket
                    x_test = x + step * d
                    fx_test = safe_func(x_test)
                    calls_used += 1
                    if fx_test < fx:
                        # Keep expanding
                        while calls_used < max_calls and time.time() < deadline:
                            x_test2 = x + 2 * step * d
                            fx_test2 = safe_func(x_test2)
                            calls_used += 1
                            if fx_test2 < fx_test:
                                step *= 2
                                x_test = x_test2
                                fx_test = fx_test2
                            else:
                                break
                        # Now we have a bracket [x, x_test] with minimum inside
                        # Use golden section search
                        a = 0.0
                        b = step
                        fa = fx
                        fb = fx_test
                        # Golden section
                        gr = (np.sqrt(5) - 1) / 2
                        max_gs = 8
                        for _ in range(max_gs):
                            if calls_used >= max_calls or time.time() > deadline:
                                break
                            c = b - gr * (b - a)
                            dc = c * d
                            x_c = x + dc
                            fc = safe_func(x_c)
                            calls_used += 1
                            d2 = a + gr * (b - a)
                            dd2 = d2 * d
                            x_d = x + dd2
                            fd = safe_func(x_d)
                            calls_used += 1
                            if fc < fd:
                                b = d2
                                fb = fd
                            else:
                                a = c
                                fa = fc
                            if abs(b - a) < 1e-6:
                                break
                        # Take the better of c and d points
                        # Actually, let's just take the midpoint of the final bracket
                        x_opt = x + (a + b) / 2 * d
                        fx_opt = safe_func(x_opt)
                        calls_used += 1
                    else:
                        # Try negative direction
                        x_test_neg = x - step * d
                        fx_test_neg = safe_func(x_test_neg)
                        calls_used += 1
                        if fx_test_neg < fx:
                            while calls_used < max_calls and time.time() < deadline:
                                x_test2_neg = x - 2 * step * d
                                fx_test2_neg = safe_func(x_test2_neg)
                                calls_used += 1
                                if fx_test2_neg < fx_test_neg:
                                    step *= 2
                                    x_test_neg = x_test2_neg
                                    fx_test_neg = fx_test2_neg
                                else:
                                    break
                            a = -step
                            b = 0.0
                            fa = fx_test_neg
                            fb = fx
                            gr = (np.sqrt(5) - 1) / 2
                            max_gs = 8
                            for _ in range(max_gs):
                                if calls_used >= max_calls or time.time() > deadline:
                                    break
                                c = b - gr * (b - a)
                                dc = c * d
                                x_c = x + dc
                                fc = safe_func(x_c)
                                calls_used += 1
                                d2 = a + gr * (b - a)
                                dd2 = d2 * d
                                x_d = x + dd2
                                fd = safe_func(x_d)
                                calls_used += 1
                                if fc < fd:
                                    b = d2
                                    fb = fd
                                else:
                                    a = c
                                    fa = fc
                                if abs(b - a) < 1e-6:
                                    break
                            x_opt = x + (a + b) / 2 * d
                            fx_opt = safe_func(x_opt)
                            calls_used += 1
                        else:
                            # No improvement, keep x
                            x_opt = x.copy()
                            fx_opt = fx

                    if fx_opt < fx:
                        delta = fx_opt - fx
                        if delta < min_delta:
                            min_delta = delta
                            min_idx = i
                        x = x_opt
                        fx = fx_opt
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

                # Check if significant improvement
                if min_idx == -1 or min_delta >= -1e-10:
                    break

                # Add new direction
                new_dir = x - x_prev
                norm = np.linalg.norm(new_dir)
                if norm > 1e-10:
                    new_dir = new_dir / norm
                # Remove the worst direction (the one with min_delta)
                directions.pop(min_idx)
                directions.append(new_dir)

            return x, fx, calls_used

        # Multiple restarts with different initial points
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        # Use a combination of random points and structured points
        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Random points
        for _ in range(20):
            start_points.append(rng.uniform(-5, 5, dim))

        # Structured grid-like points
        for i in range(-2, 3):
            for j in range(-2, 3):
                pt = np.zeros(dim)
                pt[0] = i
                pt[1] = j
                start_points.append(pt)

        # Shuffle start points
        rng.shuffle(start_points)

        # Allocate budget across restarts
        # Give more budget to earlier restarts
        n_restarts = min(len(start_points), 15)
        # Budget allocation: decreasing
        budgets = []
        remaining = budget
        for i in range(n_restarts):
            if i == n_restarts - 1:
                b = remaining
            else:
                b = max(10, int(remaining * 0.6))
                remaining -= b
            budgets.append(b)

        for i in range(n_restarts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            x0 = start_points[i]
            max_calls_for_this = min(budgets[i], budget - call_count)
            if max_calls_for_this < 5:
                break

            _, _, _ = powell_method(x0, call_count + max_calls_for_this, rng)

            # If we're close to budget, do a final local refinement
            if call_count < budget and time.time() < deadline:
                # Do a few more random perturbations around best_x
                if best_x is not None:
                    for _ in range(5):
                        if call_count >= budget or time.time() > deadline:
                            break
                        perturbation = 0.1 * rng.standard_normal(dim)
                        x_try = np.clip(best_x + perturbation, -5, 5)
                        f_try = safe_func(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()

        # Final: return best solution
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
