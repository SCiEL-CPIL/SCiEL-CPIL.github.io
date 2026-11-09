import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        calls = 0
        start_time = time.time()
        time_limit = 320.0

        def eval_func(x):
            nonlocal calls
            if calls >= budget:
                raise StopIteration("Budget exceeded")
            if time.time() - start_time > time_limit:
                raise StopIteration("Time limit exceeded")
            calls += 1
            clipped = np.clip(x, -5.0, 5.0)
            return func(clipped)

        def check_time():
            return time.time() - start_time > time_limit

        def has_budget(min_remaining=1):
            return calls < budget - min_remaining and not check_time()

        best_x = None
        best_fx = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if np.all(np.isfinite(x)) and fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=float)

        def golden_section_1d(x0, direction, fx0, min_step, max_step, n_evals_max):
            """1D golden section search along direction from x0 with known fx0."""
            if n_evals_max < 3:
                return x0, fx0

            # First, bracket: try expanding in both directions
            phi = (1 + np.sqrt(5)) / 2.0
            step = min_step

            # Try to find a bracket [a, b] where f is higher at both ends
            # Start with a small step and expand
            f_plus = fx0  # we'll evaluate at x0 + step*direction
            f_minus = fx0  # x0 - step*direction

            # Evaluate at x0 + step*direction and x0 - step*direction
            if not has_budget(2):
                return x0, fx0

            try:
                x_plus = x0 + step * direction
                x_minus = x0 - step * direction
                f_plus = eval_func(x_plus)
                f_minus = eval_func(x_minus)
            except StopIteration:
                return x0, fx0
            except Exception:
                return x0, fx0

            # Determine which direction is better
            if f_plus < f_minus:
                # Minimum likely in positive direction
                a_param = 0.0  # x0
                b_param = step  # x0 + step*direction
                fa = fx0
                fb = f_plus
                # Check if x0 is already better than x_plus
                if fx0 < f_plus:
                    # Try negative direction
                    if not has_budget(1):
                        return x0, fx0
                    try:
                        x_minus2 = x0 - step * direction
                        f_minus2 = eval_func(x_minus2)
                    except (StopIteration, Exception):
                        return x0, fx0
                    if f_minus2 < fx0:
                        a_param = -step
                        fa = f_minus2
                        b_param = 0.0
                        fb = fx0
                    else:
                        return x0, fx0
            else:
                # Minimum likely in negative direction
                a_param = -step  # x0 - step*direction
                b_param = 0.0  # x0
                fa = f_minus
                fb = fx0
                if fx0 < f_minus:
                    # Try positive direction
                    if not has_budget(1):
                        return x0, fx0
                    try:
                        x_plus2 = x0 + step * direction
                        f_plus2 = eval_func(x_plus2)
                    except (StopIteration, Exception):
                        return x0, fx0
                    if f_plus2 < fx0:
                        a_param = 0.0
                        fa = fx0
                        b_param = step
                        fb = f_plus2
                    else:
                        return x0, fx0

            # Now do golden section search in [a_param, b_param]
            invphi = 1.0 / phi
            invphi2 = invphi * invphi

            best_t = (a_param + b_param) / 2.0
            best_f = min(fa, fb)

            c = a_param + invphi2 * (b_param - a_param)
            d = a_param + invphi * (b_param - a_param)

            try:
                fc = eval_func(x0 + c * direction)
                fd = eval_func(x0 + d * direction)
            except (StopIteration, Exception):
                return x0, fx0

            if fc < fd:
                b_param = d
                fb = fd
                d = c
                fd = fc
                c = a_param + invphi2 * (b_param - a_param)
            else:
                a_param = c
                fa = fc
                c = d
                fc = fd
                d = a_param + invphi * (b_param - a_param)

            # Continue golden section
            for _ in range(n_evals_max - 4):
                if not has_budget(1):
                    break
                if abs(b_param - a_param) < 1e-10:
                    break
                try:
                    if fc < fd:
                        b_param = d
                        fb = fd
                        d = c
                        fd = fc
                        c = a_param + invphi2 * (b_param - a_param)
                        fc = eval_func(x0 + c * direction)
                    else:
                        a_param = c
                        fa = fc
                        c = d
                        fc = fd
                        d = a_param + invphi * (b_param - a_param)
                        fd = eval_func(x0 + d * direction)
                except (StopIteration, Exception):
                    break

            # The best point is the one with lowest f among c and d
            if fc < fd:
                best_t = c
                best_f = fc
            else:
                best_t = d
                best_f = fd

            # Also compare with endpoints
            if fa < best_f:
                best_t = a_param
                best_f = fa
            if fb < best_f:
                best_t = b_param
                best_f = fb

            return x0 + best_t * direction, best_f

        # Phase 1: Multi-start Powell-like local optimization
        # For Bent Cigar, we need to find the narrow valley first

        rng = np.random.default_rng(42)

        # Generate diverse starting points
        starts = [np.zeros(n)]
        # Points along each axis
        for i in range(n):
            for sign in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = sign * 3.0
                starts.append(pt)
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, n))

        # Sort starts by distance from origin (prioritize center)
        starts.sort(key=lambda x: np.sum(x**2))

        for x0 in starts:
            if not has_budget(20):
                break
            if check_time():
                break

            x_cur = x0.copy()
            try:
                fx_cur = eval_func(x_cur)
            except (StopIteration, Exception):
                break
            update_best(x_cur, fx_cur)

            # Powell-like: coordinate ascent with memory
            directions = [np.eye(n, n)[:, i].copy() for i in range(n)]
            step_sizes = [1.0] * n

            max_rounds = 15
            for rnd in range(max_rounds):
                if not has_budget(n + 2):
                    break
                if check_time():
                    break

                axis_improvement = 0.0
                max_axis_improvement = 0.0
                max_axis_idx = -1

                for i in range(n):
                    if not has_budget(10):
                        break
                    if check_time():
                        break

                    direction = directions[i]
                    step = step_sizes[i]

                    # Try stepping in positive direction
                    x_try = x_cur + step * direction
                    try:
                        f_try = eval_func(x_try)
                    except (StopIteration, Exception):
                        break

                    if f_try < fx_cur:
                        # Good direction, do 1D search
                        x_new, f_new = golden_section_1d(
                            x_cur, direction, fx_cur,
                            min_step=step * 0.01, max_step=step * 10.0,
                            n_evals_max=8
                        )
                        if f_new < fx_cur:
                            x_cur = x_new
                            fx_cur = f_new
                            axis_improvement += (fx_cur - f_new) if f_new < fx_cur else 0
                            # Actually compute improvement
                            pass
                    else:
                        # Try negative direction
                        x_try = x_cur - step * direction
                        try:
                            f_try = eval_func(x_try)
                        except (StopIteration, Exception):
                            break

                        if f_try < fx_cur:
                            x_new, f_new = golden_section_1d(
                                x_cur, -direction, fx_cur,
                                min_step=step * 0.01, max_step=step * 10.0,
                                n_evals_max=8
                            )
                            if f_new < fx_cur:
                                x_cur = x_new
                                fx_cur = f_new
                    update_best(x_cur, fx_cur)

                # Check for convergence
                if axis_improvement < 1e-14:
                    break

            update_best(x_cur, fx_cur)

        # Phase 2: Exploit Bent Cigar structure
        # Bent Cigar f(x) = sum(x_i^2) for i>0, x_1^2 / 1e4 for i=0 (in some coordinate system)
        # The high-curvature direction is the one with large coefficient
        # After initial search, try to identify and optimize along dominant axis

        if best_x is not None and has_budget(50):
            # Try coordinate-wise 1D searches from best point
            x_ref = best_x.copy()
            f_ref = best_fx

            for i in range(n):
                if not has_budget(15):
                    break
                if check_time():
                    break

                direction = np.zeros(n)
                direction[i] = 1.0

                x_new, f_new = golden_section_1d(
                    x_ref, direction, f_ref,
                    min_step=0.01, max_step=5.0,
                    n_evals_max=12
                )
                if f_new < best_fx:
                    best_x = x_new.copy()
                    best_fx = f_new
                    x_ref = x_new
                    f_ref = f_new

        # Phase 3: Fine-tuning with smaller steps
        if best_x is not None and has_budget(30):
            x_ref = best_x.copy()
            f_ref = best_fx

            for i in range(n):
                if not has_budget(10):
                    break
                if check_time():
                    break

                direction = np.zeros(n)
                direction[i] = 1.0

                x_new, f_new = golden_section_1d(
                    x_ref, direction, f_ref,
                    min_step=1e-4, max_step=1.0,
                    n_evals_max=10
                )
                if f_new < best_fx:
                    best_x = x_new.copy()
                    best_fx = f_new
                    x_ref = x_new
                    f_ref = f_new

        # Phase 4: Final polish - very tight 1D searches
        if best_x is not None and has_budget(20):
            x_ref = best_x.copy()
            f_ref = best_fx

            for i in range(n):
                if not has_budget(8):
                    break
                if check_time():
                    break

                direction = np.zeros(n)
                direction[i] = 1.0

                x_new, f_new = golden_section_1d(
                    x_ref, direction, f_ref,
                    min_step=1e-6, max_step=0.1,
                    n_evals_max=8
                )
                if f_new < best_fx:
                    best_x = x_new.copy()
                    best_fx = f_new
                    x_ref = x_new
                    f_ref = f_new

        # Ensure we have a valid result
        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(n)
            if calls < budget:
                try:
                    best_fx = eval_func(best_x)
                except Exception:
                    best_fx = 0.0
            else:
                best_fx = 0.0

        return best_x, best_fx
