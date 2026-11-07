"""Derivative-free BFGS with finite-difference gradients and multi-start strategy."""
import numpy as np
import time


class Optimizer:
    """Derivative-free BFGS with finite-difference gradients for Rosenbrock optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf
        call_count = [0]
        rng = np.random.default_rng(12345)

        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                raise RuntimeError("Budget exceeded")
            return float(func(np.asarray(x, dtype=np.float64)))

        def time_left():
            return (deadline - time.time()) > 3.0

        def budget_left():
            return (budget - call_count[0]) > 2

        def can_continue():
            return time_left() and budget_left()

        def finite_difference_gradient(x, f0, h=1e-5):
            """Compute gradient using forward differences."""
            grad = np.zeros(dim)
            for i in range(dim):
                x_perturbed = x.copy()
                x_perturbed[i] += h
                x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                f_perturbed = wrapped_func(x_perturbed)
                grad[i] = (f_perturbed - f0) / h
            return grad

        def bfgs_local(x0, max_iterations=50, tolerance=1e-10):
            """BFGS with finite-difference gradients."""
            x = x0.copy()
            f_curr = wrapped_func(x)
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()

            # Initialize inverse Hessian approximation
            H_inv = np.eye(dim)
            prev_grad = finite_difference_gradient(x, f_curr)
            prev_x = x.copy()

            for iteration in range(max_iterations):
                if not can_continue():
                    break

                # Check convergence
                if np.linalg.norm(prev_grad, ord=np.inf) < tolerance:
                    break

                # Compute search direction
                p = -H_inv @ prev_grad

                # Line search with backtracking
                alpha = 1.0
                c1 = 1e-4
                slope = np.dot(prev_grad, p)

                if slope >= 0:
                    # If not a descent direction, reset H_inv
                    H_inv = np.eye(dim)
                    p = -prev_grad
                    slope = np.dot(prev_grad, p)

                max_backtracks = 30
                backtrack_count = 0
                while backtrack_count < max_backtracks:
                    x_new = x + alpha * p
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = wrapped_func(x_new)

                    if f_new <= f_curr + c1 * alpha * slope:
                        break
                    alpha *= 0.5
                    backtrack_count += 1
                else:
                    # If line search fails, try a smaller step
                    alpha = 1e-6
                    x_new = x + alpha * p
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = wrapped_func(x_new)

                # Update best
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

                # Compute new gradient
                new_grad = finite_difference_gradient(x_new, f_new)

                # Check convergence
                if np.linalg.norm(new_grad, ord=np.inf) < tolerance:
                    x = x_new
                    f_curr = f_new
                    break

                # BFGS update
                s = x_new - x
                y = new_grad - prev_grad
                sy = np.dot(s, y)

                if sy > 1e-12:
                    # Update inverse Hessian approximation
                    rho = 1.0 / sy
                    H_inv = (I - rho * np.outer(s, y)) @ H_inv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

                x = x_new
                f_curr = f_new
                prev_grad = new_grad

            return x, f_curr

        def nelder_mead_local(x0, max_iterations=100, tolerance=1e-10):
            """Nelder-Mead simplex method."""
            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [wrapped_func(x0)]

            step_size = 0.5
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += step_size
                x_i = np.clip(x_i, -5.0, 5.0)
                simplex.append(x_i)
                f_values.append(wrapped_func(x_i))

            # Update best
            for i, f in enumerate(f_values):
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()

            for iteration in range(max_iterations):
                if not can_continue():
                    break

                # Sort by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check convergence
                if abs(f_values[-1] - f_values[0]) < tolerance:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                gamma = 1.0
                x_reflect = centroid + gamma * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = wrapped_func(x_reflect)

                if f_values[0] <= f_reflect < f_values[-2]:
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect
                elif f_reflect < f_values[0]:
                    # Expansion
                    gamma_exp = 2.0
                    x_expand = centroid + gamma_exp * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = wrapped_func(x_expand)

                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_values[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_values[-1] = f_reflect
                else:
                    # Contraction
                    gamma_contract = 0.5
                    x_contract = centroid + gamma_contract * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = wrapped_func(x_contract)

                    if f_contract < f_values[-1]:
                        simplex[-1] = x_contract
                        f_values[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_values[i] = wrapped_func(simplex[i])

                # Update best
                for i, f in enumerate(f_values):
                    if f < best_f:
                        best_f = f
                        best_x = simplex[i].copy()

            return simplex[0], f_values[0]

        # Phase 1: Diversified random sampling
        n_initial = min(15, max(5, budget // 30))
        for i in range(n_initial):
            if not can_continue():
                break

            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Phase 2: Primary BFGS from the best point
        remaining = budget - call_count[0] - 10
        if remaining > 20 and can_continue():
            max_iter = max(20, min(50, remaining // (dim + 2)))
            try:
                x_opt, f_opt = bfgs_local(best_x.copy(), max_iter)
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()
            except Exception:
                pass

        # Phase 3: Multi-start local refinement
        n_local_starts = min(6, max(2, (budget - call_count[0] - 20) // 50))

        for j in range(n_local_starts):
            if not can_continue():
                break

            # Create perturbations around the best point
            scale = 2.0 / (j + 1)
            x0 = best_x + rng.normal(0, scale, dim)
            x0 = np.clip(x0, -5.0, 5.0)

            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = budget - call_count[0] - 10
            if remaining > 15 and can_continue():
                max_iter = max(10, min(30, remaining // (dim + 2)))
                try:
                    # Alternate between BFGS and Nelder-Mead
                    if j % 2 == 0:
                        x_opt, f_opt = bfgs_local(x0.copy(), max_iter)
                    else:
                        x_opt, f_opt = nelder_mead_local(x0.copy(), max_iter)

                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = x_opt.copy()
                except Exception:
                    pass

        # Phase 4: Global exploration with random restarts
        n_global = min(3, max(1, (budget - call_count[0] - 30) // 60))
        for j in range(n_global):
            if not can_continue():
                break

            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = budget - call_count[0] - 10
            if remaining > 20 and can_continue():
                max_iter = max(15, min(40, remaining // (dim + 2)))
                try:
                    x_opt, f_opt = bfgs_local(x0.copy(), max_iter)
                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = x_opt.copy()
                except Exception:
                    pass

        # Phase 5: Final high-precision polish
        if best_x is not None and can_continue():
            remaining = budget - call_count[0] - 5
            if remaining > 10:
                max_iter_final = max(10, min(60, remaining // (dim + 2)))
                try:
                    x_opt, f_opt = bfgs_local(best_x.copy(), max_iter_final, tolerance=1e-12)
                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = x_opt.copy()
                except Exception:
                    # Fallback to Nelder-Mead
                    remaining = budget - call_count[0] - 3
                    if remaining > 5 and can_continue():
                        try:
                            x_opt, f_opt = nelder_mead_local(best_x.copy(), min(30, remaining // 2), tolerance=1e-12)
                            if f_opt < best_f:
                                best_f = f_opt
                                best_x = x_opt.copy()
                        except Exception:
                            pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
