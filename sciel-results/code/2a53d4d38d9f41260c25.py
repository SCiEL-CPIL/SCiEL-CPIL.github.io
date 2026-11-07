<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based multi-start optimizer for BBOB f9 (rotated Rosenbrock) in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evals = 0

        def evaluate(x):
            """Evaluate function and track best."""
            nonlocal best_f, best_x, evals
            x = np.clip(x, -5, 5)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def gradient_descent(x0, max_iters):
            """Variable-step gradient descent with central differences and backtracking."""
            nonlocal best_f, best_x, evals

            x = x0.copy()
            f = evaluate(x)

            # Initial step size
            step = 0.5
            min_step = 1e-8
            max_step = 4.0

            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if evals >= budget:
                    break

                # Compute gradient via central differences
                if evals + 2 * n > budget:
                    break

                grad = np.zeros(n)
                for i in range(n):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += step * 0.01
                    x_minus[i] -= step * 0.01
                    x_plus = np.clip(x_plus, -5, 5)
                    x_minus = np.clip(x_minus, -5, 5)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    evals += 2
                    grad[i] = (f_plus - f_minus) / (2 * step * 0.01)

                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-12:
                    break

                # Normalize gradient direction
                direction = -grad / grad_norm

                # Backtracking line search
                alpha = step
                c1 = 1e-4  # Armijo condition
                rho = 0.5

                # Try current step
                x_trial = x + alpha * direction
                x_trial = np.clip(x_trial, -5, 5)
                if evals >= budget:
                    break
                f_trial = func(x_trial)
                evals += 1
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()

                # Armijo condition: f_trial <= f + c1 * alpha * grad_norm * (-1)
                # i.e., f_trial <= f - c1 * alpha * grad_norm
                if f_trial <= f - c1 * alpha * grad_norm:
                    # Accept step
                    x = x_trial
                    f = f_trial
                    # Increase step slightly if we made good progress
                    if f_trial < f - 0.5 * c1 * alpha * grad_norm:
                        step = min(step * 1.1, max_step)
                else:
                    # Backtrack
                    while alpha > min_step:
                        alpha *= rho
                        x_trial = x + alpha * direction
                        x_trial = np.clip(x_trial, -5, 5)
                        if evals >= budget:
                            break
                        f_trial = func(x_trial)
                        evals += 1
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()
                        if f_trial <= f - c1 * alpha * grad_norm:
                            x = x_trial
                            f = f_trial
                            break
                    else:
                        # Couldn't find a good step, reduce step size
                        step *= 0.5
                        if step < min_step:
                            break

            return x, f
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """L-BFGS multi-start optimizer for BBOB f9 (rotated Rosenbrock) in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evals = 0

        def evaluate(x):
            """Evaluate function and track best."""
            nonlocal best_f, best_x, evals
            x = np.clip(x, -5, 5)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def compute_gradient(x):
            """Compute gradient via central differences."""
            nonlocal evals
            h = 1e-5
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                evals += 2
                grad[i] = (f_plus - f_minus) / (2 * h)
            return grad

        def lbfgs(x0, max_iters):
            """L-BFGS with two-loop recursion and Wolfe line search."""
            nonlocal best_f, best_x, evals

            m = 10  # memory size
            x = x0.copy()
            f = evaluate(x)
            g = compute_gradient(x)

            # L-BFGS history
            s_list = []  # step vectors
            y_list = []  # gradient differences
            rho_list = []  # 1/(y^T s)

            gamma = 1.0 / (np.dot(g, g) + 1e-12)

            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if evals + 2 * n + 10 > budget:
                    break

                # Two-loop recursion to compute search direction
                r = -g.copy()
                alphas = np.zeros(len(s_list))
                qs = np.zeros_like(r)

                for i in range(len(s_list) - 1, -1, -1):
                    alphas[i] = rho_list[i] * np.dot(y_list[i], r)
                    qs = qs + alphas[i] * s_list[i]

                r = r + gamma * qs

                for i in range(len(s_list)):
                    beta = rho_list[i] * np.dot(s_list[i], r)
                    r = r + (alphas[i] - beta) * y_list[i]

                direction = -r

                # Wolfe line search
                c1 = 1e-4
                c2 = 0.9
                dg = np.dot(g, direction)
                if dg >= 0:
                    direction = -g
                    dg = -np.dot(g, g)

                alpha = 1.0
                f_trial = f
                x_trial = x
                g_trial = g
                accepted = False

                for ls_iter in range(30):
                    if evals >= budget:
                        break
                    x_trial = x + alpha * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    evals += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()

                    if f_trial <= f + c1 * alpha * dg:
                        # Check curvature condition
                        g_trial = compute_gradient(x_trial)
                        dg_trial = np.dot(g_trial, direction)
                        if dg_trial >= c2 * dg:
                            accepted = True
                            break
                    alpha *= 0.5

                if not accepted:
                    # Use best point found during line search
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        g = compute_gradient(x)
                    else:
                        # Try steepest descent
                        direction = -g
                        alpha = 1.0
                        for ls_iter in range(10):
                            if evals >= budget:
                                break
                            x_trial = x + alpha * direction
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = func(x_trial)
                            evals += 1
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = x_trial.copy()
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                g = compute_gradient(x)
                                break
                            alpha *= 0.5

                if np.linalg.norm(g) < 1e-8:
                    break

                # Update L-BFGS history
                s = alpha * direction
                y = g - g  # placeholder, will compute after accepting
                # Recompute properly
                g_new = compute_gradient(x)
                y = g_new - g
                sy = np.dot(s, y)
                if sy > 1e-12:
                    s_list.append(s)
                    y_list.append(y)
                    rho_list.append(1.0 / sy)
                    if len(s_list) > m:
                        s_list.pop(0)
                        y_list.pop(0)
                        rho_list.pop(0)

                g = g_new
                # Update gamma
                if np.dot(g, g) > 1e-12:
                    gamma = np.dot(s, y) / np.dot(y, y)

            return x, f
>>>>>>> REPLACE

