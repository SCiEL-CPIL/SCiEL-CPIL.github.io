"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free stochastic quasi-Newton (L-BFGS with FD gradients)
    with multi-start for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        call_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        start_time = time.time()
        deadline = start_time + 285.0

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return 1e10
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        def finite_diff_gradient(x, h=1e-6):
            """Compute gradient via central differences.
            Uses 2*dim function evaluations."""
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                fp = wrapped_func(x_plus)
                fm = wrapped_func(x_minus)
                grad[i] = (fp - fm) / (2 * h)
            return grad

        def lbfgs_step(x, grad, history, m=5):
            """One L-BFGS step.
            history: list of (s_i, y_i) pairs, most recent first."""
            if len(history) == 0:
                # Use steepest descent with line search
                direction = -grad
            else:
                # Two-loop recursion for L-BFGS
                s_list = [h[0] for h in history]
                y_list = [h[1] for h in history]
                n_hist = len(s_list)
                
                rho = [1.0 / np.dot(s_list[i], y_list[i]) for i in range(n_hist)
                       if np.dot(s_list[i], y_list[i]) > 1e-12]
                # Filter out invalid pairs
                valid_idx = [i for i in range(n_hist) if np.dot(s_list[i], y_list[i]) > 1e-12]
                if not valid_idx:
                    direction = -grad
                else:
                    s_valid = [s_list[i] for i in valid_idx]
                    y_valid = [y_list[i] for i in valid_idx]
                    rho_valid = [1.0 / np.dot(s, y) for s, y in zip(s_valid, y_valid)]
                    n_v = len(s_valid)
                    
                    q = grad.copy()
                    alpha = np.zeros(n_v)
                    for i in range(n_v - 1, -1, -1):
                        alpha[i] = rho_valid[i] * np.dot(s_valid[i], q)
                        q -= alpha[i] * y_valid[i]
                    
                    # Initial Hessian approximation
                    if n_v > 0:
                        gamma = np.dot(s_valid[-1], y_valid[-1]) / np.dot(y_valid[-1], y_valid[-1])
                        if gamma <= 0:
                            gamma = 1.0
                        r = gamma * q
                    else:
                        r = q
                    
                    for i in range(n_v):
                        beta = rho_valid[i] * np.dot(y_valid[i], r)
                        r += (alpha[i] - beta) * s_valid[i]
                    
                    direction = -r

            # Scale direction
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-15:
                return x, None, None
            direction /= dir_norm

            # Backtracking line search
            f_x = wrapped_func(x)
            step_size = 1.0
            for _ in range(20):
                x_new = x + step_size * direction
                x_new_clipped = np.clip(x_new, -5.0, 5.0)
                f_new = wrapped_func(x_new_clipped)
                if f_new < f_x - 1e-4 * step_size * np.dot(-direction, grad):
                    s = x_new_clipped - x
                    # Compute new gradient
                    grad_new = finite_diff_gradient(x_new_clipped)
                    y = grad_new - grad
                    # Update history
                    if np.dot(s, y) > 1e-12:
                        history.append((s, y))
                        if len(history) > 5:
                            history.pop(0)
                    return x_new_clipped, s, y
                step_size *= 0.5

            # If line search failed, just take a small step
            x_new = x + 0.01 * direction
            x_new_clipped = np.clip(x_new, -5.0, 5.0)
            f_new = wrapped_func(x_new_clipped)
            s = x_new_clipped - x
            grad_new = finite_diff_gradient(x_new_clipped)
            y = grad_new - grad
            if np.dot(s, y) > 1e-12:
                history.append((s, y))
                if len(history) > 5:
                    history.pop(0)
            return x_new_clipped, s, y

        def run_lbfgs(x0, max_calls):
            """Run L-BFGS from x0 with given call budget."""
            nonlocal call_count, best_x, best_f
            history = []
            x = x0.copy()
            f_x = wrapped_func(x)
            
            while call_count < budget and time.time() < deadline:
                if call_count >= budget:
                    break
                # Need 2*dim + 1 calls for gradient + line search
                if budget - call_count < 2 * dim + 2:
                    break
                
                grad = finite_diff_gradient(x)
                if np.linalg.norm(grad) < 1e-10:
                    break
                
                x_new, s, y = lbfgs_step(x, grad, history)
                if x_new is None:
                    break
                x = x_new
                if time.time() > deadline:
                    break

        # Generate diverse starting points
        starts = [np.zeros(dim)]
        starts.append(rng.normal(0, 1.0, dim))
        starts.append(rng.normal(0, 2.0, dim))
        for _ in range(4):
            starts.append(rng.uniform(-4.0, 4.0, dim))

        # Sort by norm
        starts.sort(key=lambda x: np.linalg.norm(x))

        # Allocate budget across starts
        n_starts = len(starts)
        calls_per_start = budget // n_starts
        calls_per_start = max(30, calls_per_start)

        for x0 in starts:
            if time.time() > deadline or call_count >= budget:
                break
            if budget - call_count < 2 * dim + 2:
                break
            run_lbfgs(x0, calls_per_start)

        # Final polish with remaining budget
        if call_count < budget and time.time() < deadline and budget - call_count >= 2 * dim + 2:
            run_lbfgs(best_x, budget - call_count)

        return best_x, best_f
# EVOLVE-BLOCK-END
