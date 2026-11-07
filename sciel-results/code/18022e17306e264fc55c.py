"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free Trust-Region (DFTR) optimizer for BBOB f18 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        n = self.dim
        total_calls = 0
        best_x = None
        best_f = np.inf

        def timed_func(x):
            nonlocal total_calls
            if total_calls >= self.budget:
                return np.inf
            fx = func(x)
            total_calls += 1
            return fx

        def time_ok():
            return time.time() < deadline

        # DFTR: Derivative-Free Trust-Region method
        def dftr(x0, delta0, budget_dftr, func_wrapper):
            """
            Derivative-Free Trust-Region method using quadratic surrogates.
            Constructs local quadratic model from directional derivatives.
            """
            n = self.dim
            x = np.array(x0, dtype=float)
            delta = delta0
            fx = func_wrapper(x)
            calls = 1
            best_x_local = x.copy()
            best_f_local = fx

            # Maximum iterations
            max_iter = budget_dftr // (2 * n + 1)  # Each iter needs ~2n+1 evals
            max_iter = min(max_iter, 100)

            # Quadratic model coefficients: f(x) ≈ f(x0) + g^T (x-x0) + (x-x0)^T H (x-x0)
            # We build the model incrementally

            for iteration in range(max_iter):
                if not time_ok() or calls >= budget_dftr:
                    break

                # Build quadratic model around x
                # Evaluate function at x + delta * e_i and x - delta * e_i for each dimension
                # This gives us directional derivative approximations
                g = np.zeros(n)  # Gradient approximation
                H = np.zeros((n, n))  # Hessian approximation (diagonal initially)

                # Central differences for gradient
                for i in range(n):
                    if calls >= budget_dftr:
                        break
                    x_plus = x.copy()
                    x_plus[i] += delta
                    fp = func_wrapper(x_plus)
                    calls += 1

                    x_minus = x.copy()
                    x_minus[i] -= delta
                    fm = func_wrapper(x_minus)
                    calls += 1

                    if np.isfinite(fp) and np.isfinite(fm):
                        g[i] = (fp - fm) / (2 * delta)

                        # Second derivative approximation
                        fpp = (fp - 2 * fx + fm) / (delta ** 2)
                        H[i, i] = fpp

                if calls >= budget_dftr:
                    break

                # Add off-diagonal Hessian terms using mixed differences
                # For ill-conditioned problems, we need cross-terms
                for i in range(n):
                    if calls >= budget_dftr:
                        break
                    for j in range(i + 1, n):
                        if calls >= budget_dftr:
                            break
                        x_pp = x.copy()
                        x_pp[i] += delta
                        x_pp[j] += delta
                        fpp = func_wrapper(x_pp)
                        calls += 1

                        x_pm = x.copy()
                        x_pm[i] += delta
                        x_pm[j] -= delta
                        fpm = func_wrapper(x_pm)
                        calls += 1

                        x_mp = x.copy()
                        x_mp[i] -= delta
                        x_mp[j] += delta
                        fmp = func_wrapper(x_mp)
                        calls += 1

                        x_mm = x.copy()
                        x_mm[i] -= delta
                        x_mm[j] -= delta
                        fmm = func_wrapper(x_mm)
                        calls += 1

                        # Mixed partial: (fpp - fpm - fmp + fmm) / (4*delta^2)
                        h_ij = (fpp - fpm - fmp + fmm) / (4 * delta ** 2)
                        H[i, j] = h_ij
                        H[j, i] = h_ij

                if calls >= budget_dftr:
                    break

                # Solve subproblem: minimize model within trust region
                # min g^T s + 0.5 s^T H s  s.t. ||s|| <= delta
                # Use projected gradient or direct solve for small n

                # For n=5, we can try to solve the KKT conditions directly
                # or use a simple approach: try gradient step with line search

                # Regularize Hessian to be positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(H)
                eigvals = np.maximum(eigvals, 1e-8)
                H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Solve (H_reg + lambda*I) s = -g for the trust region subproblem
                # Start with lambda = 0 and increase if ||s|| > delta
                s = np.zeros(n)
                found = False
                for lam in [0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
                    A = H_reg + lam * np.eye(n)
                    try:
                        s_candidate = np.linalg.solve(A, -g)
                    except np.linalg.LinAlgError:
                        s_candidate = -g / (np.linalg.norm(g) + 1e-10) * delta

                    if np.linalg.norm(s_candidate) <= delta * (1 + 1e-6):
                        s = s_candidate
                        found = True
                        break

                if not found:
                    # Scale s to trust region boundary
                    norm_s = np.linalg.norm(s)
                    if norm_s > delta:
                        s = s * (delta / norm_s)
                    else:
                        s = s

                # Evaluate new point
                x_new = x + s
                fx_new = func_wrapper(x_new)
                calls += 1

                if not np.isfinite(fx_new):
                    # Shrink trust region
                    delta *= 0.5
                    continue

                # Check improvement
                predicted_reduction = -g @ s - 0.5 * s @ H_reg @ s
                actual_reduction = fx - fx_new

                if actual_reduction > 0.25 * predicted_reduction and predicted_reduction > 0:
                    # Good step: accept and possibly expand
                    x = x_new
                    fx = fx_new
                    if fx < best_f_local:
                        best_f_local = fx
                        best_x_local = x.copy()

                    # Expand trust region if step was very good
                    if actual_reduction > 0.75 * predicted_reduction:
                        delta = min(delta * 1.5, 5.0)
                    else:
                        delta = min(delta * 1.2, 5.0)
                else:
                    # Reject step: shrink trust region
                    delta *= 0.5
                    if delta < 1e-10:
                        break

            return best_x_local, best_f_local, calls

        # Multi-start DFTR
        rng = np.random.default_rng(42)

        # Phase 1: Coarse search with multiple starts
        n_starts = 5
        budget_per_start = self.budget // n_starts

        for start_idx in range(n_starts):
            if not time_ok() or total_calls >= self.budget:
                break

            if start_idx == 0:
                x0 = np.zeros(n)
            elif start_idx == 1:
                x0 = rng.uniform(-2, 2, n)
            elif start_idx == 2:
                x0 = rng.uniform(-3, 3, n)
            elif start_idx == 3:
                x0 = rng.uniform(-4, 4, n)
            else:
                x0 = rng.uniform(-4.5, 4.5, n)

            # Clip to box
            x0 = np.clip(x0, -5, 5)

            x_dftr, f_dftr, calls_dftr = dftr(x0, 2.0, budget_per_start, timed_func)

            if f_dftr < best_f:
                best_f = f_dftr
                best_x = x_dftr.copy()

        # Phase 2: Fine-tune from best point with smaller trust region
        if best_x is not None and total_calls < self.budget and time_ok():
            remaining = self.budget - total_calls
            if remaining > 2 * n:
                x_fine, f_fine, calls_fine = dftr(best_x, 0.5, remaining, timed_func)
                if f_fine < best_f:
                    best_f = f_fine
                    best_x = x_fine.copy()

        # Phase 3: Very fine local search
        if best_x is not None and total_calls < self.budget and time_ok():
            remaining = self.budget - total_calls
            if remaining > 2 * n:
                x_very_fine, f_very_fine, calls_vf = dftr(best_x, 0.1, remaining, timed_func)
                if f_very_fine < best_f:
                    best_f = f_very_fine
                    best_x = x_very_fine.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
