"""Derivative-Free Trust-Region optimizer using quadratic model from 2n+1 points."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """DFTR optimizer: local quadratic model from sparse points, direct inversion for ill-conditioning."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        n = self.dim
        call_count = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def update_best(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def build_quadratic_model(x0, f0, delta):
            """
            Build a quadratic model using 2n+1 points (center + axis points).
            Returns (b, H, f0) where model(x) ≈ f0 + b^T(x-x0) + 0.5*(x-x0)^T H (x-x0)
            """
            nonlocal call_count
            # Evaluate center
            fc = make_call(x0)
            if fc is None:
                return None
            if not np.isfinite(fc):
                return None

            # Evaluate axis points: x0 ± delta*e_i
            axis_points = []
            axis_vals = []
            for i in range(n):
                for sign in [1, -1]:
                    xi = x0.copy()
                    xi[i] += sign * delta
                    xi = np.clip(xi, -5, 5)
                    fi = make_call(xi)
                    if fi is None:
                        return None
                    if not np.isfinite(fi):
                        return None
                    axis_points.append(xi)
                    axis_vals.append(fi)

            # Compute gradient (b) and Hessian (H) from axis values
            # For axis i: f(x0 + delta*e_i) = f0 + b_i*delta + 0.5*H_ii*delta^2
            #             f(x0 - delta*e_i) = f0 - b_i*delta + 0.5*H_ii*delta^2
            # Adding: 2*f0 = f(x0+de_i) + f(x0-de_i) - 0.5*H_ii*delta^2... wait
            # Actually: f(x0+de_i) = f0 + b_i*delta + 0.5*H_ii*delta^2
            #           f(x0-de_i) = f0 - b_i*delta + 0.5*H_ii*delta^2
            # Sum: f(x0+de_i) + f(x0-de_i) = 2*f0 + H_ii*delta^2
            # Diff: f(x0+de_i) - f(x0-de_i) = 2*b_i*delta

            b = np.zeros(n)
            H = np.zeros((n, n))

            for i in range(n):
                f_plus = axis_vals[2*i]
                f_minus = axis_vals[2*i + 1]
                b[i] = (f_plus - f_minus) / (2.0 * delta)
                H[i, i] = (f_plus + f_minus - 2.0 * fc) / (delta ** 2)

            return b, H, fc

        def solve_trust_region(x0, f0, b, H, delta):
            """
            Find the step that minimizes the quadratic model within the trust region ||step|| <= delta.
            Solves (H + 2*mu*I) s = -b for various mu to satisfy KKT conditions.
            """
            # Start with small mu
            mu = 1e-6
            for _ in range(50):
                A = H + 2.0 * mu * np.eye(n)
                try:
                    s = np.linalg.solve(A, -b)
                except np.linalg.LinAlgError:
                    mu *= 10
                    continue

                s_norm = np.linalg.norm(s)
                if s_norm < delta * (1 - 1e-6):
                    # Interior solution
                    return s, mu
                elif s_norm > delta * (1 + 1e-6):
                    # Boundary: project s to sphere
                    s = s * (delta / s_norm)
                    # Check if this is the right boundary solution
                    # For boundary, we need (H + 2*mu*I)s = -lambda*s for some lambda
                    # Simplified: just use the projected step
                    return s, mu
                else:
                    # Very close to boundary
                    return s, mu

                # Adjust mu
                if s_norm > delta:
                    mu *= 10
                else:
                    mu *= 0.1

            # Fallback: steepest descent step
            s = -b / (np.linalg.norm(b) + 1e-12) * delta
            return s, mu

        def dftr_search(x0, delta0, max_calls):
            """
            Run DFTR from x0 with initial trust region delta0.
            Returns (best_x_local, best_f_local, calls_made)
            """
            nonlocal best_x, best_f, call_count

            x = x0.copy()
            delta = delta0
            local_best_x = x.copy()
            local_best_f = np.inf
            calls_in_run = 0
            stagnation = 0

            # Initial evaluation
            fx = make_call(x)
            if fx is None:
                return local_best_x, local_best_f, 1
            calls_in_run += 1
            if np.isfinite(fx):
                local_best_f = fx
                local_best_x = x.copy()
                update_best(x, fx)

            while calls_in_run < max_calls:
                if call_count >= self.budget:
                    break
                if time.time() > deadline - 3.0:
                    break

                # Build quadratic model
                model = build_quadratic_model(x, local_best_f, delta)
                if model is None:
                    break
                b, H, fc = model
                calls_in_run += (2 * n + 1)

                # Ensure we haven't exceeded budget
                if call_count >= self.budget:
                    break

                # Solve trust region subproblem
                s, mu = solve_trust_region(x, fc, b, H, delta)
                x_trial = x + s
                x_trial = np.clip(x_trial, -5, 5)

                # Evaluate trial point
                f_trial = make_call(x_trial)
                if f_trial is None:
                    break
                calls_in_run += 1

                if not np.isfinite(f_trial):
                    # Bad point, shrink trust region
                    delta *= 0.5
                    continue

                # Compute actual reduction vs predicted reduction
                predicted_reduction = fc - (fc + b @ s + 0.5 * (s @ H @ s))
                actual_reduction = fc - f_trial

                if actual_reduction > 0.1 * abs(predicted_reduction) and actual_reduction > 1e-10:
                    # Good step: accept and expand trust region
                    x = x_trial
                    local_best_f = f_trial
                    local_best_x = x.copy()
                    update_best(x, f_trial)
                    delta = min(delta * 1.5, 10.0)
                    stagnation = 0
                else:
                    # Bad step: shrink trust region
                    delta *= 0.5
                    stagnation += 1

                if delta < 1e-6:
                    break

                if stagnation > 20:
                    break

            return local_best_x, local_best_f, calls_in_run

        # Strategy:
        # 1. Start from center
        # 2. Run DFTR with moderate trust region
        # 3. If stuck, restart from best with new perturbation

        x_start = np.zeros(n)
        fx_start = make_call(x_start)
        if fx_start is not None and np.isfinite(fx_start):
            update_best(x_start, fx_start)

        # First DFTR run
        delta_init = 2.0
        max_calls_first = 300
        x_best, f_best, calls_made = dftr_search(x_start, delta_init, max_calls_first)

        # Additional restarts if budget remains
        restart_count = 0
        max_restarts = 2

        while call_count < self.budget - 20 and restart_count < max_restarts:
            if time.time() > deadline - 10.0:
                break

            if best_x is not None:
                perturbation = rng.standard_normal(n) * 0.3
                x_new_start = best_x + perturbation
                x_new_start = np.clip(x_new_start, -5, 5)
            else:
                x_new_start = rng.uniform(-3, 3, n)

            delta_restart = 1.0 if restart_count == 0 else 0.5
            max_calls_restart = min(150, self.budget - call_count - 20)

            if max_calls_restart <= 10:
                break

            x_best_r, f_best_r, calls_made_r = dftr_search(x_new_start, delta_restart, max_calls_restart)
            restart_count += 1

        # Final fallback
        if best_x is None:
            x = np.zeros(n)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
