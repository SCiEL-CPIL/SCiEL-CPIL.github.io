"""Covariance-free adaptive search with rotation-based anisotropy for Bent Cigar (f12)."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Covariance-free adaptive search with rotation-based anisotropy.

    Instead of maintaining a full covariance matrix (CMA-ES), we maintain a
    random rotation matrix that is iteratively updated via a simple
    correlation-based heuristic. This emulates anisotropic adaptation
    without the computational overhead or instability of full CMA-ES.
    Multi-start Powell's method provides local refinement on the elongated
    valleys characteristic of Bent Cigar.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 275.0
        n = self.dim
        budget = self.budget

        # Global best tracking
        best_x = np.zeros(n)
        best_fx = func(best_x)
        calls_used = 1

        def adaptive_rotation_search(x0, sigma0, max_calls, deadline):
            """Covariance-free adaptive search using rotation matrix updates."""
            nonlocal best_x, best_fx, calls_used

            x_mean = x0.copy()
            sigma = sigma0

            # Initialize rotation matrix
            R = np.eye(n)

            # Track recent successful steps for correlation-based adaptation
            successful_steps = []
            step_history = []

            no_improve_count = 0
            stagnation_threshold = 10
            prev_best = best_fx

            for gen in range(1, 5000):
                if time.time() > deadline:
                    break
                if calls_used >= max_calls:
                    break

                # Population size
                lam = 10

                # Check if we have enough budget
                if calls_used + lam > max_calls:
                    rem = max_calls - calls_used
                    for _ in range(rem):
                        if time.time() > deadline:
                            break
                        z = rng.standard_normal(n)
                        y = x_mean + sigma * (R @ z)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < best_fx:
                            best_fx = fy
                            best_x = y.copy()
                    break

                # Sample population
                ys = np.zeros((lam, n))
                fs = np.zeros(lam)
                for i in range(lam):
                    z = rng.standard_normal(n)
                    ys[i] = x_mean + sigma * (R @ z)
                    ys[i] = np.clip(ys[i], -5, 5)
                    fs[i] = func(ys[i])
                    calls_used += 1
                    if fs[i] < best_fx:
                        best_fx = fs[i]
                        best_x = ys[i].copy()

                # Check improvement
                if best_fx < prev_best - 1e-10:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                prev_best = best_fx

                # Sort by fitness
                idx = np.argsort(fs)
                ys_sorted = ys[idx]
                fs_sorted = fs[idx]

                # Update mean using top mu individuals
                mu = 5
                w = np.log(np.arange(mu, 0, -1))
                w /= w.sum()
                x_mean_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)

                # Compute step direction in rotated space
                step = (x_mean_new - x_mean) / sigma
                step_history.append(step.copy())

                # Keep last 20 steps for correlation computation
                if len(step_history) > 20:
                    step_history.pop(0)

                # Correlation-based rotation update
                # Find directions of recent successful steps
                if len(step_history) >= 5:
                    recent_steps = np.array(step_history[-10:])
                    # Compute average direction
                    avg_step = np.mean(recent_steps, axis=0)
                    if np.linalg.norm(avg_step) > 1e-6:
                        avg_step = avg_step / np.linalg.norm(avg_step)
                        # Update rotation to align first axis with avg_step
                        # Use Gram-Schmidt-like update
                        R_new = R.copy()
                        # Project avg_step onto current axes
                        projections = R_new.T @ avg_step
                        # Update: rotate towards avg_step
                        for i in range(n):
                            for j in range(n):
                                R_new[i, j] += 0.05 * (avg_step[i] * projections[j] - R_new[i, j] * avg_step[j])
                        # Re-orthogonalize
                        Q, _ = np.linalg.qr(R_new)
                        R = Q

                # Update sigma based on progress
                progress = fs_sorted[0] - fs_sorted[-1]
                if progress > 0:
                    sigma *= 0.95
                else:
                    sigma *= 1.05
                sigma = np.clip(sigma, 1e-4, 50.0)

                x_mean = x_mean_new

                # Stagnation handling: restart from best with larger sigma
                if no_improve_count >= stagnation_threshold:
                    x_mean = best_x.copy()
                    sigma = max(sigma * 2.0, 0.5)
                    sigma = np.clip(sigma, 0.1, 20.0)
                    R = np.eye(n)
                    step_history = []
                    no_improve_count = 0

            return best_x, best_fx, calls_used

        def powell_polish(x0, max_calls, deadline):
            """Run Powell's method polish from x0, using at most max_calls evaluations."""
            nonlocal best_x, best_fx, calls_used

            if max_calls < 10:
                return

            func_calls = [0]

            def powell_func(x):
                func_calls[0] += 1
                if func_calls[0] > max_calls:
                    raise RuntimeError("Budget exceeded")
                x_clipped = np.clip(x, -5, 5)
                return func(x_clipped)

            try:
                result = minimize(powell_func, x0, method='Powell',
                                  options={'maxiter': max_calls, 'maxfev': max_calls,
                                           'xtol': 1e-8, 'ftol': 1e-10})
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            calls_used += func_calls[0]

        # Phase 1: Adaptive rotation search with main budget
        search_budget = int(budget * 0.6)
        adaptive_rotation_search(np.zeros(n), 2.0, search_budget, deadline)

        # Phase 2: Multi-start Powell's method for local refinement
        remaining = budget - calls_used
        if remaining > 20 and time.time() < deadline - 10:
            # Try Powell from best point
            powell_polish(best_x.copy(), min(remaining, 150), deadline)

            # If still have budget, try a few more starting points
            remaining = budget - calls_used
            if remaining > 30 and time.time() < deadline - 5:
                # Generate a few diverse starting points near best
                for i in range(3):
                    if calls_used >= budget or time.time() > deadline - 5:
                        break
                    offset = rng.standard_normal(n) * 0.5
                    x_start = np.clip(best_x + offset, -5, 5)
                    calls_before = calls_used
                    powell_polish(x_start, min(remaining - (calls_used - calls_before), 50), deadline)
                    remaining = budget - calls_used

        return best_x, best_fx
