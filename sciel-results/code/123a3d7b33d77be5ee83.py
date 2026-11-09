"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic BFGS with finite-difference gradients and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # We'll track the best point found overall
        best_x = None
        best_f = np.inf
        calls_used = 0

        def count_call():
            nonlocal calls_used
            calls_used += 1

        def safe_func(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            count_call()
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Try multiple starting points with BFGS-like local optimization
        # Use a deterministic set of starting points
        rng = np.random.default_rng(123)

        # Starting points: center, corners, and random
        starts = [np.zeros(dim)]
        # Add some structured points
        for _ in range(3):
            starts.append(rng.uniform(-4, 4, dim))

        # For each start, run a local BFGS-like optimization
        for x0 in starts:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize BFGS state
            x = np.clip(x0, -5, 5).copy()
            fx = safe_func(x)
            if calls_used >= budget:
                break

            # Finite difference gradient
            h = 1e-4
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xp[i] += h
                xp = np.clip(xp, -5, 5)
                xp = np.clip(xp, -5, 5)
                fp = safe_func(xp)
                if calls_used >= budget:
                    break
                xm = x.copy()
                xm[i] -= h
                xm = np.clip(xm, -5, 5)
                fm = safe_func(xm)
                if calls_used >= budget:
                    break
                grad[i] = (fp - fm) / (2 * h)

            if calls_used >= budget:
                break

            # BFGS Hessian inverse approximation
            H = np.eye(dim)
            max_local_iters = 30

            for it in range(max_local_iters):
                if calls_used >= budget - 2 * dim:
                    break
                if time.time() > deadline:
                    break

                # Check gradient norm
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-6:
                    break

                # Step direction
                d = -H @ grad
                # Line search: backtracking
                alpha = 1.0
                x_new = np.clip(x + alpha * d, -5, 5)
                fx_new = safe_func(x_new)
                if calls_used >= budget:
                    break

                # Simple backtracking line search
                c1 = 1e-4
                for _ in range(20):
                    if fx_new <= fx + c1 * alpha * np.dot(grad, d):
                        break
                    alpha *= 0.5
                    x_new = np.clip(x + alpha * d, -5, 5)
                    fx_new = safe_func(x_new)
                    if calls_used >= budget:
                        break

                if calls_used >= budget:
                    break

                # Update BFGS
                s = x_new - x
                g_new = np.zeros(dim)
                for i in range(dim):
                    xp = x_new.copy()
                    xp[i] += h
                    xp = np.clip(xp, -5, 5)
                    fp = safe_func(xp)
                    if calls_used >= budget:
                        break
                    xm = x_new.copy()
                    xm[i] -= h
                    xm = np.clip(xm, -5, 5)
                    fm = safe_func(xm)
                    if calls_used >= budget:
                        break
                    g_new[i] = (fp - fm) / (2 * h)

                if calls_used >= budget:
                    break

                y = g_new - grad
                sy = np.dot(s, y)
                if sy > 1e-10:
                    Hy = H @ y
                    H = H + (s @ s.T) / sy - (Hy @ Hy.T) / sy

                x = x_new
                fx = fx_new
                grad = g_new

            # After local opt, restart if budget remains
            if calls_used >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
