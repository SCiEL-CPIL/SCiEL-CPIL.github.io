"""Derivative-free BFGS with multi-start for BBOB f11 (Discus)."""
import numpy as np
import time


class Optimizer:
    """Multi-start derivative-free BFGS local optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _finite_diff_grad(self, func, x, h=1e-5):
        """Compute gradient via central finite differences."""
        g = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            x_plus = np.clip(x_plus, -5, 5)
            x_minus = np.clip(x_minus, -5, 5)
            g[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return g

    def _bfgs_step(self, func, x, fx, H, max_iter, remaining_budget):
        """
        Perform BFGS optimization from x with inverse Hessian approximation H.
        Returns best x, fx, and updated H.
        """
        dim = self.dim
        best_x = x.copy()
        best_fx = fx
        calls_used = 0

        for _ in range(max_iter):
            if calls_used >= remaining_budget:
                break

            # Compute gradient
            g = self._finite_diff_grad(func, x, h=1e-5)
            calls_used += dim  # central diff uses 2*dim evals, but we count conservatively

            if np.linalg.norm(g) < 1e-10:
                break

            # Search direction: p = -H @ g
            p = -H @ g

            # Line search (backtracking)
            alpha = 1.0
            c1 = 1e-4
            for _ls in range(30):
                y_new = np.clip(x + alpha * p, -5, 5)
                fy = func(y_new)
                calls_used += 1
                if fy < best_fx:
                    best_x = y_new.copy()
                    best_fx = fy
                # Armijo condition
                if fy <= fx + c1 * alpha * (g @ p):
                    break
                alpha *= 0.5
            else:
                # If line search fails, take a small step in negative gradient direction
                alpha = 1e-3
                y_new = np.clip(x - alpha * g, -5, 5)
                fy = func(y_new)
                calls_used += 1

            # Update H via BFGS
            s = y_new - x
            y_vec = self._finite_diff_grad(func, y_new, h=1e-5) - g
            calls_used += dim

            sy = s @ y_vec
            if sy > 1e-12:
                # BFGS update for inverse Hessian
                rho = 1.0 / sy
                I = np.eye(dim)
                H = (I - rho * np.outer(s, y_vec)) @ H @ (I - rho * np.outer(y_vec, s)) + rho * np.outer(s, s)
                # Symmetrize
                H = 0.5 * (H + H.T)

            x = y_new
            fx = fy

            if calls_used >= remaining_budget:
                break

        return best_x, best_fx, H, calls_used

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s

        # Multi-start: use several diverse initial points
        rng = np.random.default_rng(42)

        # Generate initial points: center, corners, and random
        init_points = [np.zeros(dim)]
        # Add some structured points
        for corner in range(2 ** dim):
            pt = np.array([(-5 if (corner >> i) & 1 == 0 else 5) for i in range(dim)])
            init_points.append(pt * 0.8)  # slightly inside
        # Random points
        for _ in range(10):
            init_points.append(rng.uniform(-4, 4, dim))

        # Shuffle to diversify
        rng.shuffle(init_points)

        best_x = init_points[0].copy()
        best_fx = func(init_points[0])
        calls_used = 1

        # Allocate budget across starts
        n_starts = min(len(init_points), 8)
        budget_per_start = max(50, (budget - 1) // n_starts)

        for start_idx in range(n_starts):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            x0 = init_points[start_idx].copy()
            fx0 = func(x0)
            calls_used += 1

            if fx0 < best_fx:
                best_x = x0.copy()
                best_fx = fx0

            # Initialize inverse Hessian as identity
            H = np.eye(dim)

            remaining = min(budget_per_start, budget - calls_used)
            if remaining < 10:
                continue

            # Run BFGS
            x_new, fx_new, H_new, calls_this = self._bfgs_step(
                func, x0, fx0, H, max_iter=50, remaining_budget=remaining
            )
            calls_used += calls_this

            if fx_new < best_fx:
                best_x = x_new.copy()
                best_fx = fx_new

        return best_x, best_fx
