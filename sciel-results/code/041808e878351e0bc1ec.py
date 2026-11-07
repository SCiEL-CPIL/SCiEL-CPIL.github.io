"""Optimizer for BBOB f4 (Büche-Rastrigin) with Rastrigin-aware coordinate jumps and GP-guided global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Rastrigin-aware optimizer: coordinate ridge-jumps + local polish + GP-guided global restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rastrigin_jumps(self, x, func, budget_used, deadline, rng):
        """Try to escape local minima by jumping across Rastrigin ridges along coordinates."""
        best_x = x.copy()
        best_f = func(best_x)
        budget_used[0] += 1

        # For each coordinate, try shifting by ±2*pi/k for small k to cross ridges
        for coord in range(self.dim):
            if budget_used[0] >= self.budget or time.time() > deadline:
                break
            for k in [1, 2, 3, 4]:
                for sign in [1, -1]:
                    if budget_used[0] >= self.budget or time.time() > deadline:
                        break
                    shift = sign * 2 * np.pi / k
                    y = best_x.copy()
                    y[coord] = np.clip(best_x[coord] + shift, -5, 5)
                    fy = func(y)
                    budget_used[0] += 1
                    if fy < best_f:
                        best_x = y
                        best_f = fy
        return best_x, best_f

    def _local_polish(self, x, func, budget_used, deadline, rng):
        """Fine-tune using Nelder-Mead or simple gradient-free local search."""
        best_x = x.copy()
        best_f = func(best_x)
        budget_used[0] += 1

        # Simple coordinate descent with adaptive step
        step = 0.5
        for _ in range(min(50, self.budget - budget_used[0])):
            if time.time() > deadline:
                break
            improved = False
            for coord in range(self.dim):
                if budget_used[0] >= self.budget or time.time() > deadline:
                    break
                for sign in [1, -1]:
                    if budget_used[0] >= self.budget or time.time() > deadline:
                        break
                    y = best_x.copy()
                    y[coord] = np.clip(best_x[coord] + sign * step, -5, 5)
                    fy = func(y)
                    budget_used[0] += 1
                    if fy < best_f:
                        best_x = y
                        best_f = fy
                        improved = True
            if not improved:
                step *= 0.7
            else:
                step *= 1.1
        return best_x, best_f

    def _gp_suggest(self, X, Y, rng):
        """Suggest next point using a simple GP with squared exponential kernel and UCB acquisition."""
        n = len(X)
        if n < 2:
            return rng.uniform(-5, 5, self.dim)

        # Simple GP implementation
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-4

        # Compute kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sq_dist = np.sum((X[i] - X[j]) ** 2)
                K[i, j] = signal_var * np.exp(-0.5 * sq_dist / length_scale ** 2)
        K[np.diag_indices(n)] += noise_var

        # Center the data
        y_mean = np.mean(Y)
        Yc = Y - y_mean

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fallback to random point
            return rng.uniform(-5, 5, self.dim)

        alpha = np.linalg.solve(np.linalg.cholesky(K).T, np.linalg.solve(L, Yc))

        # Evaluate acquisition function on a grid of candidates
        n_candidates = 200
        candidates = rng.uniform(-5, 5, (n_candidates, self.dim))

        best_acq = -np.inf
        best_candidate = candidates[0]

        for c in candidates:
            # Compute kernel between candidate and training points
            k_star = np.zeros(n)
            for i in range(n):
                sq_dist = np.sum((c - X[i]) ** 2)
                k_star[i] = signal_var * np.exp(-0.5 * sq_dist / length_scale ** 2)

            # Posterior mean and variance
            mean = y_mean + np.dot(k_star, alpha)
            v = np.linalg.solve(L, k_star)
            var = signal_var - np.dot(k_star, v)
            std = np.sqrt(max(var, 0))

            # UCB acquisition
            beta = 2.0
            acq = mean + beta * std

            if acq > best_acq:
                best_acq = acq
                best_candidate = c

        return best_candidate

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 280  # Safety margin before 300s limit

        budget_used = [0]
        best_x = None
        best_f = np.inf

        # Store all evaluations for GP
        X_list = []
        Y_list = []

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            budget_used[0] += 1
            X_list.append(x.copy())
            Y_list.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to get a landscape overview
        n_initial = min(20, self.budget)
        for _ in range(n_initial):
            if budget_used[0] >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            evaluate(x)

        # Phase 2: Iterative loop: local polish -> ridge jumps -> GP-guided restart
        while budget_used[0] < self.budget and time.time() < deadline:
            # Get current best
            if best_x is None:
                x = rng.uniform(-5, 5, self.dim)
                evaluate(x)
                continue

            # Local polish around best
            x_polished, f_polished = self._local_polish(
                best_x, func, budget_used, deadline, rng
            )
            if f_polished < best_f:
                best_f = f_polished
                best_x = x_polished

            # Try ridge jumps
            x_jumped, f_jumped = self._rastrigin_jumps(
                best_x, func, budget_used, deadline, rng
            )
            if f_jumped < best_f:
                best_f = f_jumped
                best_x = x_jumped

            # GP-guided global suggestion
            if len(X_list) >= 5:
                X = np.array(X_list)
                Y = np.array(Y_list)
                x_gp = self._gp_suggest(X, Y, rng)
                if budget_used[0] < self.budget and time.time() < deadline:
                    f_gp = evaluate(x_gp)
                    if f_gp < best_f:
                        best_f = f_gp
                        best_x = x_gp

                    # Polish the GP suggestion
                    x_gp_polished, f_gp_polished = self._local_polish(
                        x_gp, func, budget_used, deadline, rng
                    )
                    if f_gp_polished < best_f:
                        best_f = f_gp_polished
                        best_x = x_gp_polished

                    # Jump from GP suggestion
                    x_gp_jumped, f_gp_jumped = self._rastrigin_jumps(
                        x_gp, func, budget_used, deadline, rng
                    )
                    if f_gp_jumped < best_f:
                        best_f = f_gp_jumped
                        best_x = x_gp_jumped

            # If no improvement in last several iterations, do a fresh random restart
            if budget_used[0] < self.budget and time.time() < deadline:
                if len(X_list) > 0 and (budget_used[0] % 30 == 0):
                    x_restart = rng.uniform(-5, 5, self.dim)
                    f_restart = evaluate(x_restart)
                    if f_restart < best_f:
                        best_f = f_restart
                        best_x = x_restart

        return best_x, best_f
# EVOLVE-BLOCK-END
