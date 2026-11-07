"""Rastrigin-aware coordinate local search with ridge-jumping for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware coordinate local search with ridge-jumping for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(12345)

        dim = self.dim
        calls = 0
        x_best = None
        f_best = np.inf

        def safe_func(x):
            nonlocal calls, x_best, f_best
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64).ravel()[:dim]
            x = np.clip(x, -5, 5)
            f = float(func(x))
            calls += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
            return f

        # --- Phase 1: Latin Hypercube Sampling for initial exploration ---
        n_lhs = min(100, self.budget // 5)
        X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            X[:, d] = (perm + rng.random(n_lhs)) / n_lhs * 10.0 - 5.0

        X_obs = []
        y_obs = []
        for i in range(n_lhs):
            if calls >= self.budget or time.time() > deadline:
                break
            xi = X[i]
            fi = safe_func(xi)
            if fi < np.inf:
                X_obs.append(xi)
                y_obs.append(fi)

        if not X_obs:
            # Fallback: random search
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                safe_func(x)
            return x_best, f_best

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)

        # --- Phase 2: Rastrigin-aware coordinate local search with ridge-jumping ---
        # Sort by objective value, take top candidates for local refinement
        sorted_indices = np.argsort(y_obs)
        n_local = min(10, len(sorted_indices))

        # Rastrigin period is 2*pi ≈ 6.28
        period = 2 * np.pi

        for k in range(n_local):
            if calls >= self.budget or time.time() > deadline:
                break

            x_curr = X_obs[sorted_indices[k]].copy()
            f_curr = safe_func(x_curr)

            if f_curr >= np.inf:
                continue

            # Coordinate-wise line search with ridge-jumping
            max_iter = 50
            for it in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break

                improved = False
                for d in range(dim):
                    if calls >= self.budget or time.time() > deadline:
                        break

                    # Try small perturbations in each coordinate
                    for delta in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0,
                                  period/2, -period/2, period, -period]:
                        if calls >= self.budget or time.time() > deadline:
                            break

                        x_trial = x_curr.copy()
                        x_trial[d] += delta
                        x_trial = np.clip(x_trial, -5, 5)

                        f_trial = safe_func(x_trial)
                        if f_trial < f_curr:
                            x_curr = x_trial.copy()
                            f_curr = f_trial
                            improved = True
                            break  # Move to next coordinate

                if not improved:
                    # Try ridge-jumping: large steps to escape local minima
                    for d in range(dim):
                        if calls >= self.budget or time.time() > deadline:
                            break

                        # Jump by half-period or full period in a random direction
                        for jump in [period/2, -period/2, period, -period]:
                            if calls >= self.budget or time.time() > deadline:
                                break

                            x_trial = x_curr.copy()
                            x_trial[d] += jump
                            x_trial = np.clip(x_trial, -5, 5)

                            f_trial = safe_func(x_trial)
                            if f_trial < f_curr:
                                x_curr = x_trial.copy()
                                f_curr = f_trial
                                improved = True
                                break

                        if improved:
                            break

                if not improved:
                    break

            if f_curr < f_best:
                f_best = f_curr
                x_best = x_curr.copy()

        # --- Phase 3: Additional random starts with local refinement ---
        n_extra = min(20, max(0, (self.budget - calls) // 30))
        for _ in range(n_extra):
            if calls >= self.budget or time.time() > deadline:
                break

            x_start = rng.uniform(-5, 5, dim)
            f_start = safe_func(x_start)
            if f_start >= np.inf:
                continue

            # Quick coordinate local search
            x_curr = x_start.copy()
            f_curr = f_start
            for it in range(20):
                if calls >= self.budget or time.time() > deadline:
                    break

                improved = False
                for d in range(dim):
                    if calls >= self.budget or time.time() > deadline:
                        break

                    for delta in [0.1, -0.1, 0.5, -0.5]:
                        if calls >= self.budget or time.time() > deadline:
                            break

                        x_trial = x_curr.copy()
                        x_trial[d] += delta
                        x_trial = np.clip(x_trial, -5, 5)

                        f_trial = safe_func(x_trial)
                        if f_trial < f_curr:
                            x_curr = x_trial.copy()
                            f_curr = f_trial
                            improved = True
                            break

                if not improved:
                    break

            if f_curr < f_best:
                f_best = f_curr
                x_best = x_curr.copy()

        # --- Phase 4: Final polishing with fine coordinate search ---
        if x_best is not None and calls < self.budget - 10 and time.time() < deadline:
            x_curr = x_best.copy()
            f_curr = f_best
            for it in range(30):
                if calls >= self.budget or time.time() > deadline:
                    break

                improved = False
                for d in range(dim):
                    if calls >= self.budget or time.time() > deadline:
                        break

                    for delta in [0.01, -0.01, 0.05, -0.05]:
                        if calls >= self.budget or time.time() > deadline:
                            break

                        x_trial = x_curr.copy()
                        x_trial[d] += delta
                        x_trial = np.clip(x_trial, -5, 5)

                        f_trial = safe_func(x_trial)
                        if f_trial < f_curr:
                            x_curr = x_trial.copy()
                            f_curr = f_trial
                            improved = True
                            break

                if not improved:
                    break

            if f_curr < f_best:
                f_best = f_curr
                x_best = x_curr.copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
