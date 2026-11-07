import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 285.0  # Safety margin before 330s kill

        def time_ok():
            return time.time() - start_time < max_time

        calls = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f_val = float(func(x))
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # ---- Phase 1: Coarse grid + random to find promising basin ----
        # Lunacek has two basins: one near 0 and one near d* (shifted by hidden offset)
        # Do a quick coarse search to identify which region is better

        # Start with center and a few structured points
        candidates = [np.zeros(dim)]
        # Random points
        rng = np.random.RandomState(42)
        for _ in range(10):
            candidates.append(rng.uniform(-5, 5, dim))
        # Points along diagonals (helps find the shifted basin)
        for s in [1.5, 2.5, 3.0, -1.5, -2.5, -3.0]:
            candidates.append(np.full(dim, s))

        for x0 in candidates:
            if not time_ok() or calls >= budget:
                break
            f0 = wrapped_func(x0)

        # ---- Phase 2: Population-based CMA-ES on best region ----
        # Use a (mu+lambda) CMA-ES with moderate population
        x_mean = best_x.copy() if best_x is not None else np.zeros(dim)
        f_mean = best_f

        # Estimate initial sigma from the spread of evaluated points
        sigma = 2.0

        # CMA-ES parameters
        mu = max(1, dim // 2)       # number of parents
        lam = max(4 * mu, dim + 2)  # number of offspring
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + cs + 2 / max(dim - 1, 1)
        c1 = (2 - mueff / (dim + 1) + 2 * mueff / (dim + 1)) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        n_restart = 1
        total_calls_spent = calls

        # Run CMA-ES until budget or time
        while calls < budget and time_ok():
            # Generate lambda candidates
            offspring = []
            offspring_f = []

            for _ in range(lam):
                if not time_ok() or calls >= budget:
                    break
                z = rng.randn(dim)
                # Transform z by C: use Cholesky or eigendecomposition
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    step = L @ z
                except np.linalg.LinAlgError:
                    step = z
                x_trial = x_mean + sigma * step
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = wrapped_func(x_trial)
                offspring.append((f_trial, x_trial))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Select best mu
            selected = offspring[:min(mu, len(offspring))]
            selected_f = np.array([s[0] for s in selected])
            selected_x = np.array([s[1] for s in selected])

            # Update mean (weighted)
            w = weights[:len(selected)]
            new_mean = (w[:, None] * selected_x).sum(axis=0)

            # Update step-size control
            y_mean = (new_mean - x_mean) / (sigma + 1e-30)
            h_sigma = min(1.0, (np.sum(weights * np.abs(
                (selected_x - x_mean) / (sigma + 1e-30)
            )) + 1e-10) / (dim + 1e-10))
            # Simplified: just update
            p_sigma = (1 - cs) * p_sigma + cs * y_mean
            norm_ps = np.linalg.norm(p_sigma)
            hat_cs = (1 - cs) ** 0.5
            sigma *= np.exp(cs / ds * (norm_ps / np.sqrt(2 * dim) - 1))
            sigma = max(1e-10, min(sigma, 50.0))

            # Update covariance
            y_c = (new_mean - x_mean) / sigma
            h_c = min(1.0, np.sqrt(
                (dim + 2) / (dim - 1) - 1
            ))
            p_c = (1 - cs) * p_c + cs * y_c

            # Rank-one update
            C = (1 - cs * (2 - cs) - c1 - cmu) * C + cs * (2 - cs) * np.outer(y_c, y_c)
            # Rank-mu update
            C += cmu * (w[:, None, None] * selected_x[:, None, :] @ selected_x[:, None, :].T) / (sigma ** 2 + 1e-20)
            # Actually rank-mu should be:
            C = (1 - c1 - cmu) * C + c1 * np.outer(y_c, y_c)
            C += cmu * np.sum(w[:, None, None] * (
                (selected_x - x_mean) / (sigma + 1e-30)
            )[:, None, :] * ((selected_x - x_mean) / (sigma + 1e-30)[:, None, :]), axis=0)

            C = (C + C.T) / 2

            x_mean = new_mean
            f_mean = selected_f[0]

            # Restart if sigma too small
            if sigma < 1e-7:
                break

        # ---- Phase 3: Local polish with Nelder-Mead if time permits ----
        if time_ok() and calls < budget and best_x is not None:
            remaining = budget - calls
            if remaining > 15:
                try:
                    from scipy.optimize import minimize
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
