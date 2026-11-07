"""Bayesian Optimization (Gaussian Process) optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """GP-based Bayesian Optimizer for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X, X_obs, y_obs, length_scale, noise):
        """Predict mean and variance at X using GP with Matern-5/2 kernel."""
        n = X_obs.shape[0]
        d = self.dim
        # Matern-5/2 kernel
        def kernel(A, B):
            # A: (m, d), B: (k, d)
            diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
            r = np.sqrt(np.sum(diff ** 2, axis=-1))
            sr = np.sqrt(5.0) * r / length_scale
            K = (1.0 + sr + sr * sr / 3.0) * np.exp(-sr)
            return K

        K = kernel(X_obs, X_obs)
        K[np.diag_indices_from(K)] += noise
        KX = kernel(X_obs, X)
        # Solve K * alpha = y_obs
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            v = np.linalg.solve(L, KX)
        except np.linalg.LinAlgError:
            # Fallback: add jitter
            K[np.diag_indices_from(K)] += 1e-6
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
                v = np.linalg.solve(L, KX)
            except np.linalg.LinAlgError:
                return np.zeros(X.shape[0]), np.ones(X.shape[0])

        KXX = kernel(X, X)
        mean = KX.T @ alpha
        var = np.diag(KXX) - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mean, var

    def _expected_improvement(self, mu, sigma, f_best, xi=0.01):
        """Compute Expected Improvement acquisition."""
        sigma = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        n = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # --- Phase 1: Initial exploration (uniform + random) ---
        # Evaluate a diverse set of initial points
        init_points = [np.zeros(n)]
        for _ in range(7):
            init_points.append(rng.uniform(-4.0, 4.0, n))
        # Add some structured points
        init_points.append(np.array([2.0] * n))
        init_points.append(np.array([-2.0] * n))

        X_obs = []
        y_obs = []
        for p in init_points:
            if calls_made >= self.budget or time.time() > deadline - 10:
                break
            val = safe_call(p)
            if val is not None:
                X_obs.append(p)
                y_obs.append(val)

        # --- Phase 2: GP-guided Bayesian Optimization ---
        length_scale = 2.0
        noise = 1e-6
        batch_size = 5
        f_best_for_ei = best_f

        while calls_made < self.budget and time.time() < deadline - 15:
            if len(X_obs) < 2:
                # Not enough data for GP, do random sampling
                for _ in range(batch_size):
                    if calls_made >= self.budget or time.time() > deadline - 15:
                        break
                    p = rng.uniform(-4.0, 4.0, n)
                    val = safe_call(p)
                    if val is not None:
                        X_obs.append(p)
                        y_obs.append(val)
                continue

            X_obs_arr = np.array(X_obs)
            y_obs_arr = np.array(y_obs)
            f_best_for_ei = np.min(y_obs_arr)

            # Generate candidate points and evaluate EI
            # Use a mix of random candidates and points near the current best
            n_candidates = 200
            candidates = rng.uniform(-4.5, 4.5, (n_candidates, n))
            # Add candidates near the best point
            if best_x is not None:
                local_candidates = best_x + rng.normal(0, 0.5, (50, n))
                local_candidates = np.clip(local_candidates, -4.5, 4.5)
                candidates = np.vstack([candidates, local_candidates])

            # Predict mean and variance
            mu, var = self._gp_predict(candidates, X_obs_arr, y_obs_arr, length_scale, noise)
            sigma = np.sqrt(var)

            # Compute EI
            ei = self._expected_improvement(mu, sigma, f_best_for_ei, xi=0.005)

            # Select top batch_size candidates with diversity
            # Greedy selection with diversity penalty
            selected = []
            ei_sorted_idx = np.argsort(-ei)
            for idx in ei_sorted_idx:
                if len(selected) >= batch_size:
                    break
                cand = candidates[idx]
                # Check diversity: distance to already selected
                if selected:
                    dists = np.array([np.linalg.norm(cand - s) for s in selected])
                    if np.min(dists) < 0.1:
                        continue
                selected.append(cand)

            # If not enough diverse points, just take top by EI
            if len(selected) < batch_size:
                for idx in ei_sorted_idx:
                    if len(selected) >= batch_size:
                        break
                    cand = candidates[idx]
                    if cand not in selected:
                        selected.append(cand)

            # Evaluate selected points
            for cand in selected:
                if calls_made >= self.budget or time.time() > deadline - 15:
                    break
                val = safe_call(cand)
                if val is not None:
                    X_obs.append(cand)
                    y_obs.append(val)

            # Adapt length scale: shrink as we converge
            if best_f < 1.0:
                length_scale = max(0.3, length_scale * 0.95)
            elif best_f < 10.0:
                length_scale = max(0.5, length_scale * 0.97)

        # --- Phase 3: Local polish with Nelder-Mead ---
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 10:
            # Run Nelder-Mead from the best point
            x0 = best_x.copy()
            best_f_nm = best_f

            def nm_func(x):
                nonlocal calls_made, best_f, best_x
                if calls_made >= self.budget or time.time() > deadline - 5:
                    return best_f_nm
                x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
                val = func(x)
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                    best_f_nm = val
                return val

            # Try Nelder-Mead with adaptive initial simplex
            for attempt in range(3):
                if calls_made >= self.budget or time.time() > deadline - 5:
                    break
                step = 0.3 if attempt == 0 else (0.1 if attempt == 1 else 0.05)
                x_start = best_x + rng.normal(0, step * 0.5, n)
                x_start = np.clip(x_start, -4.5, 4.5)

                try:
                    res = minimize(
                        nm_func, x_start, method='Nelder-Mead',
                        options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200,
                                 'adaptive': True, 'initial_simplex': step}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
