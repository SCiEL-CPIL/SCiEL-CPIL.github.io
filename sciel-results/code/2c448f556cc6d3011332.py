import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """
    Surrogate-assisted global optimizer using a Gaussian Process model
    for exploration, with L-BFGS-B local exploitation on promising points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        # GP data storage
        X_gp = []
        y_gp = []

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("time exceeded")
            return float(func(x))

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def gp_predict(X_test):
            """Gaussian Process prediction with RBF kernel."""
            n = len(X_gp)
            if n == 0:
                return np.zeros(X_test.shape[0]), np.zeros(X_test.shape[0])
            if n == 1:
                return np.full(X_test.shape[0], y_gp[0]), np.full(X_test.shape[0], 1.0)

            y = np.array(y_gp)
            y_mean = y.mean()
            y_centered = y - y_mean

            # RBF kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    diff = X_gp[i] - X_gp[j]
                    K[i, j] = np.exp(-0.5 * np.sum(diff ** 2))
            K += 1e-8 * np.eye(n)

            # Kernel between test and training
            K_s = np.zeros((X_test.shape[0], n))
            for i in range(X_test.shape[0]):
                for j in range(n):
                    diff = X_test[i] - X_gp[j]
                    K_s[i, j] = np.exp(-0.5 * np.sum(diff ** 2))

            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)

            mu = y_mean + K_s @ K_inv @ y_centered
            var = 1.0 - K_s @ K_inv @ K_s.T
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)

            return mu, sigma

        def expected_improvement(mu, sigma, f_best):
            """EI acquisition function for minimization."""
            improvement = f_best - mu
            z = improvement / (sigma + 1e-10)
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        rng = np.random.default_rng(42)

        # --- Phase 1: Initial diverse sampling ---
        n_initial = min(20, budget // 4)
        starts = [np.zeros(dim)]
        for _ in range(n_initial - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                f = wrapped(s)
                update_best(s, f)
                X_gp.append(s.copy())
                y_gp.append(f)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)
            X_gp.append(best_x.copy())
            y_gp.append(best_f)

        # --- Phase 2: Surrogate-assisted search loop ---
        while calls[0] < budget and time.time() < deadline:
            # Check if we have enough data for GP
            if len(X_gp) >= 3:
                # Generate candidate points
                n_candidates = 50
                candidates = []

                # Random candidates
                for _ in range(n_candidates // 2):
                    candidates.append(rng.uniform(-5, 5, dim))

                # Perturbations around best
                for _ in range(n_candidates // 4):
                    pert = rng.standard_normal(dim) * 1.0
                    candidates.append(np.clip(best_x + pert, -5, 5))

                # Perturbations around recent good points
                if len(X_gp) > 5:
                    recent = X_gp[-5:]
                    for rp in recent:
                        pert = rng.standard_normal(dim) * 0.5
                        candidates.append(np.clip(rp + pert, -5, 5))

                # Deduplicate and clip
                unique_candidates = []
                for c in candidates:
                    c = np.clip(c, -5, 5)
                    if not any(np.allclose(c, u, atol=0.05) for u in unique_candidates):
                        unique_candidates.append(c)

                if not unique_candidates:
                    unique_candidates = [rng.uniform(-5, 5, dim)]

                cand_arr = np.array(unique_candidates)

                # GP predictions
                try:
                    mu, sigma = gp_predict(cand_arr)
                    ei = expected_improvement(mu, sigma, best_f)

                    # Select top candidate by EI
                    best_idx = np.argmax(ei)
                    next_x = cand_arr[best_idx]
                except Exception:
                    # Fallback: random perturbation
                    next_x = np.clip(best_x + rng.standard_normal(dim) * 1.0, -5, 5)
            else:
                # Not enough data yet, use random
                next_x = rng.uniform(-5, 5, dim)

            # Evaluate the selected point
            try:
                f_next = wrapped(next_x)
                X_gp.append(next_x.copy())
                y_gp.append(f_next)

                # Local exploitation if improvement
                if f_next < best_f:
                    update_best(next_x, f_next)
                    # Run local optimization from this point
                    if calls[0] < budget - 10 and time.time() < deadline:
                        try:
                            res = minimize(
                                wrapped,
                                next_x,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * dim,
                                options={'maxiter': 30, 'maxls': 15, 'gtol': 1e-5}
                            )
                            update_best(res.x, res.fun)
                            X_gp.append(res.x.copy())
                            y_gp.append(res.fun)
                        except Exception:
                            pass
                else:
                    # No improvement; try local opt from current best occasionally
                    if calls[0] < budget - 10 and time.time() < deadline and rng.random() < 0.3:
                        try:
                            res = minimize(
                                wrapped,
                                best_x,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * dim,
                                options={'maxiter': 20, 'maxls': 10, 'gtol': 1e-4}
                            )
                            if res.fun < best_f:
                                update_best(res.x, res.fun)
                                X_gp.append(res.x.copy())
                                y_gp.append(res.fun)
                        except Exception:
                            pass

            except Exception:
                pass

        # --- Phase 3: Final polish ---
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 30, 'gtol': 1e-8}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

            if calls[0] < budget and time.time() < deadline:
                try:
                    res_nm = minimize(
                        wrapped,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-10,
                                 'fatol': 1e-14, 'adaptive': True}
                    )
                    update_best(res_nm.x, res_nm.fun)
                except Exception:
                    pass

        return best_x, best_f
