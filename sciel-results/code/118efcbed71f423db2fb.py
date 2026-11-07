"""BBOB f14 (Different powers) optimizer.

Hybrid Bayesian Optimization (Gaussian Process) with local (1+1)-ES polish.
Uses Expected Improvement acquisition to guide global exploration, then
switches to local refinement once close to the optimum.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid BO + ES optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf
        x_data = []
        f_data = []

        def t_ok():
            return time.time() < deadline

        def ok():
            return t_ok() and calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def call(x):
            nonlocal calls
            calls += 1
            f = func(np.asarray(x, dtype=float))
            if np.isfinite(f):
                x_data.append(x.copy())
                f_data.append(f)
                record(x, f)
            return f

        # Phase 1: Latin Hypercube initial sampling
        n_init = 20
        lhs_points = []
        for d in range(dim):
            strata = np.linspace(-5, 5, n_init + 1)
            for i in range(n_init):
                lhs_points.append(rng.uniform(strata[i], strata[i + 1]))
        lhs_points = np.array(lhs_points).reshape(n_init, dim)

        for i in range(n_init):
            if not ok():
                break
            call(lhs_points[i])

        # Phase 2: Bayesian Optimization with GP surrogate
        # We'll implement a simple GP with RBF kernel

        def gp_predict(X_new, X_obs, y_obs, length_scale=1.0, signal_var=1.0, noise_var=1e-4):
            """Simple GP prediction with RBF kernel."""
            n = len(X_obs)
            m = len(X_new)

            # Compute kernel matrix K(X_obs, X_obs)
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    diff = X_obs[i] - X_obs[j]
                    K[i, j] = signal_var * np.exp(-0.5 * np.sum(diff ** 2) / length_scale ** 2)
            K += noise_var * np.eye(n)

            # Compute K(X_new, X_obs)
            Ks = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    diff = X_new[i] - X_obs[j]
                    Ks[i, j] = signal_var * np.exp(-0.5 * np.sum(diff ** 2) / length_scale ** 2)

            # Solve for mean and variance
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)

            mu = Ks @ K_inv @ y_obs
            var = signal_var - np.sum(Ks @ K_inv * Ks, axis=1)
            var = np.maximum(var, 1e-8)

            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, f_best, xi=0.01):
            """Compute EI acquisition values."""
            improvement = f_best - mu - xi
            sigma = np.maximum(sigma, 1e-8)
            z = improvement / sigma
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            ei[improvement < 0] = 0
            return ei

        def optimize_gp(max_iter=80):
            """Run GP-based Bayesian optimization."""
            nonlocal best_x, best_f, x_data, f_data, calls

            if len(x_data) < 2:
                return

            X_obs = np.array(x_data)
            y_obs = np.array(f_data)
            f_best_local = np.min(y_obs)

            # Adaptive length scale based on spread of points
            if len(X_obs) > 1:
                dists = []
                for i in range(len(X_obs)):
                    for j in range(i + 1, len(X_obs)):
                        dists.append(np.linalg.norm(X_obs[i] - X_obs[j]))
                length_scale = max(0.5, np.median(dists) * 0.5) if dists else 1.0
            else:
                length_scale = 1.0

            signal_var = max(1.0, np.std(y_obs) ** 2 + 1.0)

            for it in range(max_iter):
                if not ok():
                    break

                # Generate candidate points
                n_cand = 200
                candidates = rng.uniform(-5, 5, (n_cand, dim))

                # Predict on candidates
                mu, sigma = gp_predict(candidates, X_obs, y_obs, length_scale, signal_var)

                # Compute EI
                ei = expected_improvement(mu, sigma, f_best_local, xi=0.01)

                # Select best candidate by EI
                best_idx = int(np.argmax(ei))
                x_new = candidates[best_idx]

                f_new = call(x_new)

                if f_new < f_best_local:
                    f_best_local = f_new
                    # Adapt length scale to be smaller as we get closer
                    length_scale *= 0.8
                    signal_var = max(0.1, signal_var * 0.9)

        # Phase 3: Local (1+1)-ES polish on best point
        def polish_es(x_start, f_start, max_iter=150):
            """(1+1)-ES local refinement."""
            nonlocal best_x, best_f, calls

            x = x_start.copy()
            f = f_start
            sigma = 0.5

            for it in range(max_iter):
                if not ok():
                    break

                # Adapt sigma based on recent success rate (simple heuristic)
                perturbation = rng.normal(0, sigma, dim)
                x_try = x + perturbation
                x_try = np.clip(x_try, -5, 5)

                f_try = call(x_try)

                if f_try < f:
                    x = x_try
                    f = f_try
                    record(x, f)
                    sigma = min(sigma * 1.1, 2.0)
                else:
                    sigma = max(sigma * 0.85, 0.01)

        # Run the phases
        # First do some GP exploration
        if ok():
            optimize_gp(max_iter=60)

        # Then local polish
        if best_x is not None and ok():
            polish_es(best_x, best_f, max_iter=150)

        # If we still have budget, do more GP rounds with tighter settings
        if ok():
            # Reset for another round of GP with smaller exploration
            optimize_gp(max_iter=40)

        # Final local polish
        if best_x is not None and ok():
            polish_es(best_x, best_f, max_iter=100)

        # Coordinate descent final polish
        if best_x is not None and ok():
            x_polish = best_x.copy()
            f_polish = best_f
            step_size = 0.1

            for _ in range(10):
                if not ok():
                    break
                improved_cd = False
                for d in range(dim):
                    if not ok():
                        break
                    for direction in [1, -1]:
                        x_try = x_polish.copy()
                        x_try[d] += direction * step_size
                        x_try = np.clip(x_try, -5, 5)
                        f_try = call(x_try)
                        if f_try < f_polish - 1e-12:
                            x_polish = x_try
                            f_polish = f_try
                            record(x_polish, f_polish)
                            improved_cd = True
                            break
                    if improved_cd:
                        break
                if not improved_cd:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
