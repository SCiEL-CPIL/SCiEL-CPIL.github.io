"""Bayesian Optimization with GP surrogates and EHVI acquisition for RE34 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """Bayesian multi-objective optimizer using GP surrogates and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        # Storage for all evaluated points
        X_eval = []
        F_eval = []

        def evaluate(x):
            """Evaluate a single point and store it."""
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X_eval.append(x)
            F_eval.append(f)
            return f

        # Initial random sampling
        n_init = min(15, budget)
        rng = np.random.default_rng(42)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        # GP-based Bayesian optimization loop

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points with given reference point.
            Points must be strictly better than ref in all objectives."""
            points = np.array(points)
            mask = np.all(points < ref, axis=1)
            points = points[mask]
            if len(points) == 0:
                return 0.0

            n = len(points)
            if n == 1:
                return float(np.prod(ref - points[0]))

            # Sort by first coordinate
            order = np.argsort(points[:, 0])
            points = points[order]

            hv = 0.0
            for i in range(n):
                x_min = points[i, 0]
                x_max = points[i + 1, 0] if i + 1 < n else ref[0]

                active = [points[j, 1:3] for j in range(i, n) if points[j, 0] < x_max]
                if not active:
                    continue

                hv_2d = _hypervolume_2d(active, ref[1:3])
                hv += hv_2d * (x_max - x_min)

            return hv

        def _hypervolume_2d(points, ref):
            """Compute 2D hypervolume for points with reference point."""
            points = np.array(points)
            if len(points) == 0:
                return 0.0

            idx = np.argsort(points[:, 0])
            points = points[idx]
            n = len(points)

            max_z = np.full(n, -np.inf)
            for i in range(n - 1, -1, -1):
                max_z[i] = max(points[i, 1], max_z[i + 1]) if i + 1 < n else points[i, 1]

            hv = 0.0
            for i in range(n):
                y_next = points[i + 1, 0] if i + 1 < n else ref[0]
                width = y_next - points[i, 0]
                if width > 0:
                    hv += width * max_z[i]

            return hv

        def ehvi_acquisition(X_train, F_train, X_cand, ref, n_samples=200):
            """Compute Expected Hypervolume Improvement for candidate points.
            Uses Monte Carlo sampling from GP posteriors."""
            d = X_train.shape[1]
            m = F_train.shape[1]

            # Standardize training data
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0) + 1e-8
            X_train_std = (X_train - X_mean) / X_std

            # Fit GPs for each objective
            gps = []
            for j in range(m):
                y = F_train[:, j]
                y_mean = y.mean()
                y_std = y.std() + 1e-8
                y_std_arr = (y - y_mean) / y_std

                def rbf(K1, K2, ls=0.5):
                    sq_dists = np.sum(K1 ** 2, axis=1)[:, None] + np.sum(K2 ** 2, axis=1)[None, :] - 2 * K1 @ K2.T
                    return np.exp(-0.5 * sq_dists / (ls ** 2))

                K = rbf(X_train_std, X_train_std) + 1e-8 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_std_arr))
                except:
                    alpha = np.linalg.pinv(K) @ y_std_arr

                gps.append((y_mean, y_std, X_train_std, L, alpha))

            def gp_sample(gps, X_cand_std, n_samples):
                """Sample from GP posteriors at candidate points."""
                n_cand = len(X_cand_std)
                samples = np.zeros((n_samples, n_cand, m))
                for j in range(m):
                    y_mean, y_std, X_train_std, L, alpha = gps[j]
                    K_star = rbf(X_train_std, X_cand_std)
                    v = np.linalg.solve(L, K_star.T)
                    mu = v.T @ alpha
                    var = 1.0 - np.sum(v ** 2, axis=0)
                    std = np.sqrt(np.maximum(var, 0))
                    eps = rng.standard_normal((n_samples, n_cand))
                    samples[:, :, j] = (mu[None, :] + std[None, :] * eps) * y_std + y_mean
                return samples

            # Baseline HV
            F_baseline = np.array(F_train)
            mask = np.all(F_baseline < ref, axis=1)
            F_base = F_baseline[mask]
            hv_base = hypervolume_3d(F_base, ref) if len(F_base) > 0 else 0.0

            # Evaluate EHVI for each candidate
            n_cand = len(X_cand)
            X_cand_std = (X_cand - X_mean) / X_std

            ehvi = np.zeros(n_cand)
            for i in range(n_cand):
                samples = gp_sample(gps, X_cand_std[i:i + 1], n_samples)[:, 0, :]

                total_improvement = 0.0
                for s in range(n_samples):
                    new_point = samples[s]
                    if np.all(new_point < ref):
                        all_points = np.vstack([F_base, new_point]) if len(F_base) > 0 else new_point.reshape(1, -1)
                        hv_new = hypervolume_3d(all_points, ref)
                    else:
                        hv_new = hv_base
                    total_improvement += max(0, hv_new - hv_base)

                ehvi[i] = total_improvement / n_samples

            return ehvi

        # Main BO loop
        while len(X_eval) < budget:
            F_arr = np.array(F_eval)
            X_arr = np.array(X_eval)

            # Find non-dominated points
            n_pts = len(F_arr)
            is_dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dominated[i] = True
                        break
            pareto_idx = np.where(~is_dominated)[0]
            pareto_X = X_arr[pareto_idx]

            if len(pareto_X) < 3:
                x = lo + (hi - lo) * rng.random(dim)
                evaluate(x)
                continue

            # Generate candidates
            n_cand = 50
            candidates = []

            # Some candidates near Pareto points
            for i in range(min(10, len(pareto_X))):
                for _ in range(3):
                    x = pareto_X[i] + 0.1 * (hi - lo) * rng.standard_normal(dim)
                    x = np.clip(x, lo, hi)
                    candidates.append(x)

            # Some random candidates
            for _ in range(20):
                x = lo + (hi - lo) * rng.random(dim)
                candidates.append(x)

            # Some candidates in high-uncertainty regions (based on GP)
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            lhc = sampler.random(n=20)
            for x in lhc:
                x = lo + (hi - lo) * x
                candidates.append(x)

            candidates = np.array(candidates[:n_cand])

            # Compute EHVI for each candidate
            try:
                ehvi_vals = ehvi_acquisition(X_arr, F_arr, candidates, ref, n_samples=50)
                best_idx = np.argmax(ehvi_vals)
            except:
                best_idx = rng.integers(len(candidates))

            x_next = candidates[best_idx]
            evaluate(x_next)

        return X_eval
# EVOLVE-BLOCK-END
