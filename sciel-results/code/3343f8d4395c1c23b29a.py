"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class Optimizer:
    """Bayesian multi-objective optimizer with GP surrogate and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([3.0, 5.0, 7.0])

        # Initialize with random points
        n_init = min(20, self.budget)
        X = np.random.uniform(lo, hi, size=(n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # Track all evaluated points
        all_X = X.copy()
        all_F = F.copy()

        def hypervolume_3d(points, ref):
            """Compute hypervolume for 3D points dominated by ref."""
            pts = points[points < ref].copy()
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            # Remove dominated points (keep only non-dominated)
            nd_idx = []
            for i in range(len(pts)):
                dominated = False
                for j in range(len(pts)):
                    if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            pts = pts[nd_idx]
            if len(pts) == 0:
                return 0.0
            # Compute hypervolume using inclusion-exclusion for 3D
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_z = 0.0
            for i in range(len(pts)):
                z = pts[i, 0]
                if i < len(pts) - 1:
                    # Find min y and min z for points from i to end
                    sub = pts[i:, 1:]
                    min_y = sub[:, 0].min()
                    min_z = sub[:, 1].min()
                    hv += (ref[0] - z) * max(0, ref[1] - min_y) * max(0, ref[2] - min_z)
                    prev_z = z
                else:
                    hv += (ref[0] - z) * max(0, ref[1] - pts[i, 1]) * max(0, ref[2] - pts[i, 2])
            return hv

        def ehvi_acquisition(x, gp_models, ref_pt, current_hv):
            """Expected Hypervolume Improvement acquisition function."""
            x = np.atleast_2d(x)
            mean = np.zeros((1, self.n_obj))
            cov = np.zeros((1, self.n_obj, self.n_obj))
            for obj in range(self.n_obj):
                m, s = gp_models[obj].predict(x, return_std=True)
                mean[0, obj] = m[0]
                cov[0, obj, obj] = s[0] ** 2

            # Sample-based EHVI estimation
            n_samples = 100
            total_ehvi = 0.0
            for _ in range(n_samples):
                sample = mean + np.sqrt(np.diag(cov[0])) * rng.standard_normal(self.n_obj)
                # Compute hypervolume with and without this point
                current_pts = all_F[all_F < ref_pt]
                if len(current_pts) == 0:
                    hv_without = 0.0
                else:
                    hv_without = hypervolume_3d(current_pts, ref_pt)
                new_pts = np.vstack([current_pts, sample]) if len(current_pts) > 0 else sample
                if len(new_pts) == 0:
                    hv_with = 0.0
                else:
                    hv_with = hypervolume_3d(new_pts, ref_pt)
                ehvi = max(0, hv_with - hv_without)
                total_ehvi += ehvi
            return -total_ehvi / n_samples  # Negate for minimization

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Train GP models for each objective
            valid_mask = np.all(np.isfinite(all_F), axis=1)
            if np.sum(valid_mask) < 2:
                # Not enough data, just do random search
                x = np.random.uniform(lo, hi, size=self.dim)
                f = problem(x)
                used += 1
                all_X = np.vstack([all_X, x])
                all_F = np.vstack([all_F, f])
                continue

            X_train = all_X[valid_mask]
            F_train = all_F[valid_mask]

            gp_models = []
            for obj in range(self.n_obj):
                y = F_train[:, obj]
                y_mean = y.mean()
                y_std = max(y.std(), 1e-8)
                y_norm = (y - y_mean) / y_std
                kernel = Matern(length_scale=np.ones(self.dim) * 0.5, nu=2.5) + WhiteKernel(noise_level=1e-4)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
                gp.fit(X_train, y_norm)
                gp._y_mean = y_mean
                gp._y_std = y_std
                gp_models.append(gp)

            # Compute current hypervolume
            valid_F = all_F[np.all(all_F < ref_point, axis=1)]
            if len(valid_F) > 0:
                current_hv = hypervolume_3d(valid_F, ref_point)
            else:
                current_hv = 0.0

            # Find candidate points using multi-start optimization of EHVI
            n_candidates = 10
            best_x = None
            best_ehvi = -np.inf

            # Generate diverse starting points
            start_points = np.random.uniform(lo, hi, size=(20, self.dim))

            for sp in start_points:
                def obj_func(x):
                    return ehvi_acquisition(x, gp_models, ref_point, current_hv)

                res = minimize(obj_func, sp, bounds=list(zip(lo, hi)), method='L-BFGS-B',
                               options={'maxiter': 50, 'ftol': 1e-6})
                if res.fun < best_ehvi:
                    best_ehvi = res.fun
                    best_x = res.x

            if best_x is None:
                best_x = np.random.uniform(lo, hi, size=self.dim)

            # Evaluate the best candidate
            f = problem(best_x)
            used += 1
            all_X = np.vstack([all_X, best_x])
            all_F = np.vstack([all_F, f])

        return all_X
# EVOLVE-BLOCK-END
