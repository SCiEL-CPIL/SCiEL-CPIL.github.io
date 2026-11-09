"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Multi-objective optimizer for RE37 using Gaussian Process surrogate
    with Expected Hypervolume Improvement (EHVI) acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        ref = np.array([1.1, 1.3, 1.1])

        # Kernel parameters (RBF)
        length_scales = np.ones(dim)
        signal_var = 1.0
        noise_var = 1e-4

        def rbf_kernel(x1, x2):
            """Compute RBF kernel between arrays of points."""
            # x1: (n1, dim), x2: (n2, dim)
            diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            return signal_var * np.exp(-0.5 * sq_dist / np.outer(length_scales, length_scales) ** 2)

        def fit_gp(X_train, Y_train):
            """Fit GP: return K, alpha, L for efficient prediction."""
            n = len(X_train)
            K = rbf_kernel(X_train, X_train) + noise_var * np.eye(n)
            # Cholesky with jitter for stability
            for jitter in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
                try:
                    L = np.linalg.cholesky(K + jitter * np.eye(n))
                    break
                except np.linalg.LinAlgError:
                    continue
            else:
                # Fallback: add more noise
                K += 0.1 * np.eye(n)
                L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            return K, alpha, L

        def predict_gp(X_new, X_train, Y_train, L, alpha):
            """Predict mean and variance at X_new points."""
            K_star = rbf_kernel(X_new, X_train)  # (n_new, n_train)
            K_star_inv_L = np.linalg.solve(L.T, K_star.T)  # (n_train, n_new)
            mu = (K_star @ alpha).T  # (n_new,)
            v = np.linalg.solve(L, K_star.T)  # (n_train, n_new)
            var = signal_var - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, var

        def compute_hv_3d(points, ref_point):
            """Compute 3D hypervolume using grid-based method."""
            valid = [p for p in points if np.all(p < ref_point)]
            if not valid:
                return 0.0
            xs = sorted(set([p[0] for p in valid] + [ref_point[0]]))
            ys = sorted(set([p[1] for p in valid] + [ref_point[1]]))
            zs = sorted(set([p[2] for p in valid] + [ref_point[2]]))
            total = 0.0
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    for k in range(len(zs) - 1):
                        dominated = False
                        for p in valid:
                            if p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]:
                                dominated = True
                                break
                        if dominated:
                            total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]) * (zs[k + 1] - zs[k])
            return total

        def non_dominated_front(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]:
                    continue
                for j in range(n):
                    if i == j or is_dominated[j]:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        is_dominated[j] = True
            return [i for i in range(n) if not is_dominated[i]]

        def ehvi_acquisition(x_candidates, X_train, Y_train, L, alpha, current_front):
            """Compute EHVI for candidate points. Returns EHVI values."""
            n_cand = len(x_candidates)
            # Predict mean and std for each objective at candidates
            mu = np.zeros((n_cand, n_obj))
            std = np.zeros((n_cand, n_obj))
            for obj in range(n_obj):
                m, v = predict_gp(x_candidates, X_train[:, obj], Y_train[:, obj], L, alpha)
                mu[:, obj] = m
                std[:, obj] = np.sqrt(v)

            # Current HV of the front
            current_hv = compute_hv_3d(current_front, ref)

            # MC estimate of EHVI
            n_samples = 200
            rng = np.random.RandomState(123)
            ehvi_vals = np.zeros(n_cand)

            # For each objective, sample from GP posterior
            for s in range(n_samples):
                # Sample objective values: f_s = mu + std * z
                z = rng.standard_normal((n_cand, n_obj))
                f_samples = mu + std * z  # (n_cand, n_obj)

                # For each candidate, compute HV of (current_front + f_s)
                for i in range(n_cand):
                    new_point = f_samples[i]
                    # Only consider if it could potentially improve
                    if np.all(new_point < ref):
                        new_front = current_front + [new_point]
                        new_hv = compute_hv_3d(new_front, ref)
                        improvement = new_hv - current_hv
                        if improvement > 0:
                            ehvi_vals[i] += improvement

            ehvi_vals /= n_samples
            return ehvi_vals

        # Initialize with random points
        rng = np.random.default_rng(42)
        init_size = 16
        X_init = lo + (hi - lo) * rng.random((init_size, dim))
        Y_init = np.array([problem(x) for x in X_init])
        used = init_size

        X_all = X_init.copy()
        Y_all = Y_init.copy()

        # Main BO loop
        while used < budget:
            if time.time() > deadline:
                break

            # Get current non-dominated front
            front_indices = non_dominated_front(Y_all)
            current_front = [Y_all[i] for i in front_indices]

            # Generate candidate points
            n_candidates = 50
            X_cand = lo + (hi - lo) * rng.random((n_candidates, dim))

            # Fit GP for each objective
            Ls = []
            alphas = []
            for obj in range(n_obj):
                K, alpha, L = fit_gp(X_all[:, obj * dim: (obj + 1) * dim] if dim == 1 else X_all, Y_all[:, obj], )
                # Actually X_all is (n, dim), we need to fit GP per objective
                # Let me redo: fit GP on X_all for each objective
                pass

            # Redo GP fitting properly
            Ls = []
            alphas = []
            for obj in range(n_obj):
                K, alpha, L = fit_gp(X_all, Y_all[:, obj])
                Ls.append(L)
                alphas.append(alpha)

            # Compute EHVI
            ehvi_vals = np.zeros(n_candidates)
            current_hv = compute_hv_3d(current_front, ref)

            n_mc = 100
            rng_mc = np.random.RandomState(456)

            for obj in range(n_obj):
                mu_obj, var_obj = predict_gp(X_cand, X_all, Y_all[:, obj], Ls[obj], alphas[obj])
                std_obj = np.sqrt(var_obj)
                # Store for combined sampling
                if obj == 0:
                    mu_all = np.zeros((n_candidates, n_obj))
                    std_all = np.zeros((n_candidates, n_obj))
                mu_all[:, obj] = mu_obj
                std_all[:, obj] = std_obj

            for s in range(n_mc):
                z = rng_mc.standard_normal((n_candidates, n_obj))
                f_samples = mu_all + std_all * z
                for i in range(n_candidates):
                    if np.all(f_samples[i] < ref):
                        new_front = current_front + [f_samples[i]]
                        new_hv = compute_hv_3d(new_front, ref)
                        improvement = new_hv - current_hv
                        if improvement > 0:
                            ehvi_vals[i] += improvement

            ehvi_vals /= n_mc

            # Select best candidate
            best_idx = np.argmax(ehvi_vals)
            x_new = X_cand[best_idx]
            f_new = problem(x_new)
            used += 1

            X_all = np.vstack([X_all, x_new])
            Y_all = np.vstack([Y_all, f_new])

        return X_all
# EVOLVE-BLOCK-END
