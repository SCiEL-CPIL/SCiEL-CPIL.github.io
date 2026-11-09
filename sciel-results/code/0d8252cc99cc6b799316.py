import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Multi-objective optimizer for RE34: GP-based Bayesian optimization with EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fit_gp(self, X, y, var_y, nugget):
        """Fit a simple GP: returns mu, cov at given points."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(y)
        # RBF kernel
        d2 = np.sum(X ** 2, axis=1)
        D2 = d2[:, None] + d2[None, :] - 2 * X @ X.T
        np.maximum(D2, 0, out=D2)
        K = var_y * np.exp(-D2 / (2 * 0.25 ** 2))
        K[np.diag_indices(n)] += nugget
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K[np.diag_indices(n)] += 1e-6
            L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return K, L, alpha, var_y

    def _gp_predict(self, Xnew, K, L, alpha, var_y):
        """Predict mean and std at Xnew."""
        Xnew = np.asarray(Xnew, float)
        d2 = np.sum(Xnew ** 2, axis=1)
        d2old = np.sum(np.array([]) ** 2)  # placeholder, not used
        # Recompute using stored X? We need K(Xnew, X). Let's store X separately.
        # Actually let's restructure: store X in self._X
        return None, None  # placeholder

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Initial structured sampling: Latin hypercube + corners
        rng = np.random.default_rng(0)
        X_list = []
        F_list = []

        def add(x, f):
            X_list.append(np.asarray(x, float))
            F_list.append(np.asarray(f, float))

        # Add corners and center
        corners = np.array([[lo[i] if c == i else hi[i] for i in range(dim)] for c in range(dim)])
        corners = np.vstack([lo, hi, (lo + hi) / 2])
        for x in corners:
            add(x, problem(x))

        # Latin hypercube
        n_init = min(30, budget - len(X_list))
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            add(x, problem(x))

        used = len(X_list)
        X_arr = np.array(X_list)
        F_arr = np.array(F_list)

        # GP parameters
        nugget = 1e-4

        def fit_gps():
            """Fit GP for each objective."""
            gps = []
            for k in range(n_obj):
                y = F_arr[:, k]
                var_y = np.var(y) + 1e-8
                n = len(y)
                d2 = np.sum(X_arr ** 2, axis=1)
                D2 = d2[:, None] + d2[None, :] - 2 * X_arr @ X_arr.T
                np.maximum(D2, 0, out=D2)
                K = var_y * np.exp(-D2 / (2 * 0.25 ** 2))
                K[np.diag_indices(n)] += nugget
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    K[np.diag_indices(n)] += 1e-6
                    L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                gps.append((K, L, alpha, var_y, X_arr, y))
            return gps

        def predict_gp(gp, Xnew):
            K, L, alpha, var_y, Xold, y = gp
            Xnew = np.asarray(Xnew, float)
            d2new = np.sum(Xnew ** 2, axis=1)
            d2old = np.sum(Xold ** 2, axis=1)
            D2 = d2new[:, None] + d2old[None, :] - 2 * Xnew @ Xold.T
            np.maximum(D2, 0, out=D2)
            Ks = var_y * np.exp(-D2 / (2 * 0.25 ** 2))
            v = np.linalg.solve(L, Ks.T)
            mu = v.T @ alpha
            var = var_y - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-8)
            return mu, np.sqrt(var)

        def hypervolume_2d(P, ref2):
            """Compute 2D hypervolume for minimization."""
            P = P[np.all(P < ref2, axis=1)]
            if len(P) == 0:
                return 0.0
            P = P[np.argsort(P[:, 0])]
            vol = 0.0
            prev_x = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref2[0]:
                    break
                w = min(x, ref2[0]) - prev_x
                if w <= 0:
                    continue
                y_best = np.min(P[i:, 1])
                y_best = min(y_best, ref2[1])
                if y_best < ref2[1]:
                    vol += w * (ref2[1] - y_best)
                prev_x = x
            return vol

        def hypervolume_3d(P, ref3):
            """Compute 3D hypervolume for minimization using 2D decomposition."""
            P = np.asarray(P, float)
            if len(P) == 0:
                return 0.0
            P = P[np.all(P < ref3, axis=1)]
            if len(P) == 0:
                return 0.0
            P = P[np.argsort(P[:, 0])]
            vol = 0.0
            prev_x = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref3[0]:
                    break
                w = min(x, ref3[0]) - prev_x
                if w <= 0:
                    continue
                sub = P[i:, 1:]
                vol += w * hypervolume_2d(sub, ref3[1:])
                prev_x = x
            return vol

        def ehvi_score(Xcand, gps, F_arr, ref):
            """Compute expected hypervolume improvement for candidate point(s)."""
            Xcand = np.asarray(Xcand, float)
            if Xcand.ndim == 1:
                Xcand = Xcand[None, :]
            n_cand = len(Xcand)
            # Predict at candidate
            mus = np.zeros((n_cand, n_obj))
            sigmas = np.zeros((n_cand, n_obj))
            for k in range(n_obj):
                mu, sigma = predict_gp(gps[k], Xcand)
                mus[:, k] = mu
                sigmas[:, k] = sigma

            # Compute current HV of F_arr
            current_hv = hypervolume_3d(F_arr, ref)

            # Sample-based EHVI estimate
            n_samples = 50
            ehvi = np.zeros(n_cand)
            for i in range(n_cand):
                # Sample from GP posterior
                F_samples = np.zeros((n_samples, n_obj))
                for k in range(n_obj):
                    F_samples[:, k] = mus[i, k] + sigmas[i, k] * rng.standard_normal(n_samples)
                # For each sample, compute HV including this point and existing Pareto front
                F_combined = np.vstack([F_arr, F_samples])
                # Find Pareto front
                n_p = len(F_combined)
                pareto = []
                for p in range(n_p):
                    dominated = np.any(np.all(F_combined <= F_combined[p], axis=1) & np.any(F_combined < F_combined[p], axis=1), axis=0)
                    if not dominated[p]:
                        pareto.append(p)
                F_pareto = F_combined[pareto]
                hv = hypervolume_3d(F_pareto, ref)
                ehvi[i] = max(0, hv - current_hv)
            return ehvi.mean(axis=0) if n_cand > 1 else ehvi[0]

        # Main loop
        while used < budget:
            if time.time() > deadline - 10.0:
                break

            # Fit GPs
            gps = fit_gps()

            # Generate candidates
            n_cand = 50
            X_cand = lo + (hi - lo) * rng.random((n_cand, dim))

            # Score candidates
            scores = np.array([ehvi_score(x, gps, F_arr, ref) for x in X_cand])
            best_idx = int(np.argmax(scores))
            x_next = X_cand[best_idx]
            f_next = problem(x_next)
            add(x_next, f_next)
            used += 1
            X_arr = np.array(X_list)
            F_arr = np.array(F_list)

        return X_list
