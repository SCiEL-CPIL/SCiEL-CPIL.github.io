"""CMA-ES optimizer for BBOB f6 (Attractive Sector).

Structural idea (model-and-optimize): a cheap quadratic model jump front-loads
early progress and seeds a good starting point. Refinement machinery (CMA-ES):
a proper, efficient covariance-matrix adaptation that tracks the unknown
rotation/scaling of the Attractive Sector. A (1+1) safety net polishes the
best point if the population plateaus.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model jump + efficient CMA-ES + (1+1) polish for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin for the 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # ---------------- Phase 1: structured sampling + quadratic model ----
        points = []
        values = []

        def record(x):
            nonlocal calls
            if check_deadline() or calls >= budget:
                return None
            fx = func(x)
            calls += 1
            points.append(x)
            values.append(fx)
            return fx

        record(np.zeros(dim))
        for i in range(dim):
            for s in (1.0, -1.0):
                x = np.zeros(dim)
                x[i] = 3.0 * s
                record(x)
        for _ in range(20):
            record(rng.uniform(-4.5, 4.5, dim))

        points_arr = np.array(points)
        values_arr = np.array(values)
        best_x = points_arr[np.argmin(values_arr)].copy()
        best_f = float(np.min(values_arr))

        # Fit separable quadratic f ~ c + b^T x + sum_i a_i x_i^2 and jump to it.
        if len(points_arr) >= dim + 2:
            X = np.zeros((len(points_arr), 1 + 2 * dim))
            X[:, 0] = 1.0
            X[:, 1:1 + dim] = points_arr
            X[:, 1 + dim:] = points_arr ** 2
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, values_arr, rcond=None)
                b_lin = coeffs[1:1 + dim]
                a_quad = coeffs[1 + dim:]
                x_model = np.zeros(dim)
                for i in range(dim):
                    if a_quad[i] > 1e-8:
                        x_model[i] = -b_lin[i] / (2.0 * a_quad[i])
                    else:
                        x_model[i] = -5.0 if b_lin[i] > 0 else 5.0
                x_model = np.clip(x_model, -5.0, 5.0)
                fm = record(x_model)
                if fm is not None and fm < best_f:
                    best_f = fm
                    best_x = x_model.copy()
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # ---------------- Phase 2: efficient CMA-ES -------------------------
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        c_sigma = (lam + 2) / (dim + lam + 5)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_cov_plus = (2 + 2 / (mueff + 1)) / (dim + 3)
        c_cov_min = 2 / (dim + dim / mueff + 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = best_x.copy()
        sigma = 2.0
        C = np.eye(dim)
        L = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        gen = 0
        best_in_window = best_f
        no_improve = 0

        while calls < budget and not check_deadline():
            z = rng.standard_normal((lam, dim))
            y = np.einsum('ij,ji->ij', z, L)          # cached Cholesky per gen
            pop = np.clip(x[None, :] + sigma * y, -5.0, 5.0)

            fvals = np.empty(lam)
            for i in range(lam):
                if check_deadline() or calls >= budget:
                    break
                fvals[i] = func(pop[i])
                calls += 1
            else:
                fvals = np.sort(fvals)
                order = np.argsort(fvals)
                y_sel = y[order[:mu]]
                pop_sel = pop[order[:mu]]

                if fvals[0] < best_f:
                    best_f = float(fvals[0])
                    best_x = pop[order[0]].copy()

                # weighted recombination
                x_new = (w[:, None] * pop_sel).sum(axis=0)
                x_new = np.clip(x_new, -5.0, 5.0)

                # evolution paths
                d_x = (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                    c_sigma * (2 - c_sigma) * mueff) * d_x
                norm_p_sigma = np.linalg.norm(p_sigma)
                gen += 1
                decay = (1 - c_sigma) ** (2 * gen)
                h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - decay) < (
                    1.4 + 2 / (dim + 1)) * chi_n * np.sqrt(1 - decay) else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(
                    c_c * (2 - c_c) * mueff) * d_x

                # covariance update (rank-1 + rank-mu)
                rank_mu = np.einsum('ij,ik,jk->jk', y_sel, y_sel, w)
                C = (1 + c_cov_plus * h_sigma * (c_c + 2 - c_c) - c_cov_min) * C \
                    + c_cov_plus * np.outer(p_c, p_c) + c_cov_min * rank_mu

                # step size (CSA)
                sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
                sigma = float(np.clip(sigma, 1e-12, 10.0))

                x = x_new

                # refresh cached Cholesky (guard against non-PSD)
                for _ in range(5):
                    try:
                        L = np.linalg.cholesky(C)
                        break
                    except np.linalg.LinAlgError:
                        C = 0.5 * C + 0.5 * np.eye(dim)

                # stagnation tracking
                if best_f < best_in_window - 1e-12:
                    best_in_window = best_f
                    no_improve = 0
                else:
                    no_improve += 1

                if sigma < 1e-9 or no_improve >= 12:
                    break

        # ---------------- Phase 3: (1+1) safety-net polish ------------------
        if not check_deadline() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            successes = 0
            total = 0
            while not check_deadline() and calls < budget:
                cand = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fc = func(cand)
                calls += 1
                if fc < fx - 1e-12:
                    x = cand
                    fx = fc
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                total += 1
                if total >= 5:
                    if successes / total > 0.2:
                        sigma *= 1.25
                    elif successes / total < 0.2:
                        sigma *= 0.8
                    successes = 0
                    total = 0
                if sigma < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
