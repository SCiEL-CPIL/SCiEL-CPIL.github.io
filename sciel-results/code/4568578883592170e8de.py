"""Multi-objective EGO optimizer for RE21: GP surrogates + EI acquisition,
adaptive step-size, stagnation detection, and a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import stats


class Optimizer:
    """Lightweight multi-objective GP-based EGO (minimizing all objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---------- GP helpers ----------
    @staticmethod
    def _rbf_kernel(A, B, lengths, var):
        # A: (n, d), B: (m, d)
        d2 = (
            np.sum(A * A, axis=1)[:, None]
            + np.sum(B * B, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )
        d2 = np.maximum(d2, 0.0)
        d2 = d2 / np.outer(lengths, lengths)
        return var * np.exp(-0.5 * d2)

    @staticmethod
    def _fit_gp(X, y, dim):
        """Fit a GP (RBF + white noise) with simple hyperparameter optimization."""
        n = X.shape[0]
        y = np.asarray(y, dtype=float).ravel()
        mu = y.mean()
        yc = y - mu
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        rng_ = np.where(hi > lo, hi - lo, 1.0)
        Xn = (X - lo) / rng_

        # Simple hyperparameter optimization: try a few length-scale values
        best_loglik = -np.inf
        best_lengths = np.ones(dim) * 0.5
        best_var = 1.0
        best_noise = 1e-4

        candidate_lengths = [0.1, 0.3, 0.5, 1.0, 2.0]
        candidate_vars = [0.5, 1.0, 2.0]
        candidate_noises = [1e-5, 1e-4, 1e-3]

        for ls in candidate_lengths:
            lengths = np.ones(dim) * ls
            for var in candidate_vars:
                for noise in candidate_noises:
                    K = Optimizer._rbf_kernel(Xn, Xn, lengths, var) + noise * np.eye(n)
                    try:
                        L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
                    except np.linalg.LinAlgError:
                        try:
                            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
                        except np.linalg.LinAlgError:
                            continue
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
                    # Log-likelihood
                    logdet = 2.0 * np.sum(np.log(np.diag(L)))
                    loglik = -0.5 * (n * np.log(2 * np.pi) + logdet + yc @ alpha)
                    if loglik > best_loglik:
                        best_loglik = loglik
                        best_lengths = lengths.copy()
                        best_var = var
                        best_noise = noise
                        best_L = L.copy()
                        best_alpha = alpha.copy()

        lengths = best_lengths
        var = best_var
        noise = best_noise
        K = Optimizer._rbf_kernel(Xn, Xn, lengths, var) + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))

        def predict(Xq):
            Xqn = (Xq - lo) / rng_
            Ks = Optimizer._rbf_kernel(Xqn, Xn, lengths, var)
            m = mu + Ks @ alpha
            v = np.linalg.solve(L, Ks.T)
            varr = var - np.sum(v * v, axis=1)
            varr = np.maximum(varr, 1e-12)
            return m, varr

        return predict

    # ---------- Dominance ----------
    @staticmethod
    def _nondominated_indices(F):
        """Return boolean mask of nondominated rows (minimization)."""
        n = F.shape[0]
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    # ---------- Acquisition: hypervolume-aware EI ----------
    def _acquire(self, predicts, F, candidate_pool, ref_point):
        """Approximate hypervolume contribution via per-objective EI weighted by
        the volume of the region where this objective is the bottleneck."""
        n_cand = candidate_pool.shape[0]
        scores = np.zeros(n_cand)
        best = F.min(axis=0)

        # Compute per-objective EI
        ei_per_obj = np.zeros((n_cand, self.n_obj))
        for k in range(self.n_obj):
            m, v = predicts[k](candidate_pool)
            sigma = np.sqrt(v)
            best_k = best[k]
            z = (best_k - m) / np.maximum(sigma, 1e-12)
            ei_k = (best_k - m) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            ei_per_obj[:, k] = np.maximum(ei_k, 0.0)

        # Weight by reference point distance (larger distance = more important)
        ref_dist = np.maximum(ref_point - best, 0.0)
        weights = ref_dist / (ref_dist.sum() + 1e-12)

        # Weighted sum of EI
        scores = ei_per_obj @ weights

        # Bonus for points that are likely to be non-dominated (lower predicted values)
        pred_mean = np.zeros((n_cand, self.n_obj))
        for k in range(self.n_obj):
            pred_mean[:, k] = predicts[k](candidate_pool)[0]
        # Penalize dominated predictions
        for i in range(n_cand):
            for j in range(n_cand):
                if i != j and np.all(pred_mean[j] <= pred_mean[i]) and np.any(pred_mean[j] < pred_mean[i]):
                    scores[i] *= 0.5
                    break

        return scores

    # ---------- Main ----------
    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref_point = np.array([3000.0, 0.06])

        # Deterministic RNG seeded by global state (evaluator sets np.random.seed)
        rng = np.random

        # ----- Initialization: diverse (Latin-hypercube-ish) + boundary focus -----
        init_n = min(30, max(12, budget // 6))
        idx = np.arange(init_n)
        perm = [rng.permutation(init_n) for _ in range(dim)]
        u = np.array([
            [(perm[d][i] + rng.random()) / init_n for i in range(init_n)]
            for d in range(dim)
        ]).T
        X = lo + (hi - lo) * u

        # Add boundary-focused points (corners/edges) for early coverage
        extra = []
        for corner in range(2 ** dim):
            pt = np.array([(hi[d] if (corner >> d) & 1 else lo[d]) for d in range(dim)])
            extra.append(pt)
        X = np.vstack([X, np.array(extra)])
        F = np.array([problem(x) for x in X], dtype=float)
        used = X.shape[0]

        # Fit initial surrogates
        predicts = [self._fit_gp(X, F[:, k], dim) for k in range(n_obj)]

        # Stagnation / adaptive-step tracking
        best_sum = float(np.sum(F.min(axis=0)))
        last_improve = 0
        sigma_scale = 0.15

        while used < budget:
            if time.time() > deadline:
                break

            # Build candidate pool: random + perturbations of elites
            nd = self._nondominated_indices(F)
            elites = X[nd] if np.any(nd) else X

            # Random candidates
            n_rand = 150
            cand = [lo + (hi - lo) * rng.random((n_rand, dim))]

            # Perturbations of elites with adaptive scale
            if len(elites) > 0:
                scale = sigma_scale * (hi - lo)
                n_pert = min(len(elites), 50)
                pert = elites + scale * rng.standard_normal((n_pert, dim))
                cand.append(pert)

            pool = np.vstack(cand)
            pool = np.clip(pool, lo, hi)

            # Use hypervolume-aware acquisition
            scores = self._acquire(predicts, F, pool, ref_point)
            best_idx = int(np.argmax(scores))
            x_new = pool[best_idx]

            f_new = np.asarray(problem(x_new), dtype=float)
            used += 1

            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])

            # Stagnation detection
            cur_best = float(np.sum(F.min(axis=0)))
            if cur_best < best_sum - 1e-9:
                best_sum = cur_best
                last_improve = used
                sigma_scale = max(0.05, sigma_scale * 0.9)
            elif used - last_improve > 15:
                sigma_scale = min(0.4, sigma_scale * 1.25)

            # Refit surrogates
            try:
                predicts = [self._fit_gp(X, F[:, k], dim) for k in range(n_obj)]
            except Exception:
                pass

        return X
# EVOLVE-BLOCK-END
