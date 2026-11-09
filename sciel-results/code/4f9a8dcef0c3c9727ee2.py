"""Multi-objective EGO optimizer for RE21 using per-objective GP surrogates
with expected-improvement acquisition, Pareto-elite initialization, and an
explicit wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import math
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
        # per-dimension length-scales
        d2 = d2 / np.outer(lengths, lengths)
        return var * np.exp(-0.5 * d2)

    @staticmethod
    def _fit_gp(X, y, dim):
        """Fit a simple GP (RBF + white noise). Returns (mean_fn, var_fn) closures."""
        n = X.shape[0]
        y = np.asarray(y, dtype=float).ravel()
        mu = y.mean()
        yc = y - mu
        # Normalize inputs to [0,1] for stable lengthscales
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        rng_ = np.where(hi > lo, hi - lo, 1.0)
        Xn = (X - lo) / rng_

        lengths = np.ones(dim) * 0.5
        var = 1.0
        noise = 1e-6 * (var + 1.0)

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
                # j dominates i?
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    # ---------- Acquisition: per-objective EI summed ----------
    def _acquire(self, predicts, F, candidate_pool):
        # best per objective (min)
        best = F.min(axis=0)
        scores = np.zeros(candidate_pool.shape[0])
        for k in range(self.n_obj):
            m, v = predicts[k](candidate_pool)
            sigma = np.sqrt(v)
            best_k = best[k]
            # expected improvement toward best (minimization): improvement = best - f
            # EI = E[max(0, best - f)] = (best - m)*Phi(z) + sigma*phi(z), z=(best-m)/sigma
            z = (best_k - m) / np.maximum(sigma, 1e-12)
            ei = (best_k - m) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            scores += ei
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

        # Deterministic RNG seeded by global state (evaluator sets np.random.seed)
        rng = np.random

        # ----- Initialization: Latin hypercube -----
        init_n = min(30, max(10, budget // 6))
        idx = np.arange(init_n)
        perm = [rng.permutation(init_n) for _ in range(dim)]
        u = np.array([[ (perm[d][i] + rng.random()) / init_n for i in range(init_n)] for d in range(dim)]).T
        X = lo + (hi - lo) * u
        F = np.array([problem(x) for x in X], dtype=float)
        used = init_n

        X_archive = [X.copy()]
        F_archive = [F.copy()]

        # Fit initial surrogates
        predicts = [self._fit_gp(X, F[:, k], dim) for k in range(n_obj)]

        while used < budget:
            if time.time() > deadline:
                break

            # Build candidate pool: dense grid + random + perturbations of elites
            cand = []
            # Grid (moderate resolution to keep speed)
            g = 8
            grid = np.array(np.meshgrid(*[np.linspace(lo[d], hi[d], g) for d in range(dim)])).T.reshape(-1, dim)
            cand.append(grid)
            # Random
            cand.append(lo + (hi - lo) * rng.random((200, dim)))
            # Perturb current nondominated points
            nd = self._nondominated_indices(F)
            elites = X[nd] if np.any(nd) else X
            if len(elites) > 0:
                scale = 0.15 * (hi - lo)
                cand.append(elites + scale * rng.standard_normal((len(elites), dim)))
            pool = np.vstack(cand)
            pool = np.clip(pool, lo, hi)

            scores = self._acquire(predicts, F, pool)
            best_idx = int(np.argmax(scores))
            x_new = pool[best_idx]

            # If we keep picking near-duplicates, nudge
            # (simple: just evaluate)
            f_new = np.asarray(problem(x_new), dtype=float)
            used += 1

            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])

            # Refit surrogates (lightweight; n is small)
            try:
                predicts = [self._fit_gp(X, F[:, k], dim) for k in range(n_obj)]
            except Exception:
                pass  # keep old surrogates

        # Return the full set of evaluated points (all recorded)
        return X
# EVOLVE-BLOCK-END
