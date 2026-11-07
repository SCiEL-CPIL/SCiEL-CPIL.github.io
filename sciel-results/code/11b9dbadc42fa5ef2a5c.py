"""Multi-objective optimizer on WFG4: vectorized NSGA-II + deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Fast vectorized NSGA-II multi-objective search with a wall-clock deadline."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    # ---- vectorized helpers -------------------------------------------------
    @staticmethod
    def _dominates(A, B):
        """A[i] dominates B[j] (strictly better in at least one, no worse in any)."""
        le = A[:, None, :] <= B[None, :, :]
        lt = A[:, None, :] < B[None, :, :]
        return np.all(le, axis=2) & np.any(lt, axis=2)

    def _non_dominated(self, F):
        D = self._dominates(F, F)
        np.fill_diagonal(D, False)
        return ~D.any(axis=1)

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        rank = np.full(n, -1, dtype=int)
        i = 0
        while i < n:
            Fc = F[rank == -1]
            mask = self._non_dominated(Fc)
            idx = np.where(rank == -1)[0][mask]
            rank[idx] = i
            i += 1
        return rank

    def _crowding(self, F, idx):
        """Crowding distance for the subset of points at indices idx."""
        m = len(idx)
        cd = np.zeros(m)
        if m <= 2:
            cd[:] = np.inf
            return cd
        nobj = F.shape[1]
        for k in range(nobj):
            vals = F[idx, k]
            order = np.argsort(vals)
            lo, hi = vals[order[0]], vals[order[-1]]
            if hi <= lo:
                continue
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if m > 2:
                cd[order[1:-1]] += (vals[order[2:]] - vals[order[:-2]]) / (hi - lo)
        return cd

    # ---- main search --------------------------------------------------------
    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random.default_rng(0)

        def evals(X):
            return np.array([problem(x) for x in X], dtype=float)

        def rnd(n):
            return lo + rng.random((n, dim)) * span

        # --- cheap exploration to seed the archive (WFG4 is multi-modal) -----
        n_init = 40
        X, F = rnd(n_init), evals(rnd(n_init))
        used = 2 * n_init

        # --- NSGA-II main loop ------------------------------------------------
        while used < self.budget:
            if time.time() >= deadline:
                break
            N = max(16, min(40, (self.budget - used) // 4, 48))
            if N < 1:
                break
            rank = self._fast_non_dominated_sort(F)
            cd = np.zeros(len(F))
            for r in np.unique(rank):
                idx = np.where(rank == r)[0]
                cd[idx] = self._crowding(F, idx)
            score = rank.astype(float) - cd

            # binary tournament
            def pick():
                a, b = rng.integers(len(F), size=2)
                return a if score[a] <= score[b] else b
            pa = np.array([pick() for _ in range(N)])
            pb = np.array([pick() for _ in range(N)])
            mask = rng.random(N) < 0.5
            A = F[pa].copy()
            B = F[pb].copy()
            A[:, mask], B[:, mask] = B[:, mask], A[:, mask]
            child = A + 0.1 * span * rng.standard_normal((N, dim))
            child = np.clip(child, lo, hi)
            Cf = evals(child)
            used += N

            # combine parents + children, keep best N
            Xc = np.vstack([F, Cf])
            rankc = self._fast_non_dominated_sort(Xc)
            cd = np.zeros(len(Xc))
            for r in np.unique(rankc):
                idx = np.where(rankc == r)[0]
                cd[idx] = self._crowding(Xc, idx)
            sc = rankc.astype(float) - cd
            keep = np.argsort(sc)[:N]
            F = Xc[keep]
            if time.time() >= deadline:
                break

        # --- return a valid non-dominated archive ----------------------------
        F = F[np.all(F < np.array([3.0, 5.0, 7.0]), axis=1)]
        if F.size == 0:
            return np.empty((0, n_obj))
        return F[self._non_dominated(F)]
# EVOLVE-BLOCK-END
