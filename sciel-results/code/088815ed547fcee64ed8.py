"""Multi-objective optimizer on RE34: MOSA with hypervolume-driven acceptance + archive."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective Simulated Annealing with hypervolume-driven acceptance and an archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = 320.0  # safety margin before the 330 s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Only points strictly better than the reference in every objective count.
        def valid(f):
            return bool(np.all(f < ref))

        def hv_increment(archive, f):
            # Volume contributed by f, assuming f is non-dominated w.r.t. archive.
            # 3D orthogonal decomposition.
            if not valid(f):
                return 0.0
            a = np.array(archive, dtype=float)
            n = len(a)
            if n == 0:
                return float(np.prod(np.maximum(0.0, ref - f)))
            # For each objective dimension, find the nearest archive point that is
            # <= f in the other two objectives; the box is bounded by those.
            total = 0.0
            for d in range(3):
                others = [o for o in range(3) if o != d]
                # candidates that do not dominate f (i.e., are not better in all)
                # and are "adjacent" in dimension d
                cand = []
                for i in range(n):
                    ai = a[i]
                    # ai must not dominate f
                    if np.all(ai <= f) and np.any(ai < f):
                        continue
                    # ai must be <= f in the other two dims to form a box in dim d
                    if ai[others[0]] <= f[others[0]] and ai[others[1]] <= f[others[1]]:
                        cand.append(ai)
                if not cand:
                    bound_d = ref[d]
                else:
                    bound_d = min(cand[d] for cand in cand)
                w = bound_d - f[d]
                if w <= 0:
                    continue
                # box in the other two dims bounded by f (since f is the min in those among the box)
                w1 = ref[others[0]] - f[others[0]]
                w2 = ref[others[1]] - f[others[1]]
                if w1 > 0 and w2 > 0:
                    total += w * w1 * w2
            return total

        def is_dominated_by_archive(archive, f):
            a = np.array(archive, dtype=float)
            if len(a) == 0:
                return False
            return bool(np.any(np.all(a <= f) and np.any(a < f)))

        rng = np.random.default_rng(0)

        def rand_x():
            return lo + span * rng.random(dim)

        # ---- initialization: fill a small random archive ----
        X = rand_x()
        F = np.asarray(problem(X), dtype=float)
        used = 1
        archive = []
        if valid(F) and not is_dominated_by_archive(archive, F):
            archive.append(F.copy())

        # initial best HV
        base_hv = 0.0
        for af in archive:
            base_hv += hv_increment([x for x in archive if x is not af], af)

        T0 = 1.0
        T_end = 1e-4
        n_steps = max(1, budget - used)

        best_archive = list(archive)
        best_hv = base_hv

        while used < budget:
            if time.time() >= deadline:
                break

            # temperature schedule
            frac = (used - 1) / max(1, n_steps)
            T = T0 * (T_end / T0) ** frac

            # ---- propose a move ----
            # mix: gaussian step around current, and occasionally jump from archive
            r = rng.random()
            if r < 0.15 and len(archive) > 0:
                # restart-ish: pick an archive point and take a small step
                base = archive[int(rng.integers(len(archive)))]
                Y = base + 0.2 * span * rng.standard_normal(dim)
            else:
                step = 0.15 * span * rng.standard_normal(dim)
                Y = X + step
            Y = np.clip(Y, lo, hi)

            G = np.asarray(problem(Y), dtype=float)
            used += 1

            if not valid(G):
                # dominated by reference; still, keep moving from Y (it's a fresh point)
                X = Y
                continue

            # does G dominate any archive points?
            a = np.array(archive, dtype=float) if archive else np.empty((0, n_obj))
            if len(a) > 0:
                dom_mask = np.all(G <= a, axis=1) & np.any(G < a, axis=1)
            else:
                dom_mask = np.zeros(0, dtype=bool)
            dominated_any = bool(np.any(dom_mask))

            # does archive dominate G?
            g_dominated = is_dominated_by_archive(archive, G)

            # compute new archive if we add G
            if g_dominated:
                # G is dominated by archive -> no HV gain
                gain = 0.0
                new_archive = None
            else:
                new_archive = [af for af in archive if not (np.all(G <= af) and np.any(G < af))]
                gain = hv_increment(new_archive, G)

            # ---- acceptance ----
            accepted = False
            if dominated_any or gain > 1e-12:
                # improves the non-dominated set
                accepted = True
            elif rng.random() < T:
                # annealing: accept dominated move to escape plateau
                accepted = True

            if accepted:
                if not g_dominated:
                    # add G to archive, remove dominated
                    new_arch = [af for af in archive if not (np.all(G <= af) and np.any(G < af))]
                    new_arch.append(G.copy())
                    # recompute archive HV
                    hv = 0.0
                    for af in new_arch:
                        hv += hv_increment([x for x in new_arch if x is not af], af)
                    archive = new_arch
                    if hv > best_hv:
                        best_hv = hv
                        best_archive = list(archive)
                X = Y
            # else: reject, keep X (do not move)

            # periodic stagnation check: if no archive growth for a while, jump from archive
            if used % 25 == 0 and len(archive) > 0:
                # nudge current toward a random archive point to diversify
                base = archive[int(rng.integers(len(archive)))]
                X = 0.5 * X + 0.5 * base
                X = np.clip(X, lo, hi)

        return best_archive
# EVOLVE-BLOCK-END
