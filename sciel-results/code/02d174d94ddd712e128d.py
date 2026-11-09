# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 322.0
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # ---------- helpers ----------
        def dominates(a, b):
            # a dominates b if a<=b all and a<b any
            return bool(np.all(a <= b) and np.any(a < b))

        def non_dominated_mask(F):
            n = len(F)
            nd = np.zeros(n, dtype=bool)
            for i in range(n):
                if nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        nd[i] = True
                        break
            return nd

        def fast_nondominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            frontiers = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    frontiers.append(i)
            k = 0
            result = np.empty(n, dtype=int)
            while len(frontiers) > 0:
                next_front = []
                for i in frontiers:
                    result[i] = k
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                frontiers = next_front
                k += 1
            return result

        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n, dtype=np.float64)
            for rank in set(ranks):
                idx = np.where(ranks == rank)[0]
                m = len(idx)
                if m <= 2:
                    cd[idx] = np.inf
                    continue
                for d in range(n_obj):
                    vals = F[idx, d]
                    vmin = vals.min()
                    vmax = vals.max()
                    rng = vmax - vmin
                    if rng < 1e-12:
                        continue
                    order = np.argsort(vals)
                    cd[idx[order[0]]] = np.inf
                    cd[idx[order[-1]]] = np.inf
                    for kk in range(1, m - 1):
                        cd[idx[order[kk]]] += (vals[order[kk + 1]] - vals[order[kk - 1]]) / rng
            return cd

        # ---------- structured initial population ----------
        rng = np.random.default_rng(0)

        # Grid-like structured initialization covering the space
        n_init = min(40, budget)
        # Use Latin-hypercube-like structured init
        X_init = np.empty((n_init, dim))
        for i in range(n_init):
            for d in range(dim):
                X_init[i, d] = lo[d] + (hi[d] - lo[d]) * ((i + 0.5) / n_init)
        # Add jitter
        X_init += 0.05 * span * rng.random((n_init, dim))
        X_init = np.clip(X_init, lo, hi)

        # Also add some random points for diversity
        n_rand = min(10, budget - n_init)
        if n_rand > 0:
            X_rand = lo + span * rng.random((n_rand, dim))
            X_init = np.vstack([X_init, X_rand])
            n_init = len(X_init)

        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        # Archive of non-dominated points
        archive_X = list(X_init)
        archive_F = list(F_init)

        def update_archive(ax, af):
            nonlocal archive_X, archive_F
            # Check if new point is non-dominated by any existing
            for j in range(len(archive_F)):
                if dominates(archive_F[j], af):
                    return
            # Remove any existing dominated by new
            new_ax = list(archive_X)
            new_af = list(archive_F)
            to_remove = []
            for j in range(len(new_af)):
                if dominates(af, new_af[j]):
                    to_remove.append(j)
            for j in sorted(to_remove, reverse=True):
                del new_ax[j]
                del new_af[j]
            new_ax.append(ax)
            new_af.append(af)
            archive_X = new_ax
            archive_F = new_af

        for i in range(len(X_init)):
            update_archive(X_init[i], F_init[i])

        # ---------- Main DE-based loop ----------
        pop_size = min(20, max(6, budget // 10))

        # Initialize DE population from best archive + random
        pop_X = np.empty((pop_size, dim))
        if len(archive_X) >= pop_size:
            # Sample from archive
            idx = rng.choice(len(archive_X), size=pop_size, replace=False)
            for i, j in enumerate(idx):
                pop_X[i] = archive_X[j]
        else:
            for i in range(min(pop_size, len(archive_X))):
                pop_X[i] = archive_X[i]
            for i in range(len(archive_X), pop_size):
                pop_X[i] = lo + span * rng.random(dim)
        pop_F = np.array([problem(x) for x in pop_X])
        used += pop_size

        F_archive = np.array(archive_F)

        # DE parameters
        CR = 0.8
        F_mut = 0.5

        while used < budget:
            if time.time() > deadline - 5:
                break

            # Ensure population is diverse: if too many dominated, replace
            ranks = fast_nondominated_sort(pop_F)
            cd = crowding_distance(pop_F, ranks)

            # Select parents via tournament
            n_children = 2
            for _ in range(n_children):
                if time.time() > deadline - 3:
                    break

                # Tournament selection
                cand_idx = rng.choice(pop_size, size=3, replace=False)
                cand_ranks = ranks[cand_idx]
                winner = cand_idx[np.argmin(cand_ranks)]
                parent = pop_X[winner]
                vpos = F[winner] if hasattr(self, '_vpos') else pop_F[winner]
                vpos = pop_F[winner]

                # DE mutation: x_r1 + F*(x_r2 - x_r3)
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                v = pop_X[r1] + F_mut * (pop_X[r2] - pop_X[r3])
                # Crossover
                mask = rng.random(dim) < CR
                if not np.any(mask):
                    mask[rng.integers(dim)] = True
                child = np.where(mask, v, parent)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update archive
                update_archive(child, fc)

                # Replace worst in population
                worst_idx = np.argmax(ranks)
                pop_X[worst_idx] = child
                pop_F[worst_idx] = fc

                # Update ranks
                ranks = fast_nondominated_sort(pop_F)

        # Final: ensure we use remaining budget if possible
        while used < budget:
            if time.time() > deadline - 2:
                break
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            v = pop_X[r1] + F_mut * (pop_X[r2] - pop_X[r3])
            parent_idx = rng.integers(pop_size)
            mask = rng.random(dim) < CR
            if not np.any(mask):
                mask[rng.integers(dim)] = True
            child = np.where(mask, v, pop_X[parent_idx])
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            update_archive(child, fc)

        return pop_X
# EVOLVE-BLOCK-END
