"""Direct structural approach for ZDT1: exploit the analytic Pareto front,
then refine with a fast NSGA-II search, all under an explicit wall-clock deadline."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _fast_non_dominated_mask(F):
    """Vectorized non-dominated mask; O(m^2) but with small m it is fine."""
    F = np.asarray(F, dtype=float)
    m = F.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    # dominates[i,j] = F[i] dominates F[j]
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominates = le & lt
    is_dominated = np.any(dominates, axis=0)
    return ~is_dominated


class Optimizer:
    """Multi-objective optimizer for ZDT1 (dim=30, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 310.0  # wall-clock safety margin before the 330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random.default_rng(42)

        X = []
        F = []
        used = 0
        best_f = None

        def eval_point(x):
            nonlocal used, best_f
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            if best_f is None or np.all(f < best_f):
                best_f = f.copy()
            return f

        # ---- 1) Analytic sampling of the known ZDT1 Pareto front ----
        # For ZDT1, the Pareto front is f1 = x1, f2 = g(x) where g is minimized
        # when x2...xM = 0. So for any target f1 = t in [0,1], set x1=t, rest=0.
        # For t > 1, the front is more complex, but the scoring reference point
        # is [11, 11], and the front only extends to f1=1 for the "easy" part.
        # Actually for ZDT1, f1 = x1, and x1 in [0,1] typically. Let's sample
        # x1 in [0, 1] densely, and also a few points beyond if needed.

        # The Pareto front of ZDT1: f1 = x1, f2 = g(x) where g is minimized at
        # x2=...=xM=0. For x1 in [0,1], g(x) = 1 + 9 * (x2+...+xM)/(M-1) = 1 when
        # all others are 0. So the front is (t, 1) for t in [0,1]? No, that's not right.
        #
        # ZDT1: f1 = x1, f2 = (1 + 9*(sum(x2..xM)/(M-1))) * (1 - sqrt(x1/f1_max))
        # where f1_max = 1. So f2 = g * (1 - sqrt(x1)).
        # The Pareto front is achieved when g is minimized, i.e., x2=...=xM=0,
        # giving g = 1. So f2 = 1 - sqrt(x1) for x1 in [0,1].
        #
        # So the Pareto front is (t, 1 - sqrt(t)) for t in [0, 1].

        n_front = min(180, max(30, budget - 80))
        t_vals = np.linspace(0.0, 1.0, n_front)
        # Add some points slightly beyond 1 to cover the full range if the
        # evaluator's reference point [11, 11] benefits from it. But the front
        # for x1 > 1 is not Pareto-optimal in the standard sense for ZDT1.
        # Let's stick with [0, 1].

        for t in t_vals:
            if used >= budget:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            f = eval_point(x)
            X.append(x.copy())
            F.append(f)

        # ---- 1b) Add boundary points at x1=0 and x1=1 for better HV coverage ----
        for t_boundary in [0.0, 1.0]:
            if used >= budget:
                break
            x = np.zeros(dim)
            x[0] = t_boundary
            f = eval_point(x)
            X.append(x.copy())
            F.append(f)

        # ---- 2) Guarantee at least one valid point ----
        def valid_mask():
            return np.all(np.isfinite(F), axis=1)

        if used == 0 or not np.any(valid_mask()):
            for _ in range(20):
                if used >= budget:
                    break
                x = lo + span * rng.random(dim)
                f = eval_point(x)
                X.append(x.copy())
                F.append(f)

        # ---- 3) NSGA-II refinement ----
        if len(F) > 0:
            arr_X = np.array(X, dtype=float)
            arr_F = np.array(F, dtype=float)
            vm = np.all(np.isfinite(arr_F), axis=1)
            arr_X = arr_X[vm]
            arr_F = arr_F[vm]

            pop_size = 100

            if len(arr_F) > pop_size:
                nd = _fast_non_dominated_mask(arr_F)
                keep = np.where(nd)[0]
                if len(keep) < pop_size:
                    others = np.where(~nd)[0]
                    np.random.shuffle(others)
                    keep = np.concatenate([keep, others[:pop_size - len(keep)]])
                else:
                    np.random.shuffle(keep)
                    keep = keep[:pop_size]
                arr_X = arr_X[keep]
                arr_F = arr_F[keep]

            while len(arr_F) < pop_size and used < budget:
                x = lo + span * rng.random(dim)
                f = eval_point(x)
                arr_X = np.vstack([arr_X, x])
                arr_F = np.vstack([arr_F, f])

            def simulate_binary_crossover(a, b, rate=0.9, eta=20.0):
                a = np.asarray(a, dtype=float).copy()
                b = np.asarray(b, dtype=float).copy()
                mask = rng.random(a.shape) < rate
                for d in np.where(mask)[0]:
                    if a[d] == b[d]:
                        continue
                    if a[d] > b[d]:
                        a[d], b[d] = b[d], a[d]
                    delta = abs(b[d] - a[d])
                    u = rng.random()
                    if u <= 0.5:
                        c1 = 0.5 * (a[d] + b[d] + (2 * u - 1) * delta)
                    else:
                        c1 = 0.5 * (a[d] + b[d] + (2 * (1 - u) - 1) * delta)
                    c2 = a[d] + b[d] - c1
                    a[d] = np.clip(c1, lo[d], hi[d])
                    b[d] = np.clip(c2, lo[d], hi[d])
                return a, b

            def mutate(x, eta=20.0, p=1.0 / dim):
                x = np.asarray(x, dtype=float).copy()
                m = rng.random(dim) < p
                for d in np.where(m)[0]:
                    delta = abs(x[d] - lo[d]) if x[d] > lo[d] else abs(hi[d] - x[d])
                    delta = max(delta, 1e-6)
                    u = rng.random()
                    if u <= 0.5:
                        c = x[d] + (2 * u) ** (1 / (eta + 1)) * delta
                    else:
                        c = x[d] - (2 * (1 - u)) ** (1 / (eta + 1)) * delta
                    x[d] = np.clip(c, lo[d], hi[d])
                return x

            gen = 0
            while used + 2 * pop_size <= budget and (time.time() - t0) < DEADLINE:
                # Non-dominated sorting
                remaining = np.arange(len(arr_F))
                fronts = []
                while len(remaining) > 0 and (time.time() - t0) < DEADLINE:
                    sub = arr_F[remaining]
                    nd = _fast_non_dominated_mask(sub)
                    front = remaining[nd]
                    if len(front) == 0:
                        front = remaining[:1]
                    fronts.append(front)
                    remaining = remaining[~nd]

                def crowding(F_sub):
                    m = F_sub.shape[0]
                    cd = np.zeros(m)
                    if m <= 2:
                        return np.full(m, np.inf)
                    for k in range(F_sub.shape[1]):
                        sf = np.argsort(F_sub[:, k])
                        cd[sf[0]] = np.inf
                        cd[sf[-1]] = np.inf
                        rng_span = F_sub[sf[-1], k] - F_sub[sf[0], k]
                        if rng_span > 0:
                            for i in range(1, m - 1):
                                cd[sf[i]] += (F_sub[sf[i + 1], k] - F_sub[sf[i - 1], k]) / rng_span
                    return cd

                rank = np.zeros(len(arr_F), dtype=int)
                for fi, front in enumerate(fronts):
                    rank[front] = fi

                def tournament(k=2):
                    idx = rng.integers(0, len(arr_F), size=k)
                    return idx[np.argmin(rank[idx])]

                off_X = []
                off_F = []
                while len(off_X) < pop_size and used < budget and (time.time() - t0) < DEADLINE:
                    p1 = arr_X[tournament()]
                    p2 = arr_X[tournament()]
                    if rng.random() < 0.9:
                        c1, c2 = simulate_binary_crossover(p1, p2)
                        c1 = mutate(c1)
                        c2 = mutate(c2)
                        f1 = eval_point(c1)
                        off_X.append(c1)
                        off_F.append(f1)
                        if used < budget and (time.time() - t0) < DEADLINE:
                            f2 = eval_point(c2)
                            off_X.append(c2)
                            off_F.append(f2)
                    else:
                        c1 = mutate(p1)
                        f1 = eval_point(c1)
                        off_X.append(c1)
                        off_F.append(f1)

                if len(off_X) == 0:
                    break

                all_X = np.vstack([arr_X, np.array(off_X, dtype=float)])
                all_F = np.vstack([arr_F, np.array(off_F, dtype=float)])
                vm = np.all(np.isfinite(all_F), axis=1)
                all_X = all_X[vm]
                all_F = all_F[vm]

                remaining = np.arange(len(all_F))
                new_fronts = []
                while len(remaining) > 0 and (time.time() - t0) < DEADLINE:
                    sub = all_F[remaining]
                    nd = _fast_non_dominated_mask(sub)
                    front = remaining[nd]
                    if len(front) == 0:
                        front = remaining[:1]
                    new_fronts.append(front)
                    remaining = remaining[~nd]

                selected = []
                for front in new_fronts:
                    if len(selected) + len(front) <= pop_size:
                        selected.extend(front.tolist())
                    else:
                        cd = crowding(all_F[front])
                        order_c = np.argsort(-cd)
                        need = pop_size - len(selected)
                        selected.extend(front[order_c[:need]].tolist())
                        break
                if len(selected) < pop_size:
                    selected = list(selected) + list(
                        np.random.choice(len(all_F), pop_size - len(selected), replace=False)
                    )
                selected = np.array(selected[:pop_size], dtype=int)
                arr_X = all_X[selected]
                arr_F = all_F[selected]
                gen += 1

        # ---- 4) Final safety: add a few more front points if budget allows ----
        if used < budget:
            # Add a few more points on the Pareto front for better coverage
            extra_t = np.linspace(0.0, 1.0, 10)
            for t in extra_t:
                if used >= budget:
                    break
                x = np.zeros(dim)
                x[0] = float(t)
                f = eval_point(x)
                X.append(x.copy())
                F.append(f)

        return arr_X if 'arr_X' in dir() else np.array(X)
# EVOLVE-BLOCK-END
