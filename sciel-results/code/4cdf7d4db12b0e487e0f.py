import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh


def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under the 150 s kill

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def fast_swap_delta(p, i, j):
        """Exact cost change from swapping the locations held at i and j."""
        pu = p[i]
        pv = p[j]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[i, :] - a[j, :], d_uv[p])
        s2 = np.dot(a[:, i] - a[:, j], d_vu[p])

        # remove the k==i and k==j terms that were included in the full dots
        s1 -= (a[i, i] - a[j, i]) * d_uv[pu]
        s1 -= (a[i, j] - a[j, j]) * d_uv[pv]

        s2 -= (a[i, i] - a[i, j]) * d_vu[pu]
        s2 -= (a[j, i] - a[j, j]) * d_vu[pv]

        cross = (a[i, j] - a[j, i]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=40):
        """First-improvement 2-opt. Returns (p, cost) at a 2-opt local optimum."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        """Linearize the quadratic objective for a fixed partner permutation and
        re-solve the resulting assignment problem (Hungarian). This is a strong
        descent step that can move far from p, escaping 2-opt local optima."""
        Bp = b[np.ix_(np.arange(n), p)]      # Bp[k, i] = b[k, p[i]]
        M1 = a @ Bp.T                        # M1[i, k] = sum_j a[i,j] b[k, p[j]]
        Bp2 = b[np.ix_(p, np.arange(n))]     # Bp2[i, k] = b[p[i], k]
        M2 = a.T @ Bp2                       # M2[i, k] = sum_j a[j,i] b[p[j], k]
        C = (M1 + M2).astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def rank_init():
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def spectral_init():
        """Hierarchical spectral bisection: recursively split facilities by their
        flow graph and locations by their distance graph, then assign the halves.
        Produces a structurally different starting point than rank_init."""
        all_facs = np.arange(n)
        all_locs = np.arange(n)
        assign = {}

        def fiedler(matrix, size):
            try:
                D = np.diag(matrix.sum(axis=1))
                L = D - matrix
                L = 0.5 * (L + L.T)
                vals, vecs = eigh(L)
                return vecs[:, 1]
            except Exception:
                return np.random.randn(size)

        def rec(facs, locs):
            if len(facs) == 0:
                return
            if len(facs) == 1:
                assign[facs[0]] = locs[0]
                return
            if len(facs) <= 2:
                for i, f in enumerate(facs):
                    assign[f] = locs[i]
                return
            sub_a = a[np.ix_(facs, facs)]
            sub_b = b[np.ix_(locs, locs)]
            fp = fiedler(sub_a, len(facs))
            lp = fiedler(-sub_b, len(locs))
            f1 = facs[fp > 0]
            f2 = facs[fp <= 0]
            l1 = locs[lp > 0]
            l2 = locs[lp <= 0]
            if len(f1) == 0 or len(f2) == 0 or len(l1) == 0 or len(l2) == 0:
                for i, f in enumerate(facs):
                    assign[f] = locs[i]
                return
            # balance the two halves
            if len(f1) > len(l1):
                m = len(f1) - len(l1)
                f2 = np.concatenate([f1[-m:], f2])
                f1 = f1[:-m]
            elif len(f2) > len(l2):
                m = len(f2) - len(l2)
                f1 = np.concatenate([f1, f2[:m]])
                f2 = f2[m:]
            rec(f1, l1)
            rec(f2, l2)

        rec(all_facs, all_locs)
        p = np.empty(n, dtype=np.int64)
        for f, l in assign.items():
            p[f] = l
        used = set(assign.values())
        missing = [l for l in range(n) if l not in used]
        for f in range(n):
            if f not in assign:
                p[f] = missing.pop(0)
        return p

    def perturb(p):
        q = p.copy()
        rng = np.random.default_rng()
        for _ in range(n // 2):
            i = rng.integers(n)
            j = rng.integers(n)
            q[i], q[j] = q[j], q[i]
        return q

    # --- build initial candidate solutions ---
    inits = [rank_init()]
    try:
        if time.time() < deadline - 3.0:
            inits.append(spectral_init())
    except Exception:
        pass

    best_cost = None
    best_p = None

    for p0 in inits:
        if time.time() > deadline - 5.0:
            break
        p = p0.copy()
        cost = compute_cost(p)
        p, cost = local_search(p, cost)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    # --- iterated local search: Hungarian re-optimization + perturbation ---
    while time.time() < deadline - 5.0:
        if best_p is None:
            break
        q = hungarian_reopt(best_p)
        if time.time() > deadline - 5.0:
            break
        q, qc = local_search(q, compute_cost(q))
        if qc < best_cost:
            best_cost = qc
            best_p = q.copy()
        else:
            q = perturb(best_p)
            if time.time() > deadline - 5.0:
                break
            q, qc = local_search(q, compute_cost(q))
            if qc < best_cost:
                best_cost = qc
                best_p = q.copy()

    if best_p is None:
        best_p = rank_init()
    return best_p.tolist()
