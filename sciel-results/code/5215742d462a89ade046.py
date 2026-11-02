"""Kissing number in dimension 11: MIS on conflict graph via SA."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Generate all weight-4 vectors with entries in {-1,0,1} (norm^2 = 4)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    # Generate weight-1 vectors with entries in {-2, 0, 2} (norm^2 = 4)
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = 2 * sign
            v1_list.append(v)

    vectors_v4 = np.array(v4_list, dtype=np.int64)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Build conflict graph: two W4 vectors conflict if |dot| > 2
    # (normalized to norm 2, we need |dot|/4 <= 0.5 => |dot| <= 2)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = (dots > 2) | (dots < -2)
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists and degrees
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(12345)

    def simulate_annealing(init_x, time_budget):
        """SA with soft constraint: minimize E = -|S| + lambda * (#conflicts)"""
        x = init_x.copy()
        # conflict_count[i] = number of selected neighbors of i
        ccount = np.zeros(N4, dtype=np.int32)
        for idx in np.where(x)[0]:
            ccount[neighbors[idx]] += 1

        lam = 1.5
        size = int(x.sum())
        nconf = int(np.sum(ccount)) // 2
        E = -size + lam * nconf

        best_E = E
        best_x = x.copy()

        T = 5.0
        T_min = 0.005
        cool_rate = 0.997
        end = time.time() + time_budget

        while time.time() < end:
            i = int(rng.integers(N4))
            if x[i]:
                dE = 1.0 - lam * ccount[i]
            else:
                dE = -1.0 + lam * ccount[i]

            if dE < 0 or rng.random() < np.exp(-dE / T):
                x[i] = not x[i]
                nbrs = neighbors[i]
                if x[i]:
                    ccount[nbrs] += 1
                else:
                    ccount[nbrs] -= 1
                E += dE
                if E < best_E:
                    best_E = E
                    best_x = x.copy()
            T *= cool_rate
            if T < T_min:
                T = T_min

        return best_x

    def greedy_independent_set():
        """Greedy min-degree heuristic for initial independent set."""
        selected = np.zeros(N4, dtype=bool)
        remaining = np.ones(N4, dtype=bool)
        deg = degrees.copy()

        while remaining.any():
            rem_idx = np.where(remaining)[0]
            if len(rem_idx) == 0:
                break
            min_d = deg[rem_idx].min()
            cands = rem_idx[deg[rem_idx] == min_d]
            i = cands[int(rng.integers(len(cands)))]
            selected[i] = True
            to_rm = np.union1d(np.array([i]), neighbors[i])
            remaining[to_rm] = False
            # Update degrees of remaining nodes
            rem_now = np.where(remaining)[0]
            if len(rem_now) > 0:
                deg[rem_now] -= conflicts[rem_now][:, to_rm].sum(axis=1)

        return selected

    def repair_to_independent(x_in):
        """Remove vertices to make set independent (greedy)."""
        x_out = x_in.copy()
        idxs = np.where(x_out)[0]
        # Process in order of increasing degree among selected
        sel_deg = degrees[idxs]
        order = idxs[np.argsort(sel_deg)]
        for i in order:
            if not x_out[i]:
                continue
            nbrs = neighbors[i]
            if np.any(x_out[nbrs]):
                x_out[i] = False
        return x_out

    best_x = np.zeros(N4, dtype=bool)
    best_size = 0

    # Multiple restarts with different SA parameters
    n_restarts = 0
    while time.time() < deadline - 3:
        n_restarts += 1
        x0 = greedy_independent_set()
        # Vary SA time budget slightly
        sa_time = 0.6 + 0.2 * (n_restarts % 5)
        x_sa = simulate_annealing(x0, sa_time)
        x_clean = repair_to_independent(x_sa)
        sz = int(x_clean.sum())
        if sz > best_size:
            best_size = sz
            best_x = x_clean.copy()

        # Occasionally try random init instead of greedy
        if n_restarts % 7 == 0:
            x_rand = rng.random(N4) < 0.15
            x_sa2 = simulate_annealing(x_rand, sa_time)
            x_clean2 = repair_to_independent(x_sa2)
            sz2 = int(x_clean2.sum())
            if sz2 > best_size:
                best_size = sz2
                best_x = x_clean2.copy()

    # Collect final W4 vectors
    sel_v4 = vectors_v4[best_x]
    all_vecs = np.vstack([sel_v4, vectors_v1])

    return all_vecs

# EVOLVE-BLOCK-END
