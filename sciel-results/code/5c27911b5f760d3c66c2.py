import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # ---------- Type A: one non-zero entry +/-2 ----------
    vectors_A = []
    for i in range(D):
        v1 = [0] * D; v1[i] = 2; vectors_A.append(v1)
        v2 = [0] * D; v2[i] = -2; vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # ---------- Type D: four non-zero entries +/-1 ----------
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # conflict: two 4-sets conflict iff |intersection| >= 3
    sm = np.zeros(N_S, dtype=np.int32)
    for s_idx, s in enumerate(supports):
        m = 0
        for i in s:
            m |= (1 << i)
        sm[s_idx] = m

    pop = np.zeros(2048, dtype=np.int8)
    for i in range(1, 2048):
        pop[i] = pop[i >> 1] + (i & 1)

    # fully vectorized conflict matrix
    inter = sm[:, None] & sm[None, :]
    conflict = (pop[inter] >= 3)
    np.fill_diagonal(conflict, False)

    # bitset adjacency for fast local search
    nb = np.zeros((N_S, (N_S + 63) // 64), dtype=np.int64)
    for s in range(N_S):
        cs = np.where(conflict[s])[0]
        for c in cs:
            nb[s, c >> 6] |= (1 << (c & 63))
    nwords = nb.shape[1]
    degree = np.array([np.count_nonzero(conflict[s]) for s in range(N_S)])

    # ---------- Type M: two +/-2 and three +/-1 (96 vectors, norm 2) ----------
    vectors_M = []
    for pair in itertools.combinations(indices, 2):
        for triple in itertools.combinations(indices, 3):
            if len(set(pair) & set(triple)) == 0:
                for sp in itertools.product([1, -1], repeat=2):
                    for st in itertools.product([1, -1], repeat=3):
                        v = [0] * D
                        for i, sgn in zip(pair, sp):
                            v[i] = 2 * sgn
                        for i, sgn in zip(triple, st):
                            v[i] = sgn
                        vectors_M.append(v)
    vectors_M = np.array(vectors_M, dtype=np.int64)

    # ---------- Build all Type D vectors ----------
    all_vectors_D = []
    all_support_indices = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            all_support_indices.append(s_idx)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    all_support_indices = np.array(all_support_indices, dtype=np.int64)

    # ---------- Local search for max independent set of supports ----------
    py_rng = random.Random(42)

    def independent_size(sel):
        # count conflicts among selected (should be 0 if valid)
        return 0

    def greedy_init(order):
        x = np.zeros(N_S, dtype=bool)
        conf_count = np.zeros(N_S, dtype=np.int32)
        cnt = 0
        for v in order:
            if conf_count[v] == 0:
                x[v] = True
                cnt += 1
                conf_count[nb_row(v)] += 1
        return x, cnt

    def nb_row(v):
        return np.where(conflict[v])[0]

    def local_search(x, current_size, num_iterations):
        best_x = x.copy()
        best_size = current_size
        # conflict count from selected
        conf_count = np.zeros(N_S, dtype=np.int32)
        for v in np.where(x)[0]:
            conf_count[nb_row(v)] += 1

        for it in range(num_iterations):
            if time.time() > deadline - 35:
                break
            add_mask = ~x & (conf_count == 0)
            rem_mask = x
            n_add = np.count_nonzero(add_mask)
            n_rem = np.count_nonzero(rem_mask)
            if n_add == 0 and n_rem == 0:
                break
            r = py_rng.random()
            if r < 0.75 and n_add > 0:
                add_indices = np.where(add_mask)[0]
                v = add_indices[py_rng.randint(0, len(add_indices) - 1)]
                x[v] = True
                current_size += 1
                conf_count[nb_row(v)] += 1
                if current_size > best_size:
                    best_size = current_size
                    best_x = x.copy()
            elif r < 0.90 and n_rem > 0:
                rem_indices = np.where(rem_mask)[0]
                v = rem_indices[py_rng.randint(0, len(rem_indices) - 1)]
                x[v] = False
                current_size -= 1
                conf_count[nb_row(v)] -= 1
            else:
                if n_add > 0 and n_rem > 0:
                    add_indices = np.where(add_mask)[0]
                    rem_indices = np.where(rem_mask)[0]
                    va = add_indices[py_rng.randint(0, len(add_indices) - 1)]
                    vr = rem_indices[py_rng.randint(0, len(rem_indices) - 1)]
                    if va != vr:
                        x[vr] = False
                        conf_count[nb_row(vr)] -= 1
                        x[va] = True
                        conf_count[nb_row(va)] += 1
        return best_x, best_size

    best_support_set = set()
    best_size = 0
    strategies = ['degree', 'random', 'mixed']
    num_restarts = 400
    iters = 4000
    rng = np.random.RandomState(42)
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        strat = strategies[restart % len(strategies)]
        if strat == 'degree':
            order = np.argsort(degrees if False else degree)
        elif strat == 'random':
            order = np.arange(N_S)
            rng.shuffle(order)
        else:
            order = np.argsort(degree + rng.random(N_S) * 10)
        x, size = greedy_init(order)
        bx, bs = local_search(x, size, iters)
        if bs > best_size:
            best_size = bs
            best_support_set = set(np.where(bx)[0].tolist())

    # ---------- Assemble final vector set ----------
    in_selected = np.isin(all_support_indices, list(best_support_set))
    current = all_vectors_D[in_selected].tolist()

    # Vectorized greedy fill for remaining Type D vectors
    remaining = np.where(~in_selected)[0]
    rng.shuffle(remaining)

    if len(current) > 0:
        existing = np.array(current, dtype=np.int64)
    else:
        existing = np.empty((0, D), dtype=np.int64)

    # Process in chunks: for each candidate check max |dot| over existing
    CH = 256
    for s in range(0, len(remaining), CH):
        if time.time() > deadline - 12:
            break
        chunk_idx = remaining[s:s + CH]
        cand = all_vectors_D[chunk_idx]
        if len(existing) == 0:
            current.extend(cand.tolist())
            existing = cand.copy()
        else:
            dots = existing @ cand.T          # (n_existing, chunk)
            ok = np.abs(dots).max(axis=0) <= 2
            for j in np.where(ok)[0]:
                current.append(cand[j].tolist())
            if np.any(ok):
                existing = np.vstack([existing, cand[ok]])

    # Greedy add Type M vectors (vectorized compatibility check)
    if len(vectors_M) > 0:
        if len(existing) == 0:
            current.extend(vectors_M.tolist())
            existing = vectors_M.copy()
        else:
            dots = existing @ vectors_M.T
            ok = np.abs(dots).max(axis=0) <= 2
            for j in np.where(ok)[0]:
                current.append(vectors_M[j].tolist())
            if np.any(ok):
                existing = np.vstack([existing, vectors_M[ok]])

    final_vectors_D = np.array(current, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
