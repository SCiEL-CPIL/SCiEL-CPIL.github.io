import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # ---- Type A: one non-zero entry +/- 2 (22 vectors, norm 2) ----
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # ---- Type D: four non-zero entries +/- 1 (16 vectors per support, norm 2) ----
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)  # 330

    # support masks (11-bit)
    support_masks = np.zeros(N_S, dtype=np.int64)
    for s_idx, s in enumerate(supports):
        m = 0
        for i in s:
            m |= (1 << i)
        support_masks[s_idx] = m

    # popcount table for 11-bit ints
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # conflict if intersection size >= 3
    inter = support_masks[:, None] & support_masks[None, :]
    conflict = popcount[inter] >= 3
    conflict = conflict.astype(np.int8)
    np.fill_diagonal(conflict, 0)
    N_conf = np.sum(conflict, axis=1)

    # ---- Fast maximum independent set via simulated annealing on the support graph ----
    # 16 Type-D vectors per chosen support, so maximizing supports maximizes Type-D count.
    rng = np.random.RandomState(1234)
    py_rng = random.Random(1234)

    best_size = 0
    best_x = None

    def evaluate(x):
        return int(np.sum(x))

    def greedy_init():
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=np.int8)
        blocked = np.zeros(N_S, dtype=np.int8)
        for v in order:
            if not blocked[v]:
                x[v] = 1
                blocked |= conflict[v]
        return x

    x = greedy_init()
    cur_size = evaluate(x)
    best_x = x.copy()
    best_size = cur_size

    # temperature schedule
    T0 = 1.5
    T_end = 0.03
    n_iter = 2000000

    for it in range(n_iter):
        if it % 20000 == 0:
            if time.time() > deadline - 45.0:
                break
            T = T0 * (T_end / T0) ** (it / n_iter)
            # occasionally reheat if stuck
            if it % 200000 == 100000 and cur_size < best_size:
                T = max(T, 0.5)
        else:
            T = T0 * (T_end / T0) ** (it / n_iter)

        # choose a random flip
        v = rng.randint(N_S)
        if x[v]:
            # remove v
            delta = -1
            new_x = x
            new_size = cur_size - 1
            accept = (new_size >= best_size) or (py_rng.random() < np.exp((new_size - cur_size) / T))
            if accept:
                x[v] = 0
                cur_size = new_size
                if cur_size > best_size:
                    best_size = cur_size
                    best_x = x.copy()
        else:
            # try to add v if not blocked
            if np.any(conflict[v] & x):
                # blocked: instead try a swap: remove a conflicting one, add v
                conf = np.where(conflict[v] & x)[0]
                if len(conf) == 0:
                    continue
                r = conf[rng.randint(len(conf))]
                # swap v in, r out
                x[v] = 1
                x[r] = 0
                # size unchanged, but may enable future growth; accept with prob
                if py_rng.random() < np.exp((0.0) / T):
                    cur_size = cur_size  # unchanged
                # check if this enables an immediate add
                # (cheap improvement: try to add a free node)
                if it % 500 == 0:
                    blocked = np.any(conflict & x[:, None], axis=1)
                    free = np.where((x == 0) & (~blocked))[0]
                    if len(free) > 0:
                        a = free[rng.randint(len(free))]
                        x[a] = 1
                        cur_size += 1
                        if cur_size > best_size:
                            best_size = cur_size
                            best_x = x.copy()
            else:
                x[v] = 1
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_x = x.copy()

    # ---- Deterministic polish: greedy add + 1-for-1 swaps to grow the set ----
    x = best_x.copy()
    cur_size = evaluate(x)
    improved = True
    polish_iter = 0
    while improved and polish_iter < 200000:
        if time.time() > deadline - 30.0:
            break
        improved = False
        polish_iter += 1
        # greedy add pass (vectorized)
        blocked = np.any(conflict & x[:, None], axis=1)
        free = np.where((x == 0) & (~blocked))[0]
        if len(free) > 0:
            a = free[rng.randint(len(free))]
            x[a] = 1
            cur_size += 1
            improved = True
            if cur_size > best_size:
                best_size = cur_size
                best_x = x.copy()
            continue
        # try swaps: remove a selected, add a free (net 0) to enable growth
        selected = np.where(x == 1)[0]
        if len(selected) == 0:
            break
        # find a removal that unblocks at least one new free node
        found = False
        for _ in range(8):
            r = selected[rng.randint(len(selected))]
            x[r] = 0
            blocked2 = np.any(conflict & x[:, None], axis=1)
            free2 = np.where((x == 0) & (~blocked2))[0]
            free2 = free2[free2 != r]
            if len(free2) > 0:
                a = free2[rng.randint(len(free2))]
                x[a] = 1
                # net size unchanged; keep if it leads to a free node later
                found = True
                break
            else:
                x[r] = 1  # revert
        if found:
            improved = True

    best_x = best_x.copy()
    best_size = evaluate(best_x)

    selected_indices = np.where(best_x == 1)[0]

    # ---- Build the selected Type-D vectors (16 per support) ----
    selected_vectors_D = []
    for s_idx in selected_indices:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # ---- Greedy fill with remaining Type-D vectors (fast, vectorized) ----
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    # all Type-D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    existing = np.array(current_set_vectors, dtype=np.int64)

    # precompute all dot products against existing in batches
    rng2 = np.random.RandomState(999)
    order = np.arange(N_all_D)
    rng2.shuffle(order)

    for b in range(0, N_all_D, 400):
        if time.time() > deadline - 15.0:
            break
        end = min(b + 400, N_all_D)
        batch = order[b:end]
        # skip ones already present
        new_idx = []
        for idx in batch:
            if tuple(all_vectors_D[idx]) not in current_set_vectors_set:
                new_idx.append(idx)
        if not new_idx:
            continue
        new_vecs = all_vectors_D[new_idx]
        dots = new_vecs @ existing.T
        compatible = dots.max(axis=1) < 3
        for j, idx in enumerate(new_idx):
            if not compatible[j]:
                continue
            v_single = all_vectors_D[idx].reshape(1, -1)
            dots_check = v_single @ existing.T
            if dots_check.max() < 3:
                current_set_vectors.append(all_vectors_D[idx])
                current_set_vectors_set.add(tuple(all_vectors_D[idx]))
                existing = np.vstack([existing, v_single])

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
