"""Kissing number in dimension 11: structural reformulation as MIS on support graph + SA."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    D = 11

    # Generate all 4-subsets
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    # Build conflict graph: edge if |intersection| == 3
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i + 1, n_sup):
            s_j = set(supports[j])
            if len(s_i & s_j) == 3:
                adj[i].add(j)
                adj[j].add(i)

    # Convert to numpy bitmask for fast intersection
    adj_mask = np.zeros((n_sup, n_sup), dtype=np.uint8)
    for i in range(n_sup):
        for j in adj[i]:
            adj_mask[i, j] = 1
            adj_mask[j, i] = 1

    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if adj_mask[v, u])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove
        return current_set

    best_set = greedy_is()
    best_size = len(best_set)

    # Simulated annealing with vectorized operations
    np.random.seed(42)
    random.seed(42)

    S = best_set.copy()
    S_arr = np.zeros(n_sup, dtype=np.uint8)
    for v in S:
        S_arr[v] = 1
    current_size = len(S)
    T = 1.0
    cooling = 0.9995

    max_steps = 300000
    step = 0
    while step < max_steps and time.time() < deadline - 25.0:
        step += 1
        v = random.randint(0, n_sup - 1)

        if S_arr[v]:
            if random.random() < 0.05:
                S_arr[v] = 0
                S.discard(v)
                current_size -= 1
            continue

        # v not in S
        N = np.where(adj_mask[v] & S_arr)[0]
        n_N = len(N)

        if n_N == 0:
            S_arr[v] = 1
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
        else:
            delta = 1 - n_N
            if delta >= 0:
                S_arr[N] = 0
                S.difference_update(N.tolist())
                S_arr[v] = 1
                S.add(v)
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                if T > 0.01:
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S_arr[N] = 0
                        S.difference_update(N.tolist())
                        S_arr[v] = 1
                        S.add(v)
                        current_size += delta
                        if current_size > best_size:
                            best_size = current_size
                            best_set = S.copy()

        T = max(0.01, T * cooling)

    # Multiple restarts with different seeds
    for restart in range(15):
        if time.time() > deadline - 20.0:
            break
        np.random.seed(100 + restart)
        random.seed(100 + restart)

        # Randomized greedy
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if adj_mask[v, u])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove

        S = current_set.copy()
        S_arr = np.zeros(n_sup, dtype=np.uint8)
        for v in S:
            S_arr[v] = 1
        current_size = len(S)
        T = 1.0
        cooling = 0.9995

        for step2 in range(80000):
            if time.time() > deadline - 20.0:
                break
            v = random.randint(0, n_sup - 1)

            if S_arr[v]:
                if random.random() < 0.05:
                    S_arr[v] = 0
                    S.discard(v)
                    current_size -= 1
                continue

            N = np.where(adj_mask[v] & S_arr)[0]
            n_N = len(N)

            if n_N == 0:
                S_arr[v] = 1
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                delta = 1 - n_N
                if delta >= 0:
                    S_arr[N] = 0
                    S.difference_update(N.tolist())
                    S_arr[v] = 1
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    if T > 0.01:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S_arr[N] = 0
                            S.difference_update(N.tolist())
                            S_arr[v] = 1
                            S.add(v)
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_set = S.copy()

            T = max(0.01, T * cooling)

    # Construct vectors from best_set: all 16 sign patterns for each chosen support
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)

    existing = np.array(vectors, dtype=np.int32)
    existing_norms_sq = np.sum(existing ** 2, axis=1)
    max_norm_sq = int(existing_norms_sq.max())

    # Generate and try to add lower-weight vectors using batched checks
    # Weight-3 vectors: need c·v <= 1 for all existing v
    # Weight-2 vectors: need c·v <= 1 for all existing v  
    # Weight-1 vectors: need c·v <= 0 for all existing v

    def try_add_batch(candidates_arr, max_dot_allowed):
        """Try to add candidates one by one, checking against all existing vectors."""
        added = 0
        for i in range(len(candidates_arr)):
            if time.time() > deadline - 5.0:
                break
            c = candidates_arr[i]
            dots = existing @ c
            if np.all(dots <= max_dot_allowed):
                vectors.append(c.tolist())
                existing = np.vstack([existing, c.reshape(1, -1)])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())
                added += 1
        return added

    # Generate weight-3 candidates
    w3_list = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int32)
            for pos, s in zip(positions, signs):
                v[pos] = s
            w3_list.append(v)
    w3_arr = np.array(w3_list, dtype=np.int32)

    # Sort by number of conflicts (fewer conflicts first)
    # dot products with existing
    dots_w3 = w3_arr @ existing.T  # shape (n_w3, n_existing)
    max_dots_w3 = np.max(dots_w3, axis=1)
    # Candidates with max_dot <= 1 are immediately compatible
    compatible_w3 = np.where(max_dots_w3 <= 1)[0]
    # Sort remaining by how close they are to being compatible
    remaining_w3 = np.where(max_dots_w3 > 1)[0]
    remaining_w3_sorted = remaining_w3[np.argsort(max_dots_w3[remaining_w3])]
    w3_order = np.concatenate([compatible_w3, remaining_w3_sorted])
    
    for idx in w3_order:
        if time.time() > deadline - 5.0:
            break
        c = w3_arr[idx]
        dots = existing @ c
        if np.all(dots <= 1):
            vectors.append(c.tolist())
            existing = np.vstack([existing, c.reshape(1, -1)])
            existing_norms_sq = np.sum(existing ** 2, axis=1)
            max_norm_sq = int(existing_norms_sq.max())

    # Generate weight-2 candidates
    w2_list = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int32)
            for pos, s in zip(positions, signs):
                v[pos] = s
            w2_list.append(v)
    w2_arr = np.array(w2_list, dtype=np.int32)

    dots_w2 = w2_arr @ existing.T
    max_dots_w2 = np.max(dots_w2, axis=1)
    compatible_w2 = np.where(max_dots_w2 <= 1)[0]
    remaining_w2 = np.where(max_dots_w2 > 1)[0]
    remaining_w2_sorted = remaining_w2[np.argsort(max_dots_w2[remaining_w2])]
    w2_order = np.concatenate([compatible_w2, remaining_w2_sorted])

    for idx in w2_order:
        if time.time() > deadline - 3.0:
            break
        c = w2_arr[idx]
        dots = existing @ c
        if np.all(dots <= 1):
            vectors.append(c.tolist())
            existing = np.vstack([existing, c.reshape(1, -1)])
            existing_norms_sq = np.sum(existing ** 2, axis=1)
            max_norm_sq = int(existing_norms_sq.max())

    # Generate weight-1 candidates
    w1_list = []
    for pos in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int32)
            v[pos] = s
            w1_list.append(v)
    w1_arr = np.array(w1_list, dtype=np.int32)

    dots_w1 = w1_arr @ existing.T
    max_dots_w1 = np.max(dots_w1, axis=1)
    compatible_w1 = np.where(max_dots_w1 <= 0)[0]
    remaining_w1 = np.where(max_dots_w1 > 0)[0]
    remaining_w1_sorted = remaining_w1[np.argsort(max_dots_w1[remaining_w1])]
    w1_order = np.concatenate([compatible_w1, remaining_w1_sorted])

    for idx in w1_order:
        if time.time() > deadline - 2.0:
            break
        c = w1_arr[idx]
        dots = existing @ c
        if np.all(dots <= 0):
            vectors.append(c.tolist())
            existing = np.vstack([existing, c.reshape(1, -1)])
            existing_norms_sq = np.sum(existing ** 2, axis=1)
            max_norm_sq = int(existing_norms_sq.max())

    if not vectors:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = np.array(vectors, dtype=np.int64)
    return result
