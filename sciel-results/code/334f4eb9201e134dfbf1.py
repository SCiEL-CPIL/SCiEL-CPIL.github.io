import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph: conflict if intersection size >= 3
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix computation
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3
    conflict = conflict.astype(bool)

    # Degree of each node
    degrees = conflict.sum(axis=1)

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init():
        order = np.argsort(degrees + rng.rand(N_S) * 0.01)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x, int(np.sum(x))

    def local_search(x, current_size, num_iterations):
        best_x = x.copy()
        best_size_local = current_size
        improved = True
        stall_count = 0
        max_stall = 50

        while improved and num_iterations > 0 and stall_count < max_stall:
            if time.time() > deadline - 15:
                break
            improved = False
            stall_count += 1

            # Greedy add pass
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                blocked = conflict[x].any(axis=0)
                add_candidates = np.where(~x & ~blocked)[0]
                if len(add_candidates) == 0:
                    break
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                improved = True
                stall_count = 0
                num_iterations -= 1

            if current_size > best_size_local:
                best_size_local = current_size
                best_x = x.copy()

            # Try swaps
            if time.time() > deadline - 15:
                break
            remove_candidates = np.where(x)[0]
            if len(remove_candidates) == 0:
                break
            for _ in range(5):
                if time.time() > deadline - 15:
                    break
                v_rem = py_rng.choice(remove_candidates)
                x[v_rem] = False
                current_size -= 1
                blocked = conflict[x].any(axis=0)
                add_candidates = np.where(~x & ~blocked)[0]
                if len(add_candidates) >= 2:
                    a1 = py_rng.choice(add_candidates)
                    x[a1] = True
                    current_size += 1
                    blocked2 = conflict[x].any(axis=0)
                    add_candidates2 = np.where(~x & ~blocked2)[0]
                    if len(add_candidates2) > 0:
                        a2 = py_rng.choice(add_candidates2)
                        x[a2] = True
                        current_size += 1
                        improved = True
                        stall_count = 0
                elif len(add_candidates) == 1:
                    a1 = py_rng.choice(add_candidates)
                    x[a1] = True
                    current_size += 1
                else:
                    x[v_rem] = True
                    current_size += 1
                num_iterations -= 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()

        return best_x, best_size_local

    num_restarts = 300
    iterations_per_restart = 8000

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        x, size = greedy_init()
        best_x, best_size_local = local_search(x, size, iterations_per_restart)

        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Vectorized greedy fill
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    # Batch process for speed
    batch_size = 200
    for batch_start in range(0, N_all_D, batch_size):
        if time.time() > deadline - 10:
            break
        batch_end = min(batch_start + batch_size, N_all_D)
        batch_indices = indices_all[batch_start:batch_end]

        new_batch = []
        for idx in batch_indices:
            if tuple(all_vectors_D[idx]) not in current_set_vectors_set:
                new_batch.append(idx)

        if not new_batch:
            continue

        new_vecs = all_vectors_D[new_batch]

        if len(current_set_vectors) > 0:
            existing = np.array(current_set_vectors)
            dots = new_vecs @ existing.T
            compatible = (dots.max(axis=1) < 3)

            for j, idx in enumerate(new_batch):
                if not compatible[j]:
                    continue
                v_single = all_vectors_D[idx].reshape(1, -1)
                dots_check = v_single @ existing.T
                if dots_check.max() < 3:
                    current_set_vectors.append(all_vectors_D[idx])
                    current_set_vectors_set.add(tuple(all_vectors_D[idx]))
                    existing = np.vstack([existing, v_single])
        else:
            for idx in new_batch:
                current_set_vectors.append(all_vectors_D[idx])
                current_set_vectors_set.add(tuple(all_vectors_D[idx]))

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
