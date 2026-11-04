import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2 (22 vectors)
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
    N_S = len(supports)  # 330

    # Build all Type D vectors (330 * 16 = 5280... wait, 2^4=16, so 330*16=5280)
    all_vectors_D = []
    support_of = []  # which support each vector belongs to
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            support_of.append(s)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int8)
    N_all_D = len(all_vectors_D)  # 5280

    # Precompute dot product matrix between all Type D vectors (N_all_D x N_all_D)
    # Using int8, values are in [-4, 4]
    dot_matrix = np.zeros((N_all_D, N_all_D), dtype=np.int8)
    block_size = 512
    for i0 in range(0, N_all_D, block_size):
        i1 = min(i0 + block_size, N_all_D)
        dot_matrix[i0:i1, :] = all_vectors_D[i0:i1] @ all_vectors_D.T

    # Compatibility: dot < 3, i.e., dot <= 2
    compatible = dot_matrix < 3

    # Support masks for independent set (one vector per support)
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Popcount table
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Conflict matrix for supports: conflict if intersection >= 3
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Greedy initialization with random ordering
    def greedy_init():
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x, np.sum(x)

    # Vectorized local search
    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        if num_iterations > 0:
            cooling = (T_end / T0) ** (1.0 / num_iterations)
        else:
            cooling = 1.0

        for it in range(num_iterations):
            if time.time() > deadline - 20:
                break

            if current_size > 0:
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)

            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = np.where(x)[0]

            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break

            move_type = py_rng.random()
            if move_type < 0.65 and len(add_candidates) > 0:
                v = add_candidates[py_rng.randint(len(add_candidates))]
                if not x[v]:
                    x[v] = True
                    new_size = current_size + 1
                    if new_size > best_size_local:
                        best_size_local = new_size
                        best_x = x.copy()
                    current_size = new_size
            elif move_type < 0.85 and len(remove_candidates) > 0:
                v = remove_candidates[py_rng.randint(len(remove_candidates))]
                if x[v]:
                    x[v] = False
                    new_size = current_size - 1
                    if py_rng.random() < np.exp(-1.0 / T):
                        current_size = new_size
                    else:
                        x[v] = True
            else:
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_add = add_candidates[py_rng.randint(len(add_candidates))]
                    v_rem = remove_candidates[py_rng.randint(len(remove_candidates))]
                    if v_add != v_rem and not x[v_add] and x[v_rem]:
                        # Check if v_add is compatible with all other selected (excluding v_rem)
                        x[v_rem] = False
                        if not np.any(conflict[v_add, x]):
                            x[v_add] = True
                            # current_size unchanged
                            if current_size > best_size_local:
                                best_size_local = current_size
                                best_x = x.copy()
                        else:
                            x[v_rem] = True

            T *= cooling

        return best_x, best_size_local

    num_restarts = 200
    iterations_per_restart = 8000
    T0 = 1.0
    T_end = 0.01

    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        x, size = greedy_init()
        best_x, best_size_local = local_search(x, size, iterations_per_restart, T0, T_end)

        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Now build the final set: for each selected support, we want to add as many sign patterns as possible
    # that are compatible with each other and with all other selected supports' sign patterns.
    # Strategy: start with the best sign pattern per support (the one that maximizes compatibility),
    # then greedily add more.

    # For each support, get the indices of its 16 vectors in all_vectors_D
    support_to_vecs = {}
    for i, s in enumerate(supports):
        vec_start = i * 16
        vec_end = (i + 1) * 16
        support_to_vecs[i] = list(range(vec_start, vec_end))

    # Initial selection: one vector per selected support (the first one, all +1)
    selected_vecs = set()
    for s_idx in best_support_set:
        selected_vecs.add(s_idx * 16)  # all +1 pattern

    # Greedily add more sign patterns
    # For efficiency, maintain a "blocked" set of vectors that are incompatible with selected
    blocked_vecs = np.zeros(N_all_D, dtype=bool)
    for sv in selected_vecs:
        blocked_vecs |= ~compatible[sv]

    # Try to add more vectors from selected supports
    added = True
    while added and time.time() < deadline - 10:
        added = False
        for s_idx in best_support_set:
            vecs = support_to_vecs[s_idx]
            for v_idx in vecs:
                if v_idx not in selected_vecs and not blocked_vecs[v_idx]:
                    selected_vecs.add(v_idx)
                    blocked_vecs |= ~compatible[v_idx]
                    added = True

    # Now try to add vectors from non-selected supports if compatible
    # This can add more vectors
    for v_idx in range(N_all_D):
        if time.time() > deadline - 5:
            break
        if v_idx not in selected_vecs and not blocked_vecs[v_idx]:
            selected_vecs.add(v_idx)
            blocked_vecs |= ~compatible[v_idx]

    final_vectors_D = all_vectors_D[list(selected_vecs)]
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
