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

    # Precompute conflict matrix using vectorized bit operations
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        inter = support_masks[i] & support_masks[i+1:]
        # Count bits efficiently
        counts = np.array([bin(x).count('1') for x in inter])
        conflict[i, i+1:][counts >= 3] = True
        conflict[i+1:, i][counts >= 3] = True

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Simulated Annealing for Maximum Independent Set
    def greedy_init(r):
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v] and not x[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    # Simulated annealing with better move generation
    T0 = 1.0
    T_end = 0.01
    num_iterations = 30000
    cooling = (T_end / T0) ** (1.0 / num_iterations)

    for restart in range(80):
        if time.time() > deadline - 25:
            break

        x = greedy_init(restart)
        current_size = np.sum(x)
        best_x = x.copy()
        best_current_size = current_size

        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 25:
                break

            in_set = np.where(x)[0]
            not_in_set = np.where(~x)[0]

            # Compute blocked set efficiently
            if len(in_set) > 0:
                blocked = np.zeros(N_S, dtype=bool)
                for v in in_set:
                    blocked |= conflict[v]
            else:
                blocked = np.zeros(N_S, dtype=bool)

            # Add candidates: not in set and not blocked
            add_mask = ~x & ~blocked
            add_candidates = np.where(add_mask)[0]
            # Remove candidates: in set
            remove_candidates = in_set

            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break

            move_type = py_rng.random()
            if move_type < 0.5 and len(add_candidates) > 0:
                # Add a node
                v = add_candidates[py_rng.randint(0, len(add_candidates))]
                x[v] = True
                current_size += 1
                if current_size > best_current_size:
                    best_current_size = current_size
                    best_x = x.copy()
            elif move_type < 0.8 and len(remove_candidates) > 0:
                # Remove a node
                v = remove_candidates[py_rng.randint(0, len(remove_candidates))]
                x[v] = False
                current_size -= 1
                # Accept with probability exp(-1/T) for removal
                if py_rng.random() < np.exp(-1.0 / T):
                    pass  # Keep removal
                else:
                    x[v] = True
                    current_size += 1
            else:
                # Swap: remove one, add one
                if len(add_candidates) == 0 or len(remove_candidates) == 0:
                    continue
                v_remove = remove_candidates[py_rng.randint(0, len(remove_candidates))]
                v_add = add_candidates[py_rng.randint(0, len(add_candidates))]
                if v_remove == v_add:
                    continue
                # Check if v_add conflicts with any in-set node (excluding v_remove)
                in_set_no_remove = [v for v in in_set if v != v_remove]
                if len(in_set_no_remove) > 0 and np.any(conflict[v_add, in_set_no_remove]):
                    continue
                # Perform swap
                x[v_remove] = False
                x[v_add] = True
                # Size unchanged, always accept to explore

            T *= cooling

        if best_current_size > best_size:
            best_size = best_current_size
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

    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    current_matrix = selected_vectors_D.copy()

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

    # Batched vectorized greedy fill
    batch_size = 500
    for batch_start in range(0, N_all_D, batch_size):
        if time.time() > deadline - 15:
            break
        batch_end = min(batch_start + batch_size, N_all_D)
        batch_indices = indices_all[batch_start:batch_end]
        batch_vectors = all_vectors_D[batch_indices]

        # Filter out already-in-set vectors
        mask_new = np.ones(len(batch_vectors), dtype=bool)
        for i, v in enumerate(batch_vectors):
            if tuple(v) in current_set_vectors_set:
                mask_new[i] = False

        if not np.any(mask_new):
            continue

        new_vectors = batch_vectors[mask_new]

        # Check each new vector against current set using matrix multiplication
        if len(current_matrix) > 0:
            dots = new_vectors @ current_matrix.T
            can_add_mask = np.all(dots < 3, axis=1)
        else:
            can_add_mask = np.ones(len(new_vectors), dtype=bool)

        if np.any(can_add_mask):
            addable = new_vectors[can_add_mask]
            for v in addable:
                vt = tuple(v)
                if vt not in current_set_vectors_set:
                    current_set_vectors.append(v)
                    current_set_vectors_set.add(vt)
            if len(addable) > 0:
                current_matrix = np.vstack([current_matrix, addable]) if len(current_matrix) > 0 else addable

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
