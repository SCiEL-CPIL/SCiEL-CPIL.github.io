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

    # Precompute popcount for 11-bit integers (0 to 2047)
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix computation
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Weighted greedy initialization: pick node with fewest conflicts first
    def greedy_init():
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        # Precompute degrees
        deg = conflict.sum(axis=1)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x, np.sum(x)

    # Local search with simulated annealing
    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        cooling = (T_end / T0) ** (1.0 / num_iterations)
        
        for it in range(num_iterations):
            if time.time() > deadline - 15:
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
            if move_type < 0.6 and len(add_candidates) > 0:
                v = py_rng.choice(add_candidates)
                if not x[v]:
                    x[v] = True
                    new_size = current_size + 1
                    if new_size > best_size_local:
                        best_size_local = new_size
                        best_x = x.copy()
                    current_size = new_size
            elif move_type < 0.8 and len(remove_candidates) > 0:
                v = py_rng.choice(remove_candidates)
                if x[v]:
                    x[v] = False
                    new_size = current_size - 1
                    if py_rng.random() < np.exp(-1.0 / T):
                        current_size = new_size
                    else:
                        x[v] = True
            else:
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_add = py_rng.choice(add_candidates)
                    v_rem = py_rng.choice(remove_candidates)
                    if v_add != v_rem and not x[v_add] and x[v_rem]:
                        temp_x = x.copy()
                        temp_x[v_rem] = False
                        if not np.any(conflict[v_add, temp_x]):
                            x[v_rem] = False
                            x[v_add] = True
            
            T *= cooling

        return best_x, best_size_local

    num_restarts = 50
    iterations_per_restart = 12000
    T0 = 1.0
    T_end = 0.01

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        x, size = greedy_init()
        best_x, best_size_local = local_search(x, size, iterations_per_restart, T0, T_end)
        
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
        batch_vecs = all_vectors_D[batch_indices]
        
        # Filter out already-in-set vectors
        new_batch = []
        for idx in batch_indices:
            if tuple(all_vectors_D[idx]) not in current_set_vectors_set:
                new_batch.append(idx)
        
        if not new_batch:
            continue
        
        new_vecs = all_vectors_D[new_batch]
        
        # Check compatibility with existing set
        if len(current_set_vectors) > 0:
            existing = np.array(current_set_vectors)
            # Compute dot products: (N_new, N_existing)
            dots = new_vecs @ existing.T
            # A new vector is compatible if max dot < 3
            compatible = (dots.max(axis=1) < 3)
            
            for j, idx in enumerate(new_batch):
                if compatible[j]:
                    current_set_vectors.append(all_vectors_D[idx])
                    current_set_vectors_set.add(tuple(all_vectors_D[idx]))
        else:
            for idx in new_batch:
                current_set_vectors.append(all_vectors_D[idx])
                current_set_vectors_set.add(tuple(all_vectors_D[idx]))

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
