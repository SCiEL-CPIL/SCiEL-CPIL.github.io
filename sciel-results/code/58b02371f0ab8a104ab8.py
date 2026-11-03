import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
    # Conflict if intersection size >= 3 (necessary but not sufficient for incompatibility)
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    # Two supports conflict if they share >= 3 positions
    conflict = popcounts >= 3

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Greedy initialization: pick node with fewest conflicts first
    deg = conflict.sum(axis=1)
    
    def greedy_init():
        order = np.arange(N_S)
        # Sort by degree (fewest conflicts first) with random tie-breaking
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x, np.sum(x)

    # Local search with simulated annealing - optimized
    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        cooling = (T_end / T0) ** (1.0 / num_iterations)
        
        for it in range(num_iterations):
            if time.time() > deadline - 10:
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
            if move_type < 0.7 and len(add_candidates) > 0:
                v = add_candidates[py_rng.randint(0, len(add_candidates) - 1)]
                x[v] = True
                current_size += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            elif move_type < 0.85 and len(remove_candidates) > 0:
                v = remove_candidates[py_rng.randint(0, len(remove_candidates) - 1)]
                x[v] = False
                current_size -= 1
                if py_rng.random() >= np.exp(-1.0 / T):
                    x[v] = True
                    current_size += 1
            else:
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_add = add_candidates[py_rng.randint(0, len(add_candidates) - 1)]
                    v_rem = remove_candidates[py_rng.randint(0, len(remove_candidates) - 1)]
                    if v_add != v_rem:
                        # Check if adding v_add is compatible after removing v_rem
                        # v_add is not blocked by current x (it's in add_candidates)
                        # So it's already compatible with all selected supports
                        x[v_rem] = False
                        x[v_add] = True
                        # Size stays the same
            
            T *= cooling

        return best_x, best_size_local

    # Multiple greedy initializations with different orderings
    num_restarts = 100
    iterations_per_restart = 8000
    T0 = 1.0
    T_end = 0.01

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        x, size = greedy_init()
        best_x, best_size_local = local_search(x, size, iterations_per_restart, T0, T_end)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate ALL Type D vectors upfront
    all_vectors_D = []
    all_support_indices = []  # which support each vector belongs to
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            all_support_indices.append(s_idx)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    all_support_indices = np.array(all_support_indices, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    # Start with vectors from selected supports
    in_selected_support = np.isin(all_support_indices, list(best_support_set))
    current_set_vectors = all_vectors_D[in_selected_support].tolist()
    
    # For remaining vectors, do greedy fill
    remaining_indices = np.where(~in_selected_support)[0]
    rng.shuffle(remaining_indices)
    
    if len(current_set_vectors) > 0:
        existing = np.array(current_set_vectors, dtype=np.int64)
        n_existing = len(existing)
    else:
        existing = np.empty((0, D), dtype=np.int64)
        n_existing = 0
    
    # Process in batches, checking compatibility
    batch_size = 500
    for batch_start in range(0, len(remaining_indices), batch_size):
        if time.time() > deadline - 5:
            break
        batch_end = min(batch_start + batch_size, len(remaining_indices))
        batch_idx = remaining_indices[batch_start:batch_end]
        batch_vecs = all_vectors_D[batch_idx]
        
        if n_existing == 0:
            current_set_vectors.extend(batch_vecs.tolist())
            existing = np.vstack([existing, batch_vecs]) if n_existing > 0 else batch_vecs
            n_existing = len(existing)
        else:
            # Compute dot products with existing set
            dots = batch_vecs @ existing.T
            # Compatible if |dot| < 3 for all existing vectors
            # Since all vectors have entries in {-1, 0, 1, 2}, max |dot| with norm-2 vectors
            # For Type D (norm 2) vs Type D (norm 2): dot can be -4 to 4
            # Compatible if max(abs(dot)) < 3, i.e., |dot| <= 2
            compatible = (np.abs(dots).max(axis=1) <= 2)
            
            new_vecs = batch_vecs[compatible]
            if len(new_vecs) > 0:
                current_set_vectors.extend(new_vecs.tolist())
                existing = np.vstack([existing, new_vecs])
                n_existing = len(existing)

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
