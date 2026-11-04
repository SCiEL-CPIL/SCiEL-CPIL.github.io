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
    # Since supports are distinct, intersection is 0, 1, 2, or 3.
    # Conflict if intersection is 3.
    
    # Precompute support masks for fast intersection
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3
    
    # Create adjacency list for faster iteration
    neighbors = [np.where(conflict[i])[0].tolist() for i in range(N_S)]
    
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init(order):
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                for n in neighbors[v]:
                    blocked[n] = True
        return x, np.sum(x)

    def local_search(x, current_size, num_iterations):
        best_x = x.copy()
        best_size_local = current_size
        x = x.copy()
        
        for it in range(num_iterations):
            if time.time() > deadline - 15:
                break
            
            # Compute blocked nodes
            if current_size > 0:
                # Vectorized blocked computation
                blocked = np.zeros(N_S, dtype=bool)
                # This is O(N_S * avg_degree) if done naively, but numpy is fast
                # Or use matrix multiplication?
                # conflict is bool matrix. x is bool vector.
                # blocked = conflict[x].any(axis=0)
                # This is very fast for N=330.
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            
            if len(add_candidates) > 0:
                # Add a random candidate
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            else:
                # Try to swap
                selected = np.where(x)[0]
                if len(selected) == 0:
                    break
                
                v_rem = py_rng.choice(selected)
                x[v_rem] = False
                current_size -= 1
                
                # Recompute blocked
                if current_size > 0:
                    blocked = conflict[x].any(axis=0)
                else:
                    blocked = np.zeros(N_S, dtype=bool)
                
                add_candidates = np.where(~x & ~blocked)[0]
                if len(add_candidates) > 0:
                    v_add = py_rng.choice(add_candidates)
                    x[v_add] = True
                    current_size += 1
                    if current_size > best_size_local:
                        best_size_local = current_size
                        best_x = x.copy()
                else:
                    # Revert
                    x[v_rem] = True
                    current_size += 1

        return best_x, best_size_local

    # Strategies
    strategies = []
    degrees = np.sum(conflict, axis=1)
    order_deg = np.argsort(degrees)
    strategies.append(order_deg)
    strategies.append(order_deg[::-1])
    
    for _ in range(20):
        order = np.arange(N_S)
        rng.shuffle(order)
        strategies.append(order)

    num_restarts = 200
    iterations_per_restart = 2000

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        if restart < len(strategies):
            order = strategies[restart]
        else:
            order = np.arange(N_S)
            rng.shuffle(order)
        
        x, size = greedy_init(order)
        best_x, best_size_local = local_search(x, size, iterations_per_restart)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate Type D vectors
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    final_vectors = np.vstack([vectors_A, selected_vectors_D])
    return final_vectors
