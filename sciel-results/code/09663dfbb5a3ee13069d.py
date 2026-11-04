import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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

    # Build conflict graph using bitmasks
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

    # Vectorized conflict matrix
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3
    
    # Compute degrees
    degrees = conflict.sum(axis=1)
    
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init_heuristic():
        """
        Greedy initialization using minimum degree heuristic.
        """
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        current_size = 0
        
        for _ in range(N_S):
            if not available.any():
                break
            
            avail_indices = np.where(available)[0]
            static_degrees = degrees[avail_indices]
            min_deg = static_degrees.min()
            candidates = avail_indices[static_degrees == min_deg]
            
            if len(candidates) == 0:
                break
            v = py_rng.choice(candidates.tolist())
            
            x[v] = True
            current_size += 1
            blocked |= conflict[v]
            available &= ~blocked
            available[v] = False
            
        return x, current_size

    def local_search_1swap(x, current_size, max_iters=5000):
        """
        Iterative 1-swap local search with improved swap logic.
        """
        best_x = x.copy()
        best_size_local = current_size
        
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
                
            if current_size > 0:
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            
            if len(add_candidates) > 0:
                # Greedily add a random candidate
                v = py_rng.choice(add_candidates.tolist())
                x[v] = True
                current_size += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            else:
                # No add possible. Try swaps to escape local minimum.
                in_set = np.where(x)[0]
                if len(in_set) == 0:
                    break
                
                # Try several random swaps
                improved = False
                not_in_x = np.where(~x)[0]
                
                for _ in range(20):
                    if time.time() > deadline - 15:
                        break
                    if len(not_in_x) == 0:
                        break
                    
                    v_rem = py_rng.choice(in_set.tolist())
                    v_add = py_rng.choice(not_in_x.tolist())
                    
                    if v_add == v_rem:
                        continue
                    
                    # Check if v_add conflicts with any node in x, excluding v_rem
                    conflicts_with_x = conflict[v_add] & x
                    # Remove the conflict with v_rem from consideration
                    if conflicts_with_x[v_rem]:
                        conflicts_with_x[v_rem] = False
                    
                    if not conflicts_with_x.any():
                        # Safe swap
                        x[v_rem] = False
                        x[v_add] = True
                        in_set = np.where(x)[0]
                        not_in_x = np.where(~x)[0]
                        improved = True
                        break
                
                if not improved:
                    # Perturb: randomly remove one node
                    if current_size > 1:
                        v_rem = py_rng.choice(in_set.tolist())
                        x[v_rem] = False
                        current_size -= 1
                    else:
                        break

        return best_x, best_size_local

    # Run multiple restarts
    num_restarts = 300
    iters_per_restart = 3000

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        x, size = greedy_init_heuristic()
        best_x, best_size_local = local_search_1swap(x, size, iters_per_restart)
        
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

    # Vectorized greedy fill: try to add more Type D vectors from non-selected supports
    # Generate all possible Type D vectors
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
    N_all_D = len(all_vectors_D)
    
    # Start with selected vectors
    current_vectors = selected_vectors_D.copy()
    current_set = set(map(tuple, current_vectors.tolist()))
    
    # For each non-selected support, try adding its 16 sign variants
    # Use vectorized dot product checks
    remaining_support_indices = [i for i in range(N_S) if i not in best_support_set]
    
    # Shuffle for randomness
    py_rng.shuffle(remaining_support_indices)
    
    for s_idx in remaining_support_indices:
        if time.time() > deadline - 10:
            break
        
        # Get the 16 vectors for this support
        mask = (all_support_indices == s_idx)
        candidates = all_vectors_D[mask]
        
        for v in candidates:
            vt = tuple(v)
            if vt in current_set:
                continue
            
            # Check against all current vectors using dot product
            # Two Type D vectors conflict if |dot| >= 3 (since both have norm 2)
            # Actually, the constraint is min pairwise distance >= max norm
            # For vectors of norm 2: distance^2 = 8 - 2*dot >= 4 => dot <= 2
            # So we need dot < 3 for all pairs (integer dot, so dot <= 2)
            
            dots = current_vectors @ v
            if (dots >= 3).any() or (dots <= -3).any():
                continue
            
            current_vectors = np.vstack([current_vectors, v])
            current_set.add(vt)

    final_vectors_D = current_vectors
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
