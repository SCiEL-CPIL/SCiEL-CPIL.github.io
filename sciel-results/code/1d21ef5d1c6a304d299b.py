import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    # Vectorized conflict matrix
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3
    
    # Remove self-conflicts (diagonal is True because intersection is 4)
    np.fill_diagonal(conflict, False)

    # Degree of each node
    degrees = np.sum(conflict, axis=1)

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_size = 0
    best_support_set = set()

    # Simulated Annealing for Maximum Independent Set
    def simulate_annealing(num_restarts=100, steps_per_restart=50000):
        best_local_size = 0
        best_local_set = set()
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with a greedy independent set
            # Pick nodes in order of increasing degree, with random noise
            order = np.arange(N_S)
            rng.shuffle(order)
            
            # Greedy init
            in_set = np.zeros(N_S, dtype=bool)
            blocked = np.zeros(N_S, dtype=bool)
            
            # Sort by degree + noise
            noise = rng.uniform(0, 5, size=N_S)
            scores = degrees + noise
            init_order = np.argsort(scores)
            
            for v in init_order:
                if not blocked[v]:
                    in_set[v] = True
                    blocked |= conflict[v]
            
            current_size = np.sum(in_set)
            current_set = set(np.where(in_set)[0].tolist())
            
            if current_size > best_local_size:
                best_local_size = current_size
                best_local_set = current_set.copy()
            
            # SA Phase
            T_start = 1.0
            T_end = 0.001
            alpha = (T_end / T_start) ** (1.0 / steps_per_restart)
            T = T_start
            
            # Precompute neighbors for faster access? 
            # conflict is a bool matrix.
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    break
                
                T *= alpha
                
                # Choose move
                # 1. Add (if possible)
                # 2. Remove (always possible, but decreases size)
                # 3. Swap
                
                # Find add candidates: not in set, not blocked
                # blocked is maintained? 
                # It's easier to recompute or maintain carefully.
                # Let's maintain `in_set` and check validity on the fly.
                
                move_type = py_rng.random()
                
                if move_type < 0.4:
                    # Try Add
                    # Candidates: not in set
                    not_in_set = np.where(~in_set)[0]
                    if len(not_in_set) == 0:
                        continue
                    v = not_in_set[py_rng.randint(0, len(not_in_set)-1)]
                    
                    # Check if v conflicts with any in_set
                    if np.any(conflict[v] & in_set):
                        continue # Invalid add
                    
                    # Accept Add
                    in_set[v] = True
                    current_size += 1
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_set = set(np.where(in_set)[0].tolist())
                        
                elif move_type < 0.7:
                    # Try Remove
                    if current_size == 0:
                        continue
                    in_set_idx = np.where(in_set)[0]
                    v = in_set_idx[py_rng.randint(0, len(in_set_idx)-1)]
                    
                    # Accept with prob exp(-1/T)
                    if py_rng.random() < np.exp(-1.0 / T):
                        in_set[v] = False
                        current_size -= 1
                        
                else:
                    # Try Swap
                    if current_size == 0:
                        continue
                    in_set_idx = np.where(in_set)[0]
                    v_rem = in_set_idx[py_rng.randint(0, len(in_set_idx)-1)]
                    
                    not_in_set = np.where(~in_set)[0]
                    if len(not_in_set) == 0:
                        continue
                    v_add = not_in_set[py_rng.randint(0, len(not_in_set)-1)]
                    
                    # Check if v_add conflicts with in_set \ {v_rem}
                    # conflict[v_add] & in_set
                    # We need to ignore v_rem
                    temp_in_set = in_set.copy()
                    temp_in_set[v_rem] = False
                    
                    if np.any(conflict[v_add] & temp_in_set):
                        continue # Invalid swap
                    
                    # Swap is size-neutral. 
                    # Accept with prob exp(0/T) = 1? 
                    # Or use Metropolis: delta E = 0. Always accept?
                    # To explore, we might accept "worse" swaps? 
                    # In MIS, all valid sets of same size are equivalent in energy.
                    # But structure matters for future adds.
                    # So we should accept swaps to shuffle the set.
                    # Accept with prob 1.0 or high prob.
                    if py_rng.random() < 0.8: # 80% accept to allow exploration
                        in_set[v_rem] = False
                        in_set[v_add] = True
                        # Size unchanged
        
        return best_local_size, best_local_set

    # Run SA
    best_size, best_support_set = simulate_annealing(50, 20000)
    
    # One more pass with best found so far as seed for SA?
    # Or just generate vectors from best_support_set

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    
    if selected_vectors_D:
        selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
        final_vectors = np.vstack([vectors_A, selected_vectors_D])
    else:
        final_vectors = vectors_A

    return final_vectors
