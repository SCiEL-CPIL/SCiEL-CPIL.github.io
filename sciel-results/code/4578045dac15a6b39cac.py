import numpy as np
import itertools
import time
import sys

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
    # support_masks[i] is an integer bitmask of the indices in supports[i]
    support_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks.append(mask)
        
    # conflict_mask[i] is an integer bitmask of supports that conflict with i
    conflict_mask = [0] * N_S
    for i in range(N_S):
        mask_i = support_masks[i]
        for j in range(i + 1, N_S):
            mask_j = support_masks[j]
            inter = mask_i & mask_j
            if inter.bit_count() >= 3:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Simulated Annealing for Maximum Independent Set
    # State: bitmask of selected supports
    # N_S is 330, which fits in a Python int (arbitrary precision)
    
    best_size = 0
    best_mask = 0
    
    rng = np.random.RandomState(42)
    
    def greedy_init():
        # Greedy random initialization
        selected = 0
        remaining = (1 << N_S) - 1
        # Shuffle order
        order = np.arange(N_S)
        rng.shuffle(order)
        for v in order:
            if remaining & (1 << v):
                # Check if v conflicts with any selected
                # selected is a bitmask. We need to check if (conflict_mask[v] & selected) == 0
                if (conflict_mask[v] & selected) == 0:
                    selected |= (1 << v)
                    # Remove v and its neighbors from remaining
                    remaining &= ~(conflict_mask[v] | (1 << v))
        return selected

    def popcount(x):
        return bin(x).count('1')

    # SA Parameters
    T0 = 1.0
    T_end = 0.01
    num_iterations = 50000
    cooling = (T_end / T0) ** (1.0 / num_iterations)
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        current_mask = greedy_init()
        current_size = popcount(current_mask)
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 5:
                break
                
            # Generate move
            # Candidates to add: not in current, and not conflicting with any in current
            # blocked = OR of conflict_mask[v] for v in current
            # This is expensive to compute from scratch every step.
            # Optimization: Maintain a 'blocked' mask?
            # Or just compute it. N_S=330. popcount is ~30.
            # Let's compute blocked mask efficiently.
            
            # To speed up, we can just pick a random node to remove or add.
            # Add: Pick random v not in current. If (conflict_mask[v] & current_mask) == 0, add it.
            # Remove: Pick random v in current. Remove it.
            # Swap: Pick random v in current, random u not in current. If u compatible with current\{v}, swap.
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try Add
                v = rng.randint(N_S)
                if not (current_mask & (1 << v)):
                    if (conflict_mask[v] & current_mask) == 0:
                        new_mask = current_mask | (1 << v)
                        new_size = current_size + 1
                        # Always accept add
                        current_mask = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
            elif move_type < 0.7:
                # Try Remove
                # Pick random set bit
                bits = [i for i in range(N_S) if (current_mask >> i) & 1]
                if bits:
                    v = bits[rng.randint(len(bits))]
                    new_mask = current_mask & ~(1 << v)
                    new_size = current_size - 1
                    # Accept with prob exp(-1/T)
                    if rng.random() < np.exp(-1.0 / T):
                        current_mask = new_mask
                        current_size = new_size
            else:
                # Try Swap
                bits = [i for i in range(N_S) if (current_mask >> i) & 1]
                if bits:
                    v_remove = bits[rng.randint(len(bits))]
                    v_add = rng.randint(N_S)
                    if v_add == v_remove:
                        continue
                    if current_mask & (1 << v_add):
                        continue
                    # Check if v_add is compatible with current_mask without v_remove
                    temp_mask = current_mask & ~(1 << v_remove)
                    if (conflict_mask[v_add] & temp_mask) == 0:
                        new_mask = temp_mask | (1 << v_add)
                        # Size unchanged, always accept to explore
                        current_mask = new_mask
                        # current_size stays same
                        
            T *= cooling

    # Generate Type D vectors for the best MIS
    selected_indices = [i for i in range(N_S) if (best_mask >> i) & 1]
    
    vectors_D = []
    for s_idx in selected_indices:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors_D.append(v)
            
    vectors_D = np.array(vectors_D, dtype=np.int64) if vectors_D else np.zeros((0, D), dtype=np.int64)
    
    final_vectors = np.vstack([vectors_A, vectors_D])
    
    return final_vectors
