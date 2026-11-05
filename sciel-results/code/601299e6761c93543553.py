"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Pre-compute all 4-subsets of [1,16]
    all_quads = list(combinations(range(1, 17), 4))
    n_quads = len(all_quads)  # 1820
    
    # Represent each set as a 16-element binary numpy array
    set_matrix = np.zeros((n_quads, 16), dtype=np.int8)
    for i, quad in enumerate(all_quads):
        for elem in quad:
            set_matrix[i, elem - 1] = 1
    
    # Precompute pairwise intersection bitmasks using numpy
    # inter_mask[i, j] = bitmask of elements in all_quads[i] ∩ all_quads[j]
    # Use broadcasting: set_matrix[:, :, None] * set_matrix[:, None, :] gives element-wise product
    # But that's 1820x1820x16 which is 53M elements, too big.
    
    # Alternative: for each element k, the sets containing k form a subset.
    # inter_mask[i,j] has bit k set iff both i and j contain element k.
    # We can compute this by: for each k, let S_k = {i : set_matrix[i,k]==1}
    # Then for all i,j in S_k, set bit k in inter_mask[i,j].
    
    inter_mask = np.zeros((n_quads, n_quads), dtype=np.int32)
    for k in range(16):
        # Get indices of sets containing element k+1
        idxs = np.where(set_matrix[:, k] == 1)[0]
        for i_idx in range(len(idxs)):
            for j_idx in range(i_idx + 1, len(idxs)):
                inter_mask[idxs[i_idx], idxs[j_idx]] |= (1 << k)
                inter_mask[idxs[j_idx], idxs[i_idx]] |= (1 << k)
    
    best_family = []
    best_size = 0
    
    # Multiple random restarts with greedy
    num_restarts = 100
    for trial in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(trial * 104729 + 7)
        order = list(range(n_quads))
        random.shuffle(order)
        
        current_idx = []  # indices of selected sets
        current_size = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            # Check if adding idx creates a sunflower with any two existing sets
            valid = True
            for a_pos in range(current_size):
                if not valid:
                    break
                a = current_idx[a_pos]
                inter_a_idx = inter_mask[a, idx]  # A ∩ S
                for b_pos in range(a_pos + 1, current_size):
                    b = current_idx[b_pos]
                    inter_b_idx = inter_mask[b, idx]  # B ∩ S
                    inter_ab = inter_mask[a, b]       # A ∩ B
                    if inter_a_idx == inter_b_idx == inter_ab:
                        valid = False
                        break
            
            if valid:
                current_idx.append(idx)
                current_size += 1
                
            if current_size > best_size:
                best_size = current_size
                best_family = current_idx[:]
                
            if best_size >= 250:
                break
    
    # Convert back to sorted lists
    result = [sorted(all_quads[i]) for i in best_family]
    return result
# EVOLVE-BLOCK-END
