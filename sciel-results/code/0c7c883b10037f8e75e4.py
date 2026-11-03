"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate the 22 vectors of the form +/- 2 * e_i
    base_vectors = []
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            base_vectors.append(v)
            
    base_array = np.array(base_vectors, dtype=np.int64)
    
    # 2. Generate all weight 4 vectors (norm squared 4, entries +/-1)
    # Support is 4 indices, signs are 4 bits
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates_array = np.array(candidates, dtype=np.int64)
    
    # Helper to get signatures for a vector
    # Signature is (frozenset(triple), sign_tuple)
    # To make it hashable and fast, we can encode it as an integer or tuple
    def get_signatures(v):
        sigs = []
        # v is a list or array
        # Find non-zero indices
        nz = [i for i in range(D) if v[i] != 0]
        # nz has length 4
        for k in range(4):
            triple = tuple(sorted(nz[:k] + nz[k+1:]))
            signs_on_triple = tuple(v[i] for i in triple)
            sigs.append((triple, signs_on_triple))
        return sigs

    # Greedy selection with randomization
    best_set = []
    best_count = 0
    
    # Pre-compute signatures for all candidates to save time
    # There are 5280 candidates.
    # 5280 * 4 sigs = 21120 sigs.
    # Storing them is fine.
    cand_sigs = []
    for v in candidates:
        cand_sigs.append(get_signatures(v))
        
    # Try multiple random orders
    num_trials = 50 # Adjust based on speed
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        # Shuffle candidates
        indices = list(range(len(candidates)))
        random.shuffle(indices)
        
        used_sigs = set()
        current_set = []
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            sigs = cand_sigs[idx]
            # Check if any signature is already used
            # Optimization: check first, if used, skip. 
            # Since we need ALL to be unused, we can break early.
            conflict = False
            for sig in sigs:
                if sig in used_sigs:
                    conflict = True
                    break
            
            if not conflict:
                current_set.append(candidates[idx])
                for sig in sigs:
                    used_sigs.add(sig)
                
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set
            
    # Combine base vectors and best weight 4 set
    if best_set:
        best_array = np.array(best_set, dtype=np.int64)
        result = np.vstack((base_array, best_array))
    else:
        result = base_array
        
    return result

# EVOLVE-BLOCK-END
