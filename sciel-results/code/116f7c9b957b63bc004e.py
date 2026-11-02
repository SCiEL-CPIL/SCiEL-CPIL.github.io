"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    # Generate candidate vectors: all integer vectors with norm^2 <= 5
    # This gives us a good pool of short vectors that are more likely to be compatible
    candidates = []
    
    # Norm^2 = 1: unit vectors and negatives
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = sign
            candidates.append(v)
    
    # Norm^2 = 2: two non-zero entries, each +/-1
    for i, j in combinations(range(D), 2):
        for si, sj in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            candidates.append(v)
    
    # Norm^2 = 3: three non-zero entries, each +/-1
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            v[k] = sk
            candidates.append(v)
    
    # Norm^2 = 4: various combinations
    # Type 1: one entry is 2 (or -2), rest 0
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * sign
            candidates.append(v)
    # Type 2: two entries are 1 or -1, two entries are 1 or -1 (four non-zero)
    for i, j, k, l in combinations(range(D), 4):
        for si, sj, sk, sl in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            v[k] = sk
            v[l] = sl
            candidates.append(v)
    # Type 3: one entry is 1, one entry is 1 (or -1), two entries are... wait, 1^2+1^2=2, need norm^2=4
    # Actually 1^2+1^2+1^2+1^2=4, already covered. 2^2=4 covered. 1^2+1^2+0+0... that's norm^2=2.
    # So norm^2=4: either 2^2 or 1^2+1^2+1^2+1^2
    
    # Norm^2 = 5: one entry is 1, one entry is 2 (1^2+2^2=5)
    for i, j in combinations(range(D), 2):
        for si, sj in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = 2 * sj
            candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute all-pairs dot products for fast compatibility checking
    # dots[i,j] = candidates[i] . candidates[j]
    dots = candidates @ candidates.T
    
    # Precompute conflict matrix: conflict[i,j] = True if vectors i and j are incompatible
    # Incompatible if 2 * dot(i,j) > min(norm_sq[i], norm_sq[j])
    min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
    conflict = 2 * dots > min_norms
    np.fill_diagonal(conflict, False)  # A vector is always compatible with itself
    
    # Use bitsets for efficient conflict checking
    # Each vector's conflicts as a bitset
    # n_cand is at most ~8000, so we need ceil(8000/64) = 125 uint64 values per row
    n_words = (n_cand + 63) // 64
    conflict_bits = np.zeros((n_cand, n_words), dtype=np.uint64)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if conflict[i, j]:
                word_idx = j // 64
                bit_idx = j % 64
                conflict_bits[i, word_idx] |= (1 << bit_idx)
                word_idx_i = i // 64
                bit_idx_i = i % 64
                conflict_bits[j, word_idx_i] |= (1 << bit_idx_i)
    
    def is_compatible_with_set(vec_idx, selected_indices):
        """Check if vec_idx is compatible with all vectors in selected_indices."""
        if not selected_indices:
            return True
        # Check conflicts using bitsets
        vec_conflicts = conflict_bits[vec_idx]
        for sel_idx in selected_indices:
            word_idx = sel_idx // 64
            bit_idx = sel_idx % 64
            if vec_conflicts[word_idx] & (1 << bit_idx):
                return False
        return True
    
    # Greedy initialization with multiple random restarts
    rng = np.random.default_rng(42)
    best_set = []
    best_count = 0
    
    # Try multiple random orderings
    for restart in range(50):
        if time.time() > deadline - 60:
            break
        perm = rng.permutation(n_cand)
        curr_set = []
        for idx in perm:
            if is_compatible_with_set(idx, curr_set):
                curr_set.append(idx)
        if len(curr_set) > best_count:
            best_count = len(curr_set)
            best_set = curr_set.copy()
    
    # Local search: try to improve by swapping
    current_set = best_set.copy()
    
    # Build lookup: for each vector in current_set, store its index
    in_set = set(current_set)
    
    # Simulated annealing / local search
    T = 1.0
    T_min = 1e-8
    alpha = 0.9995
    
    # Precompute which candidates are NOT in the current set for faster sampling
    # Use a boolean array for membership
    in_set_arr = np.zeros(n_cand, dtype=bool)
    for idx in current_set:
        in_set_arr[idx] = True
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        # Try to insert a random candidate not in the set
        # Find a random candidate not in the set
        n_tries = 0
        while True:
            idx_new = rng.integers(0, n_cand)
            if not in_set_arr[idx_new]:
                break
            n_tries += 1
            if n_tries > 10000:
                break
        
        if n_tries > 10000:
            T *= alpha
            continue
        
        if is_compatible_with_set(idx_new, current_set):
            # Accept insertion
            current_set.append(idx_new)
            in_set_arr[idx_new] = True
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_set = current_set.copy()
        else:
            # Try a swap: find a conflicting vector and swap
            # Find all conflicting vectors in the current set
            vec_conflicts = conflict_bits[idx_new]
            conflict_candidates = []
            for sel_idx in current_set:
                word_idx = sel_idx // 64
                bit_idx = sel_idx % 64
                if vec_conflicts[word_idx] & (1 << bit_idx):
                    conflict_candidates.append(sel_idx)
            
            if conflict_candidates:
                # Pick a random conflicting vector to remove
                idx_remove = conflict_candidates[rng.integers(0, len(conflict_candidates))]
                
                # Remove idx_remove from current_set
                current_set.remove(idx_remove)
                in_set_arr[idx_remove] = False
                
                # Check if idx_new is now compatible
                if is_compatible_with_set(idx_new, current_set):
                    current_set.append(idx_new)
                    in_set_arr[idx_new] = True
                    if len(current_set) > best_count:
                        best_count = len(current_set)
                        best_set = current_set.copy()
                else:
                    # Reject swap, restore
                    current_set.append(idx_remove)
                    in_set_arr[idx_remove] = True
        
        T *= alpha
    
    # Return the best set found
    return candidates[best_set]

# EVOLVE-BLOCK-END
