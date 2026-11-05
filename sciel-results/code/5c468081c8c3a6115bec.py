import numpy as np
import time

def construct():
    N = 17
    MIN_D = 6
    deadline = time.time() + 55

    # 1. Generate Base Code [17, 8, 6]
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(len(f1)):
            if i + j < 17:
                vec[i+j] = f1[j]
        basis.append(vec)
        
    words_17 = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        words_17.append(word)
        
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        words_18.append(w + [parity])
        
    base_codewords = []
    for w18 in words_18:
        if w18[0] == 0:
            new_word = w18[1:]
            val = 0
            for i, bit in enumerate(new_word):
                if bit:
                    val |= (1 << i)
            base_codewords.append(val)
            
    # 2. Pre-compute bits for all candidates
    all_vals = np.arange(2**N, dtype=np.int64)
    bits_all = np.array([[(w >> i) & 1 for i in range(N)] for w in all_vals], dtype=np.uint8)
    
    # 3. Pre-compute bits for base code
    base_bits = np.array([[(w >> i) & 1 for i in range(N)] for w in base_codewords], dtype=np.uint8)
    base_weights = np.sum(base_bits, axis=1)
    all_weights = np.sum(bits_all, axis=1)
    
    # 4. Filter candidates: must be dist >= 6 from ALL base codewords
    # dist = w_cand + w_base - 2*dot
    # We need min over base codewords.
    # dots = bits_all @ base_bits.T (131072 x 256)
    dots = bits_all @ base_bits.T
    dists = all_weights[:, None] + base_weights[None, :] - 2 * dots
    min_dists_to_base = np.min(dists, axis=1)
    
    safe_indices = np.where(min_dists_to_base >= MIN_D)[0]
    
    if len(safe_indices) == 0:
        return base_codewords

    # 5. Greedy Optimization
    best_code = list(base_codewords)
    best_size = len(base_codewords)
    
    # We will try to add words from safe_indices
    # To speed up, we can also pre-compute pairwise distances between safe candidates?
    # If M is small, we can build a graph.
    # Let's check M size first.
    M = len(safe_indices)
    
    # Strategy: Randomized Greedy
    # If M is very small (< 1000), we can do pairwise checks easily.
    # If M is large, we need to be careful.
    
    # Let's assume M is manageable.
    # We will run multiple random restarts.
    
    rng = np.random.default_rng(42)
    
    # Pre-extract bits for safe candidates to avoid indexing bits_all repeatedly
    safe_bits = bits_all[safe_indices]
    safe_vals = all_vals[safe_indices]
    
    # We want to select a subset of safe_vals that is mutually dist >= 6.
    # And also dist >= 6 from base (already ensured).
    
    # Greedy approach:
    # 1. Shuffle safe_indices
    # 2. Iterate. If candidate is dist >= 6 from current_code, add it.
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        current_code_vals = list(base_codewords)
        current_code_bits = base_bits.copy()
        
        # Shuffle
        perm = rng.permutation(M)
        
        # We can potentially speed this up by noting that if we add a word,
        # it might invalidate others.
        
        # Optimization: Instead of checking against the whole current_code_bits,
        # we can maintain a list of "active" candidates and remove those that become invalid.
        # But simple check is O(K * 17). K is ~256. 17*256 = 4352 ops.
        # M is likely < 1000.
        # 1000 * 4352 = 4.3M ops per restart.
        # 100 restarts = 430M ops.
        # In Python/NumPy, this might be borderline.
        
        # Let's optimize the inner loop.
        # We can use a "candidate pool" approach.
        # Pool = list of indices into safe_bits that are still valid.
        # Initially, all M are valid.
        # When we add a word, we remove from the pool any candidate that is < 6 from the new word.
        
        # Let's implement this.
        pool = list(range(M))
        # To make removal fast, we can use a set or just filter.
        # Since M is small, filtering is fine.
        
        for idx in perm:
            if not pool:
                break
            if idx not in pool:
                continue
                
            cand_bits = safe_bits[idx]
            # Check against current_code_bits
            # dists = np.sum(current_code_bits ^ cand_bits, axis=1)
            # if np.min(dists) >= 6:
            
            # Faster:
            # weights
            # Let's just do the vectorized check.
            dists = np.sum(current_code_bits ^ cand_bits, axis=1)
            if np.min(dists) >= MIN_D:
                # Add
                current_code_vals.append(safe_vals[idx])
                current_code_bits = np.vstack([current_code_bits, cand_bits[None, :]])
                
                # Update pool: remove any candidate in pool that is < 6 from cand_bits
                # We can do this by checking all remaining pool items against cand_bits
                # This is O(|pool| * 17).
                # Since we are adding, |pool| shrinks.
                
                new_pool = []
                for p_idx in pool:
                    if p_idx == idx:
                        continue
                    p_bits = safe_bits[p_idx]
                    d = np.sum(p_bits ^ cand_bits)
                    if d >= MIN_D:
                        new_pool.append(p_idx)
                pool = new_pool
                
                if len(current_code_vals) > best_size:
                    best_size = len(current_code_vals)
                    best_code = list(current_code_vals)
                    
    return best_code
