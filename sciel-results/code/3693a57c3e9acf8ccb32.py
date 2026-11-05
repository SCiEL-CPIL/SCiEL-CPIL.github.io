import numpy as np
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback to simple greedy for other parameters if needed, 
        # but task is specific.
        # For safety, implement a basic greedy for other cases
        N = q ** n
        sphere_size = 1 + n * (q - 1)
        neighbors = np.empty((N, sphere_size), dtype=int)
        digits = np.zeros((N, n), dtype=int)
        for i in range(N):
            val = i
            for j in range(n):
                digits[i, j] = val % q
                val //= q
        for i in range(N):
            idx = 0
            neighbors[i, idx] = i
            idx += 1
            for pos in range(n):
                orig_val = digits[i, pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                    idx += 1
        
        covered = np.zeros(N, dtype=bool)
        result_indices = []
        while covered.sum() < N:
            gains = (~covered[neighbors]).sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            result_indices.append(best_idx)
            covered[neighbors[best_idx]] = True
        
        result_words = []
        for idx in result_indices:
            word = []
            val = idx
            for _ in range(n):
                word.append(val % q)
                val //= q
            result_words.append(word)
        return result_words

    N = 3 ** n
    # Precompute spheres for radius 1
    # Sphere size = 1 + 7 * 2 = 15
    sphere_size = 15
    neighbors = np.empty((N, sphere_size), dtype=int)
    digits = np.zeros((N, n), dtype=int)
    for i in range(N):
        val = i
        for j in range(n):
            digits[i, j] = val % q
            val //= q
    for i in range(N):
        idx = 0
        neighbors[i, idx] = i
        idx += 1
        for pos in range(n):
            orig_val = digits[i, pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                idx += 1

    def idx_to_word(idx):
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_idx(word):
        idx = 0
        for j in range(n):
            idx += word[j] * (q ** j)
        return idx

    best_code = None
    best_size = float('inf')
    
    # Try different partitions based on which 2 coordinates vary
    # There are C(7,2) = 21 pairs
    var_pairs = list(itertools.combinations(range(n), 2))
    
    for var_pos in var_pairs:
        fix_pos = [p for p in range(n) if p not in var_pos]
        
        # Generate the 3 codewords for each plane in the partition
        # A plane is defined by fixing the 'fix_pos' coordinates
        # There are 3^(n-2) = 3^5 = 243 such planes
        initial_codewords = set()
        
        # Iterate over all fixed value combinations
        for fixed_vals in itertools.product(range(q), repeat=len(fix_pos)):
            # Construct the base word for this plane
            base_word = [0] * n
            for i, pos in enumerate(fix_pos):
                base_word[pos] = fixed_vals[i]
                
            # The 3 codewords for this plane (repetition code structure on var_pos)
            # 00, 11, 22 on var_pos
            for val in range(q):
                cw = base_word.copy()
                cw[var_pos[0]] = val
                cw[var_pos[1]] = val
                idx = word_to_idx(cw)
                initial_codewords.add(idx)
                
        # Pruning phase
        # We have ~729 codewords
        # Convert to list for easier manipulation
        active_cw = list(initial_codewords)
        
        # Calculate cover counts
        cover_count = np.zeros(N, dtype=int)
        for cw in active_cw:
            cover_count[neighbors[cw]] += 1
            
        # Check if full coverage
        if not np.all(cover_count > 0):
            continue # Should not happen for this construction, but safe
            
        # Pruning loop
        changed = True
        # We can shuffle active_cw to break symmetries
        # But simple sequential pass is fast
        while changed:
            changed = False
            # Iterate through a copy or reversed list to allow removal
            # We need to be careful with indices if we remove items
            # Easier: iterate through range(len(active_cw)) and skip if removed
            # Or just build a new list
            
            new_active = []
            for i, cw in enumerate(active_cw):
                # Check if cw is redundant
                # cw is redundant if cover_count[w] > 1 for all w in neighbors[cw]
                # Note: cover_count currently includes cw
                # So we check if min(cover_count[neighbors[cw]]) > 1
                if np.min(cover_count[neighbors[cw]]) > 1:
                    # Redundant, remove it
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    # Do not add to new_active
                else:
                    new_active.append(cw)
                    
            active_cw = new_active
            
        if len(active_cw) < best_size:
            best_size = len(active_cw)
            best_code = active_cw.copy()
            
    # Convert best_code indices to words
    result_words = [idx_to_word(idx) for idx in best_code]
    return result_words
