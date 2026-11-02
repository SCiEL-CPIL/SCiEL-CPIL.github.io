"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    
    # General fallback
    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    """Optimized construction for K_3(7,1) using vectorized local search."""
    
    N = 2187
    all_indices = list(range(N))
    
    # Precompute covers as bitsets (Python ints)
    covers = [0] * N
    for i in range(N):
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * (3**pos)
                    mask |= (1 << new_i)
        covers[i] = mask
    
    full_mask = (1 << N) - 1
    
    # Precompute cover sets as lists of indices for faster iteration in local search
    cover_indices = []
    for i in range(N):
        mask = covers[i]
        idxs = []
        m = mask
        while m:
            lsb = m & -m
            idxs.append(lsb.bit_length() - 1)
            m ^= lsb
        cover_indices.append(idxs)
    
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        
        while uncovered:
            # Find best candidate - vectorize with numpy
            counts = np.array([(uncovered & covers[i]).bit_count() for i in all_indices], dtype=np.int32)
            best_count = counts.max()
            if best_count == 0:
                break
            
            # Find all indices achieving best_count
            best_mask = counts == best_count
            best_indices = [i for i, m in enumerate(best_mask) if m]
            
            if randomize and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_indices[0]
            
            selected.append(idx)
            uncovered &= ~covers[idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected, uncovered

    def local_search_remove(code, deadline):
        """
        Vectorized local search: try to remove redundant codewords.
        """
        if not code:
            return code
        
        # Use a count array for coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices_to_try = list(range(len(code)))
            random.shuffle(indices_to_try)
            
            for idx_in_code in indices_to_try:
                if idx_in_code >= len(code):
                    continue
                c = code[idx_in_code]
                
                # Check if removing c is safe: all words it covers must have count > 1
                if np.all(counts[cover_indices[c]] > 1):
                    # Remove c
                    counts[cover_indices[c]] -= 1
                    code.pop(idx_in_code)
                    changed = True
            
        return code

    def local_search_swap(code, deadline):
        """
        Try to swap a codeword with a non-codeword to potentially enable further reductions.
        Uses precomputed cover sets as Python sets for fast membership and difference.
        """
        if not code:
            return code
        
        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1
        
        # Precompute cover as set for each index (only for indices we might use)
        cover_sets = {i: set(cover_indices[i]) for i in range(N)}
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices_to_try = list(range(len(code)))
            random.shuffle(indices_to_try)
            
            for idx_in_code in indices_to_try:
                if idx_in_code >= len(code):
                    continue
                c = code[idx_in_code]
                c_set = cover_sets[c]
                
                # Find a replacement d not in code such that swapping improves or maintains coverage
                # Heuristic: try random non-codewords
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    d = random.choice(all_indices)
                    if d in code_set:
                        continue
                    
                    d_set = cover_sets[d]
                    # Words covered by c but not by d must have count > 1
                    only_c = c_set - d_set
                    if only_c:
                        # Check if any word in only_c has count == 1
                        if np.any(counts[list(only_c)] == 1):
                            continue
                    
                    # Swap is valid
                    code[idx_in_code] = d
                    code_set.remove(c)
                    code_set.add(d)
                    counts[cover_indices[c]] -= 1
                    counts[cover_indices[d]] += 1
                    changed = True
                    break
        
        return code

    best_code = None
    best_size = N
    
    # 1. Deterministic Greedy
    code, unc = greedy_cover(randomize=False)
    if not unc:
        code = local_search_remove(code, deadline)
        code = local_search_swap(code, deadline)
        code = local_search_remove(code, deadline)
        best_code = code
        best_size = len(code)
    
    # 2. Randomized Greedy + Local Search with multiple restarts
    attempts = 0
    while time.time() < deadline - 15.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts)
        if not unc:
            # Local search to reduce - do multiple passes
            code = local_search_remove(code, deadline)
            code = local_search_swap(code, deadline)
            code = local_search_remove(code, deadline)
            # Second round of swaps can find more
            code = local_search_swap(code, deadline)
            code = local_search_remove(code, deadline)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                # If we get close to known optimum, try more aggressive search
                if best_size <= 190:
                    code = local_search_swap(code, deadline)
                    code = local_search_remove(code, deadline)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code
    
    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_cover(randomize=False)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(best_code)
    
    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0]*7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)
    
    return result


def _construct_general(q, n, radius, deadline):
    """General fallback for other q, n, radius."""
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute covers
    covers = [0] * total
    for i, w in enumerate(all_words):
        mask = 1 << i
        # Generate all words within radius
        # This is expensive for large n, but for small n it's fine.
        # For general case, we might not have time.
        # Let's just do a simple greedy without precomputation for now.
        pass
    
    # Simple greedy
    uncovered = set(range(total))
    selected = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            # Count how many uncovered words are within radius of i
            cnt = 0
            for j in uncovered:
                # Hamming distance
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        # Remove covered words
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)
            
    # Ensure validity
    if uncovered:
        # Add all uncovered
        for j in uncovered:
            selected.append(j)
            
    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
