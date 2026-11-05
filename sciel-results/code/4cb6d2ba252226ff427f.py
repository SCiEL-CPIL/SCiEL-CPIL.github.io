import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        return _generic_construct(q, n, radius)
        
    space_size = 3**7 # 2187
    
    # Precompute neighbors as indices
    neighbors = []
    for idx in range(space_size):
        w = [0]*7
        tmp = idx
        for i in range(6, -1, -1):
            w[i] = tmp % 3
            tmp //= 3
        nbs = [idx]
        for pos in range(7):
            d = w[pos]
            for nd in range(3):
                if nd == d: continue
                diff = (nd - d) * (3 ** (6-pos))
                nbs.append(idx + diff)
        neighbors.append(nbs)
        
    # Precompute cover_matrix as a numpy array of uint8 for fast membership
    # cover_matrix[c, w] = 1 if w in neighbors[c]
    cover_matrix = np.zeros((space_size, space_size), dtype=np.uint8)
    for c in range(space_size):
        for w in neighbors[c]:
            cover_matrix[c, w] = 1
            
    # Greedy Set Cover
    # Using bitsets for speed in the greedy phase
    num_blocks = (space_size + 63) // 64
    cover_masks = []
    for i in range(space_size):
        blocks = [0] * num_blocks
        for nb in neighbors[i]:
            blocks[nb // 64] |= (1 << (nb % 64))
        cover_masks.append(blocks)
        
    uncovered = [(1 << min(64, space_size - b*64)) - 1 for b in range(num_blocks)]
    last_block_bits = space_size % 64
    if last_block_bits:
        uncovered[-1] = (1 << last_block_bits) - 1
        
    code = []
    
    # Incremental Greedy
    # scores[c] = number of uncovered words covered by c
    score = np.full(space_size, 15, dtype=np.int32)
    uncovered_set = set(range(space_size))
    
    # Precompute covering_codewords for fast score updates
    covering_codewords = [[] for _ in range(space_size)]
    for c in range(space_size):
        for w in neighbors[c]:
            covering_codewords[w].append(c)
            
    while uncovered_set:
        max_score = -1
        best_c = -1
        # Find max score
        # Optimization: If we had a heap, it would be faster, but scores decrease.
        # Scanning 2187 ints is ~1us. 200 steps = 200us. Negligible.
        for c in range(space_size):
            if score[c] > max_score:
                max_score = score[c]
                best_c = c
                
        if max_score == 0:
            break
            
        code.append(best_c)
        
        # Update uncovered and scores
        for w in neighbors[best_c]:
            if w in uncovered_set:
                uncovered_set.remove(w)
                for d in covering_codewords[w]:
                    score[d] -= 1
                    
        if time.time() - t_start > deadline - 10:
            break

    if uncovered_set:
        for w in uncovered_set:
            code.append(w)
            
    code = list(set(code))
    
    # Local Search
    # State:
    # in_code: bool array
    # coverage_count: int array
    in_code = np.zeros(space_size, dtype=bool)
    coverage_count = np.zeros(space_size, dtype=np.int32)
    
    for c in code:
        in_code[c] = True
        for w in neighbors[c]:
            coverage_count[w] += 1
            
    best_code = code[:]
    best_size = len(code)
    
    rng = np.random.default_rng(42)
    
    # Helper to check if c is redundant
    def is_redundant(c):
        return np.min(coverage_count[neighbors[c]]) > 1
        
    # Helper to remove c
    def remove_c(c):
        nonlocal code, in_code, coverage_count
        if c not in code:
            return False
        code.remove(c)
        in_code[c] = False
        for w in neighbors[c]:
            coverage_count[w] -= 1
        return True
        
    # Helper to add c
    def add_c(c):
        nonlocal code, in_code, coverage_count
        if in_code[c]:
            return False
        code.append(c)
        in_code[c] = True
        for w in neighbors[c]:
            coverage_count[w] += 1
        return True
        
    # Main local search loop
    while time.time() - t_start < deadline - 2:
        # 1. Try to remove redundant codewords
        removed = False
        # Iterate over a copy of code to allow removal
        for c in code[:]:
            if is_redundant(c):
                remove_c(c)
                removed = True
                if len(code) < best_size:
                    best_code = code[:]
                    best_size = len(code)
                # Restart removal loop as state changed
                break
                
        if not removed:
            # 2. Try swaps
            # Find a critical codeword (one that is not redundant)
            # A codeword c is "critical" if it covers at least one word w with coverage_count[w] == 1
            # Actually, if c is not redundant, it covers at least one word with count 1?
            # Yes, if min(count[N(c)]) == 1, then there is at least one w in N(c) with count 1.
            # If min(count[N(c)]) > 1, it is redundant.
            
            # Pick a random codeword that is not redundant
            candidates = [c for c in code if not is_redundant(c)]
            if not candidates:
                # All are redundant? Should have been removed.
                break
                
            c = candidates[rng.integers(0, len(candidates))]
            
            # Find unique words covered by c
            unique_words = [w for w in neighbors[c] if coverage_count[w] == 1]
            
            # Find valid replacements c'
            # c' must cover all unique_words
            # And c' must not be in code
            # We want to pick c' that is "good".
            # Heuristic: Maximize the number of words in N(c') that are NOT in N(c) but have count >= 1?
            # Or simply: Minimize the "damage" to other counts?
            # Let's try: Pick c' that maximizes the overlap with N(c) to keep the structure stable?
            # Or pick c' that covers the most "tight" words (count 2) to create future redundancy?
            
            best_c_prime = -1
            best_score = -1
            
            # We can't scan all 2187 c' for every c if it's slow.
            # But 2187 * 15 (check unique) * 15 (score) is ~500k ops.
            # If we do this 1000 times, it's 500M ops. Might be too slow in Python.
            # Let's limit the search for c'.
            # Or just pick the first valid c' found?
            # Random valid c' is better.
            
            # Let's collect a few valid candidates and pick one.
            valid_candidates = []
            # To speed up, we can pre-filter c' that cover unique_words[0]
            # But unique_words can be empty? No, if c is not redundant, unique_words is not empty.
            
            if unique_words:
                # Iterate through all c' not in code
                for c_prime in range(space_size):
                    if in_code[c_prime]:
                        continue
                    # Check if c' covers all unique_words
                    valid = True
                    for w in unique_words:
                        if cover_matrix[c_prime, w] == 0:
                            valid = False
                            break
                    if valid:
                        # Score c'
                        # Prefer c' that covers words with count 2?
                        # Or just add to list
                        valid_candidates.append(c_prime)
                        if len(valid_candidates) > 50:
                            break # Limit list size for speed and diversity
                
            if valid_candidates:
                c_prime = valid_candidates[rng.integers(0, len(valid_candidates))]
                # Perform swap
                remove_c(c)
                add_c(c_prime)
                
                # Check for new redundancies immediately
                # (The loop restarts and will check)
                
                if len(code) < best_size:
                    best_code = code[:]
                    best_size = len(code)
                    
    # Final verification
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        for w in neighbors[c]:
            final_cov[w] += 1
            
    if np.any(final_cov == 0):
        # Fallback to greedy code
        return [idx_to_word(c) for c in code]
    else:
        return [idx_to_word(c) for c in best_code]

def idx_to_word(idx):
    w = [0] * 7
    for i in range(6, -1, -1):
        w[i] = idx % 3
        idx //= 3
    return w

def _generic_construct(q, n, radius):
    # Simple fallback
    space_size = q ** n
    neighbors = []
    for idx in range(space_size):
        w = [0]*n
        tmp = idx
        for i in range(n-1, -1, -1):
            w[i] = tmp % q
            tmp //= q
        nbs = [idx]
        for pos in range(n):
            d = w[pos]
            for nd in range(q):
                if nd == d: continue
                diff = (nd - d) * (q ** (n-1-pos))
                nbs.append(idx + diff)
        neighbors.append(nbs)
        
    uncovered = np.ones(space_size, dtype=bool)
    code = []
    while uncovered.any():
        counts = np.zeros(space_size, dtype=np.int32)
        for c in range(space_size):
            counts[c] = np.sum(uncovered[neighbors[c]])
        best_c = np.argmax(counts)
        if counts[best_c] == 0: break
        code.append(best_c)
        uncovered[neighbors[best_c]] = False
    return [idx_to_word(c) for c in code]
