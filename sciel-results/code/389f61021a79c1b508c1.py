"""Greedy set-cover for ternary covering code K_3(6,1)."""
import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # For the specific task q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total = 729
        powers = [1, 3, 9, 27, 81, 243]
        
        # Precompute neighbors as bitmasks for fast overlap counting
        # Neighbor of x: x itself, and x +/- 3^i for each digit i where valid
        neighbors = []
        for x in range(total):
            nb_set = {x}
            for i in range(n):
                digit = (x // powers[i]) % q
                if digit != 0:
                    nb_set.add(x - powers[i])
                if digit != q - 1:
                    nb_set.add(x + powers[i])
            mask = 0
            for nb in nb_set:
                mask |= (1 << nb)
            neighbors.append(mask)
        
        all_mask = (1 << total) - 1
        
        # Initial construction: Direct sum of tetracode and all words of length n-4
        # Tetracode: 9 words of length 4
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tails = list(itertools.product(range(3), repeat=n - 4))
        
        initial_codewords = []
        for h in tetracode:
            for tail in tails:
                initial_codewords.append(h + list(tail))
        
        # Convert to indices
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * 3 + val
            return idx
        
        initial_indices = [word_to_idx(w) for w in initial_codewords]
        
        # Greedy local search with iterative codeword removal and redundancy checking
        # Start from the 81-word construction and try to remove redundant codewords
        current_code = list(initial_indices)
        current_size = len(current_code)
        
        # Track coverage counts for each word in the space
        # coverage_count[i] = number of codewords whose neighborhood covers word i
        coverage_count = np.zeros(total, dtype=np.int32)
        for idx in current_code:
            # Count how many times each word in the neighborhood is covered
            # We need to iterate over the bits in the mask
            mask = neighbors[idx]
            # Efficient way: iterate over set bits
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                coverage_count[bit_pos] += 1
                m ^= lsb
        
        # Iteratively try to remove codewords that are redundant
        # A codeword is redundant if removing it doesn't leave any word uncovered
        # i.e., all words in its neighborhood have coverage_count >= 2 (or >= 1 if we check after removal)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove each codeword
            for i in range(len(current_code) - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                idx = current_code[i]
                mask = neighbors[idx]
                
                # Check if all words in this codeword's neighborhood are covered by at least one other codeword
                # i.e., coverage_count[word] >= 2 for all word in neighborhood
                # (since this codeword contributes 1 to each, if count >= 2, removing it leaves count >= 1)
                
                redundant = True
                m = mask
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    if coverage_count[bit_pos] < 2:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    # Remove this codeword
                    # Update coverage counts
                    m = mask
                    while m:
                        lsb = m & (-m)
                        bit_pos = lsb.bit_length() - 1
                        coverage_count[bit_pos] -= 1
                        m ^= lsb
                    
                    current_code.pop(i)
                    current_size -= 1
                    improved = True
                    # Continue checking from the same position since list shifted
                    if i > 0:
                        i += 1  # Re-check the new element at position i (which was i+1)
        
        # After removal, try to add codewords that cover uncovered words (if any)
        # But since we started with a valid code and only removed redundant ones, it should still be valid
        
        # Now try to improve further with greedy addition/removal
        # Or try multiple random restarts with different initial orderings
        
        best_code = current_code
        best_size = current_size
        
        # Try a few random restarts with different removal orders
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Shuffle the initial code and try removal again
            shuffled = list(initial_indices)
            random.shuffle(shuffled)
            
            # Reset coverage counts
            coverage_count = np.zeros(total, dtype=np.int32)
            for idx in shuffled:
                mask = neighbors[idx]
                m = mask
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    coverage_count[bit_pos] += 1
                    m ^= lsb
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(shuffled) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    
                    idx = shuffled[i]
                    mask = neighbors[idx]
                    
                    redundant = True
                    m = mask
                    while m:
                        lsb = m & (-m)
                        bit_pos = lsb.bit_length() - 1
                        if coverage_count[bit_pos] < 2:
                            redundant = False
                            break
                        m ^= lsb
                    
                    if redundant:
                        m = mask
                        while m:
                            lsb = m & (-m)
                            bit_pos = lsb.bit_length() - 1
                            coverage_count[bit_pos] -= 1
                            m ^= lsb
                        
                        shuffled.pop(i)
                        improved = True
                        if i > 0:
                            i += 1
            
            if len(shuffled) < best_size:
                best_size = len(shuffled)
                best_code = shuffled.copy()
        
        # Convert best_code back to word lists
        res = []
        for idx in best_code:
            word = []
            temp = idx
            for i in range(n):
                word.append(temp % q)
                temp //= q
            res.append(word)
        
        # Verify coverage is complete
        # Check that every word is within distance 1 of some codeword
        covered = np.zeros(total, dtype=bool)
        for idx in best_code:
            mask = neighbors[idx]
            m = mask
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                covered[bit_pos] = True
                m ^= lsb
        
        if not covered.all():
            # Fallback to the original 81-word construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        return res
    
    # Generic fallback for other parameters
    # Simple greedy
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    neighbors = []
    for x in range(total_words):
        nb_set = {x}
        for i in range(n):
            digit = (x // powers[i]) % q
            for delta in range(1, radius + 1):
                if digit - delta >= 0:
                    nb_set.add(x - delta * powers[i])
                if digit + delta < q:
                    nb_set.add(x + delta * powers[i])
        mask = 0
        for nb in nb_set:
            mask |= (1 << nb)
        neighbors.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words
    
    for trial in range(100):
        if time.time() > deadline:
            break
        uncovered = all_mask
        code = []
        while uncovered:
            max_cov = -1
            best_cands = []
            for w in range(total_words):
                overlap = (neighbors[w] & uncovered).bit_count()
                if overlap > max_cov:
                    max_cov = overlap
                    best_cands = [w]
                elif overlap == max_cov:
                    best_cands.append(w)
            if not best_cands:
                break
            chosen = random.choice(best_cands)
            code.append(chosen)
            uncovered &= ~neighbors[chosen]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code:
        res = []
        for idx in best_code:
            word = []
            temp = idx
            for i in range(n):
                word.append(temp % q)
                temp //= q
            res.append(word)
        return res
    else:
        return [[0]*n]
