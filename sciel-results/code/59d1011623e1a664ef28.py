import itertools
import time
import random

def construct(q, n, radius):
    # Fallback to seed logic if not the specific problem
    if not (q == 3 and n == 7 and radius == 1):
        # Generic fallback: Direct sum of tetracode
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute force for small n
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Total words in space
    total_words = 3**7
    all_words_mask = (1 << total_words) - 1
    
    # Precompute balls for all possible codewords
    # A codeword is a tuple of 7 ints in {0,1,2}
    # We map a word to an index 0..2186
    # Index = w[0]*3^6 + w[1]*3^5 + ... + w[6]
    
    def word_to_idx(w):
        idx = 0
        for i in range(7):
            idx = idx * 3 + w[i]
        return idx

    def idx_to_word(idx):
        w = [0]*7
        for i in range(6, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute ball masks
    # ball[c] is a bitmask of all words within distance 1 of c
    balls = [0] * total_words
    for c_idx in range(total_words):
        c = idx_to_word(c_idx)
        mask = 1 << c_idx
        for pos in range(7):
            orig = c[pos]
            for val in range(3):
                if val == orig:
                    continue
                c[pos] = val
                mask |= 1 << word_to_idx(c)
            c[pos] = orig
        balls[c_idx] = mask

    # Greedy Construction
    def greedy_construct():
        uncovered = all_words_mask
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            # Optimization: Only check candidates that cover at least one uncovered word?
            # With 2187 candidates, checking all is fast enough (~2187 ops per step, ~100 steps = 200k ops)
            for c in range(total_words):
                # If c is already in code, we can skip? No, greedy doesn't track code for selection, just uncovered.
                # But adding same c again is useless.
                if any(x == c for x in code): continue # This is O(N^2), might be slow. 
                # Better: just pick best c. If it's already in code, it covers 0 new words (since its ball is subset of covered).
                # Wait, if c is in code, its ball is fully covered. So popcount(ball[c] & uncovered) will be 0.
                # So we don't need to check if it's in code.
                count = (balls[c] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_count == 0:
                break # Should not happen if uncovered is not 0 and balls cover space
            code.append(best_c)
            uncovered &= ~balls[best_c]
        return code

    # Local Search: Minimize code size
    def minimize_code(code):
        # Convert to set for faster removal
        code_set = set(code)
        # Calculate coverage counts
        coverage = [0] * total_words
        for c in code_set:
            mask = balls[c]
            # Iterate set bits in mask
            m = mask
            while m:
                # Get lowest set bit index
                idx = (m & -m).bit_length() - 1
                coverage[idx] += 1
                m &= m - 1
        
        improved = True
        while improved:
            improved = False
            # Try to remove each codeword
            for c in list(code_set):
                # Check if removing c leaves any word with 0 coverage
                mask = balls[c]
                m = mask
                valid = True
                # We need to know if any word in ball[c] has coverage == 1
                # If yes, removing c makes it 0.
                # So we just need to check if min coverage in ball[c] is 1.
                # Faster: check if any word in ball[c] has coverage 1.
                # We can precompute or just iterate.
                # Iterating 15 words per check is fast.
                # We need to check words in ball[c].
                # Let's just iterate the set bits.
                temp_m = mask
                while temp_m:
                    idx = (temp_m & -temp_m).bit_length() - 1
                    if coverage[idx] == 1:
                        valid = False
                        break
                    temp_m &= temp_m - 1
                
                if valid:
                    # Remove c
                    code_set.remove(c)
                    # Update coverage
                    temp_m = mask
                    while temp_m:
                        idx = (temp_m & -temp_m).bit_length() - 1
                        coverage[idx] -= 1
                        temp_m &= temp_m - 1
                    improved = True
                    break # Restart loop with smaller set
        
        return list(code_set)

    # Iterated Local Search
    best_code = greedy_construct()
    best_len = len(best_code)
    
    # Optimization: The greedy loop above is O(N * 2187). 
    # 2187 * 200 = 400k iterations. Each is bit_count. Very fast.
    
    current_code = best_code
    current_len = best_len
    
    # Randomized Repair/Improvement
    while time.time() < deadline:
        # Perturb: remove k random codewords
        k = random.randint(3, 10)
        if len(current_code) < k:
            k = len(current_code) // 2
        
        perturbed = current_code[:]
        random.shuffle(perturbed)
        perturbed = perturbed[k:] # Remove first k
        
        # Repair: Greedily cover uncovered words
        # Calculate uncovered words
        covered_mask = 0
        for c in perturbed:
            covered_mask |= balls[c]
        uncovered = all_words_mask & ~covered_mask
        
        # Greedily add
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total_words):
                if c in perturbed: # Avoid adding duplicates if possible? 
                    # If c is already in perturbed, it covers 0 new words.
                    continue
                count = (balls[c] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_count == 0:
                break
            perturbed.append(best_c)
            uncovered &= ~balls[best_c]
            
        # Minimize
        minimized = minimize_code(perturbed)
        
        if len(minimized) < best_len:
            best_len = len(minimized)
            best_code = minimized
            current_code = minimized
        else:
            # Accept with some probability or just update current?
            # Simple ILS: accept if better or equal, else reject?
            # Or accept if better.
            # To escape local minima, we should sometimes accept worse solutions.
            # But minimizing count is the primary goal.
            # Let's just keep the best found so far and perturb from the best.
            pass

    # Convert indices back to lists
    return [idx_to_word(c) for c in best_code]
