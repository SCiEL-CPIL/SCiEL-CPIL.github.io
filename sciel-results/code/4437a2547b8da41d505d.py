import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighbors) as bitmasks
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * (n + 1)
    for i in range(1, n + 1):
        powers[i] = powers[i-1] * q

    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[n - 1 - pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    orig_digits = [d[p] for p in pos_comb]
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[n - 1 - pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    def get_cover_mask(code_indices):
        mask = 0
        for c in code_indices:
            mask |= balls[c]
        return mask

    # Structured initial solution: direct sum with tetracode
    def tetracode_direct_sum():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                # Convert to index
                idx = 0
                for pos, val in enumerate(word):
                    idx += val * powers[n - 1 - pos]
                code.append(idx)
        return code

    # Constraint-aware local search
    def local_search(initial_code, max_time=120.0):
        if not initial_code:
            return initial_code
            
        current_code = initial_code[:]
        current_set = set(current_code)
        current_cover = get_cover_mask(current_code)
        current_size = len(current_code)
        
        # Track best found
        best_code = current_code[:]
        best_size = current_size
        
        # Precompute coverage counts for each word
        # cover_count[i] = number of codewords in current_code that cover word i
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            # For each word in ball[c], increment cover_count
            # Efficient way: iterate over bits set in balls[c]
            mask = balls[c]
            while mask:
                lsb = mask & (-mask)
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                mask ^= lsb
        
        steps = 0
        start_ls = time.time()
        
        while time.time() < deadline - 5.0:
            steps += 1
            
            # Choose a move: Remove or Swap
            move_type = random.random()
            
            new_code = current_code[:]
            new_set = current_set.copy()
            new_cover = current_cover
            new_size = current_size
            
            if move_type < 0.4:
                # Remove: Pick a codeword whose removal doesn't break coverage
                # Find all codewords that can be removed (i.e., every word in their ball is covered by at least one other codeword)
                removable = []
                for idx, c in enumerate(current_code):
                    # Check if all words in ball[c] have cover_count > 1
                    # This is expensive. Instead, just try removing and check validity.
                    # But we can precompute: a codeword is removable if min cover_count in its ball > 1
                    # Let's do a quick check: sample a few words from its ball
                    # Actually, let's just try all and pick a random one that works
                    pass
                
                # Alternative: pick a random codeword, try to remove it, check if still valid
                if not current_code:
                    continue
                idx_remove = random.randint(0, len(current_code) - 1)
                c_remove = current_code[idx_remove]
                
                # Check if removing c_remove keeps coverage
                # A word w is still covered if there exists another codeword c' != c_remove such that w in ball[c']
                # Equivalent: cover_count[w] > 1 for all w in ball[c_remove]
                # We can check this by: new_cover = current_cover & ~balls[c_remove], then check if new_cover == all_mask
                # But that's O(1) with bitmasks.
                new_cover = current_cover & ~balls[c_remove]
                if new_cover == all_mask:
                    # Valid removal
                    new_code.pop(idx_remove)
                    new_set.remove(c_remove)
                    new_size = current_size - 1
                else:
                    continue  # Not removable, skip this iteration
                
            else:
                # Swap: Pick one to remove, one to add
                if not current_code:
                    continue
                idx_remove = random.randint(0, len(current_code) - 1)
                c_remove = current_code[idx_remove]
                
                # Pick c_add: sample from words that are "useful"
                # Words that are currently uncovered? No, we need full coverage.
                # Words that are covered by only c_remove? Those are critical.
                # Let's just sample randomly from all words not in current_set
                c_add = random.randint(0, total_words - 1)
                if c_add in new_set:
                    continue
                
                # Apply swap
                new_code[idx_remove] = c_add
                new_set.remove(c_remove)
                new_set.add(c_add)
                
                # Update cover
                new_cover = (current_cover & ~balls[c_remove]) | balls[c_add]
                new_size = current_size
                
                # Check validity
                if new_cover != all_mask:
                    continue  # Invalid swap, skip
                
            # Valid move found
            current_code = new_code
            current_set = new_set
            current_cover = new_cover
            current_size = new_size
            
            # Update cover_count
            # We need to update cover_count for the words affected
            # This is complex. Let's recompute cover_count from scratch if size changed, or update incrementally.
            # For simplicity and correctness, recompute if size changed, else update incrementally.
            if current_size != len(best_code):
                # Size changed, recompute
                cover_count = np.zeros(total_words, dtype=np.int32)
                for c in current_code:
                    mask = balls[c]
                    while mask:
                        lsb = mask & (-mask)
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        mask ^= lsb
            
            # Update best
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        
        return best_code

    # Run structured initial + local search
    best_candidate = None
    best_size = float('inf')
    
    # Start with tetracode direct sum
    code = tetracode_direct_sum()
    
    if code:
        # Check validity
        cover = get_cover_mask(code)
        if cover == all_mask:
            # Run local search
            ls_code = local_search(code)
            if ls_code:
                ls_cover = get_cover_mask(ls_code)
                if ls_cover == all_mask and len(ls_code) < best_size:
                    best_size = len(ls_code)
                    best_candidate = ls_code
    
    # If we have time, try a greedy restart as well
    if time.time() < deadline - 60.0:
        def greedy_construct(seed):
            random.seed(seed)
            uncovered = all_mask
            code = []
            
            while uncovered:
                if time.time() > deadline - 10.0:
                    return code
                    
                best_idx = -1
                max_cover = 0
                candidates = []
                
                for i in range(total_words):
                    gain = (balls[i] & uncovered).bit_count()
                    if gain > max_cover:
                        max_cover = gain
                        best_idx = i
                        candidates = [i]
                    elif gain == max_cover and gain > 0:
                        candidates.append(i)
                        
                if max_cover == 0:
                    break
                    
                if len(candidates) > 1:
                    best_idx = random.choice(candidates)
                else:
                    best_idx = candidates[0]
                    
                code.append(best_idx)
                uncovered &= ~balls[best_idx]
                
            return code
        
        code2 = greedy_construct(42)
        if code2:
            cover2 = get_cover_mask(code2)
            if cover2 == all_mask:
                ls_code2 = local_search(code2)
                if ls_code2:
                    ls_cover2 = get_cover_mask(ls_code2)
                    if ls_cover2 == all_mask and len(ls_code2) < best_size:
                        best_size = len(ls_code2)
                        best_candidate = ls_code2

    if best_candidate is None:
        # Fallback
        return [h + list(tail) for h in [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)] for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(digits_of[i]) for i in best_candidate]
