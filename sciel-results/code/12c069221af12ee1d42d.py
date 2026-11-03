import time
import random
import sys

def construct(q, n, radius):
    # q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy for general case
        total_words = q ** n
        if total_words > 10000:
             # Too large for bitmask approach in pure python quickly
             # Return a trivial valid code (all words) if too big
             return [[i // (q ** (n - 1 - j)) % q for j in range(n)] for i in range(total_words)]
        
        # Precompute balls for small N
        balls = []
        for i in range(total_words):
            mask = 0
            # Convert i to base q
            w = []
            temp = i
            for _ in range(n):
                w.append(temp % q)
                temp //= q
            w.reverse()
            
            # Self
            mask |= (1 << i)
            
            # Neighbors
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig: continue
                    w[pos] = val
                    # Convert back to index
                    idx = 0
                    for j in range(n):
                        idx = idx * q + w[j]
                    mask |= (1 << idx)
                w[pos] = orig
            balls.append(mask)
            
        all_mask = (1 << total_words) - 1
        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        deadline = start_time + 110  # 110s margin
        
        while time.time() < deadline:
            uncovered = all_mask
            code = []
            
            # Greedy
            while uncovered:
                max_count = -1
                candidates = []
                # Scan all words
                for j in range(total_words):
                    count = (uncovered & balls[j]).bit_count()
                    if count > max_count:
                        max_count = count
                        candidates = [j]
                    elif count == max_count:
                        candidates.append(j)
                
                if max_count == 0:
                    break
                    
                best_j = random.choice(candidates)
                code.append(best_j)
                uncovered &= ~balls[best_j]
                
            # Pruning
            removed = True
            while removed:
                removed = False
                for i in range(len(code) - 1, -1, -1):
                    c = code[i]
                    # Check if c is redundant
                    # Union of balls of all other words
                    others_union = 0
                    for j in range(len(code)):
                        if j == i: continue
                        others_union |= balls[code[j]]
                    
                    # If ball of c is subset of others_union, c is redundant
                    if (balls[c] & ~others_union) == 0:
                        code.pop(i)
                        removed = True
                        break # Restart pruning loop as list changed
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                
            if best_size == 486: # Known optimum for this specific instance
                break
                
        # Convert best_code (list of ints) to list of lists of digits
        result = []
        for idx in best_code:
            w = []
            temp = idx
            for _ in range(n):
                w.append(temp % q)
                temp //= q
            w.reverse()
            result.append(w)
            
        return result

    # Specific optimized path for q=3, n=8
    total_words = 3 ** 8 # 6561
    balls = [0] * total_words
    
    # Precompute balls
    for i in range(total_words):
        mask = 1 << i
        # Get digits
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        # digits are reversed, so digits[0] is last pos. 
        # Let's keep them as is for index calc: idx = d0 + d1*3 + d2*9 ...
        # Actually, my index conversion above:
        # idx = 0; for j in range(n): idx = idx * q + w[j]
        # So w[0] is most significant.
        # Let's stick to one convention.
        # Let's use standard base-3 expansion: i = d0 + d1*3 + ...
        
        # Re-do precomputation carefully
        pass
        
    # Let's just use the generic one but optimize the inner loop for q=3, n=8
    # The generic one above is fine, but let's ensure the index mapping is consistent.
    
    # Reset for the specific case to be safe and fast
    total_words = 6561
    balls = [0] * total_words
    
    # Helper to convert int to base-3 list of length n (most significant first)
    def int_to_word(idx):
        w = [0] * n
        for j in range(n):
            w[j] = idx % q
            idx //= q
        return w[::-1]
    
    def word_to_int(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    # Precompute balls
    for i in range(total_words):
        mask = 1 << i
        w = int_to_word(i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                w[pos] = val
                idx = word_to_int(w)
                mask |= (1 << idx)
            w[pos] = orig
        balls[i] = mask
        
    all_mask = (1 << total_words) - 1
    best_code = None
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Random seed for reproducibility in testing, but we want diversity
    # random.seed() is default
    
    while time.time() < deadline:
        uncovered = all_mask
        code = []
        
        # Greedy
        while uncovered:
            max_count = -1
            candidates = []
            
            # Optimization: If we are close to the end, the number of candidates with max count might be large.
            # Scanning all 6561 is O(N).
            for j in range(total_words):
                count = (uncovered & balls[j]).bit_count()
                if count > max_count:
                    max_count = count
                    candidates = [j]
                elif count == max_count:
                    candidates.append(j)
            
            if max_count == 0:
                break
                
            best_j = random.choice(candidates)
            code.append(best_j)
            uncovered &= ~balls[best_j]
            
        # Pruning
        removed = True
        while removed:
            removed = False
            # Iterate backwards to allow removal
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                others_union = 0
                for j in range(len(code)):
                    if j == i: continue
                    others_union |= balls[code[j]]
                
                if (balls[c] & ~others_union) == 0:
                    code.pop(i)
                    removed = True
                    break
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        if best_size == 486:
            break
            
    if best_code is None:
        # Fallback if something went wrong (should not happen)
        best_code = list(range(total_words))
        
    result = []
    for idx in best_code:
        result.append(int_to_word(idx))
        
    return result
