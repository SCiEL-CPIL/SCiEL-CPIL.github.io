import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    total_words = q ** n
    
    # Precompute word representations and indices
    # We assume q=3, n=7 for this specific task instance, but keep it general if possible.
    # However, the problem statement says "7 matches", so n=7, q=3.
    # The evaluator passes q, n, radius.
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute neighbor masks
    # neighbors[i] is a bitmask of indices within Hamming distance <= radius from i
    neighbors = [0] * total_words
    
    for i in range(total_words):
        w = all_words[i]
        mask = 1 << i
        
        # Generate all words within distance 'radius'
        # For small n and q, we can just iterate all words and check distance?
        # 2187 * 2187 = 4.8M distance checks.
        # Distance check: sum(1 for a,b in zip(w1,w2) if a!=b)
        # This is O(n) per pair. 4.8M * 7 = 33.6M ops. Might be slow in Python (1-2s).
        
        # Better: Generate neighbors directly.
        # For radius 1:
        if radius == 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    new_w = w.copy()
                    new_w[pos] = val
                    j = word_to_idx(new_w)
                    mask |= (1 << j)
        else:
            # General case: iterate all words
            for j in range(total_words):
                w2 = all_words[j]
                dist = 0
                for k in range(n):
                    if w[k] != w2[k]:
                        dist += 1
                        if dist > radius:
                            break
                if dist <= radius:
                    mask |= (1 << j)
                    
        neighbors[i] = mask

    def greedy_cover():
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_idx = -1
            best_gain = 0
            
            # Optimization: If we have many candidates, we can break early if gain is high?
            # No, we need max.
            for i in range(total_words):
                # Only consider if it covers something new
                # gain = (neighbors[i] & uncovered).bit_count()
                gain = (neighbors[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    # If gain is 1, we can't do better? No, we want max.
                    # If best_gain == 1, we can break? No, there might be another with 1, but we just need one.
                    # Actually, if best_gain is 1, any other with 1 is equivalent.
                    # We can break if best_gain == 1?
                    # It saves time.
                    if best_gain == 1:
                        break
                
            if best_idx == -1:
                # Should not happen if uncovered != 0
                break
                
            code.append(best_idx)
            uncovered &= ~neighbors[best_idx]
            
        return code

    def local_search(code):
        # counts[i] = number of codewords covering word i
        counts = [0] * total_words
        for c in code:
            mask = neighbors[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        # Multiple passes of removal
        for _ in range(10):
            removed_any = False
            shuffled_code = code[:]
            random.shuffle(shuffled_code)
            
            for c in shuffled_code:
                mask = neighbors[c]
                m = mask
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                
                if not essential:
                    code.remove(c)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    removed_any = True
            
            if not removed_any:
                break
        
        return code

    # Initial Greedy
    best_code = greedy_cover()
    best_code = local_search(best_code)
    best_size = len(best_code)
    
    # Restart loop
    while time.time() < deadline:
        # Randomized Greedy
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_idx = -1
            best_gain = 0
            candidates = []
            
            for i in range(total_words):
                gain = (neighbors[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
                
                if best_gain == 1:
                    break
            
            if candidates:
                best_idx = random.choice(candidates)
            else:
                best_idx = 0
            
            code.append(best_idx)
            uncovered &= ~neighbors[best_idx]
        
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # If we hit a very good size, we can tighten the deadline or just keep going
            # 186 is the target.
            if best_size <= 186:
                break

    result = [all_words[i] for i in best_code]
    return result
