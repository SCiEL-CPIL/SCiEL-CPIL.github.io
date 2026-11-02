import itertools
import random
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But the function signature is general.
    # We will implement a specific solver for q=3, n=6, radius=1 
    # and fall back to a simple construction for other parameters.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct product construction if parameters differ
        # This is not optimized for other parameters, just a placeholder
        # to satisfy the interface if called with different args.
        # The task specifies q=3, n=6, radius=1.
        if n <= 4:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        # Simple greedy for others
        return _greedy_cover(q, n, radius)

    # Specific solver for K_3(6,1)
    total_words = q ** n
    # Precompute balls for all words
    # word is an integer 0..728
    # ball[word] is a list of words within distance 1
    
    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n - 1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    # Precompute neighbors for each word index
    # neighbors[i] contains i and all words at distance 1
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = idx_to_word(i)
        neighbors[i].append(i)
        for d in range(n):
            original = w[d]
            for val in range(q):
                if val != original:
                    w[d] = val
                    neighbors[i].append(word_to_idx(w))
            w[d] = original

    # Greedy Initialization
    uncovered = [1] * total_words # 1 if uncovered, 0 if covered
    code = []
    
    while any(uncovered):
        best_word = -1
        best_count = -1
        for i in range(total_words):
            count = 0
            for nb in neighbors[i]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = i
            # Early stop if we found a word covering all remaining? Unlikely.
            # If best_count == sum(uncovered), we can break inner loop? No, just track max.
        
        if best_word == -1:
            break
            
        code.append(best_word)
        for nb in neighbors[best_word]:
            uncovered[nb] = 0
            
    # Local Search with Simulated Annealing / Random Restart logic
    # State: list of word indices
    # We maintain cover_count and num_uncovered for O(1) validity checks
    
    cover_count = [0] * total_words
    for c in code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    num_uncovered = sum(1 for c in cover_count if c == 0)
    
    best_code = code[:]
    best_size = len(code)
    
    start_time = time.time()
    time_limit = 110.0 # Leave margin for 120s limit
    
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
                
            move_type = random.random()
            if move_type < 0.3:
                # Try Remove
                if not code:
                    continue
                idx_to_remove = random.randint(0, len(code) - 1)
                c = code[idx_to_remove]
                
                # Remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        num_uncovered += 1
                code.pop(idx_to_remove)
                
                if num_uncovered == 0:
                    # Valid and smaller
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                else:
                    # Revert
                    code.insert(idx_to_remove, c)
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                        if cover_count[nb] == 1:
                            num_uncovered -= 1
                        
            elif move_type < 0.7:
                # Try Swap
                if not code:
                    continue
                idx_to_swap = random.randint(0, len(code) - 1)
                c_old = code[idx_to_swap]
                c_new = random.randint(0, total_words - 1)
                
                if c_old == c_new:
                    continue
                    
                # Check if c_new is already in code? 
                # If yes, it's a duplicate, which is invalid. 
                # We should pick c_new not in code.
                # For simplicity, if c_new in code, reject or pick another.
                # Let's just check validity. If c_new is in code, adding it increases count, 
                # removing c_old decreases. Net change depends on overlap.
                # But codewords must be distinct.
                if c_new in code:
                    continue

                # Apply swap
                for nb in neighbors[c_old]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        num_uncovered += 1
                for nb in neighbors[c_new]:
                    cover_count[nb] += 1
                    if cover_count[nb] == 1:
                        num_uncovered -= 1
                        
                code[idx_to_swap] = c_new
                
                if num_uncovered == 0:
                    # Valid swap
                    # Accept with probability 1 (it doesn't change size, but helps escape minima)
                    pass
                else:
                    # Revert
                    code[idx_to_swap] = c_old
                    for nb in neighbors[c_new]:
                        cover_count[nb] -= 1
                        if cover_count[nb] == 0:
                            num_uncovered += 1
                    for nb in neighbors[c_old]:
                        cover_count[nb] += 1
                        if cover_count[nb] == 1:
                            num_uncovered -= 1
                            
            else:
                # Kick: Remove one, then add random until valid
                if not code:
                    continue
                idx_to_kick = random.randint(0, len(code) - 1)
                c_kick = code[idx_to_kick]
                old_code = code[:]
                old_cover = cover_count[:]
                old_num_unc = num_uncovered
                
                # Remove
                for nb in neighbors[c_kick]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        num_uncovered += 1
                code.pop(idx_to_kick)
                
                # Add random until valid
                # Limit additions to avoid infinite loop or huge bloat
                max_adds = len(old_code) + 5 
                adds = 0
                while num_uncovered > 0 and adds < max_adds:
                    c_add = random.randint(0, total_words - 1)
                    if c_add in code:
                        continue
                    for nb in neighbors[c_add]:
                        cover_count[nb] += 1
                        if cover_count[nb] == 1:
                            num_uncovered -= 1
                    code.append(c_add)
                    adds += 1
                    
                if num_uncovered == 0:
                    # Valid
                    if len(code) < len(old_code):
                        # Accept
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                    elif len(code) == len(old_code):
                        # Accept with some probability to explore
                        if random.random() < 0.1:
                            pass # Keep
                    else:
                        # Revert
                        code = old_code
                        cover_count = old_cover
                        num_uncovered = old_num_unc
                else:
                    # Failed to cover, revert
                    code = old_code
                    cover_count = old_cover
                    num_uncovered = old_num_unc

        T *= cooling

    # Convert indices back to lists of ints
    return [idx_to_word(i) for i in best_code]

def _greedy_cover(q, n, radius):
    # Fallback greedy for other parameters
    total_words = q ** n
    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx
    def idx_to_word(idx):
        w = [0] * n
        for d in range(n - 1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w
            
    # Precompute balls
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = idx_to_word(i)
        neighbors[i].append(i)
        for d in range(n):
            original = w[d]
            for val in range(q):
                if val != original:
                    w[d] = val
                    neighbors[i].append(word_to_idx(w))
            w[d] = original
            
    uncovered = [1] * total_words
    code = []
    while any(uncovered):
        best_word = -1
        best_count = -1
        for i in range(total_words):
            count = 0
            for nb in neighbors[i]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = i
        if best_word == -1:
            break
        code.append(best_word)
        for nb in neighbors[best_word]:
            uncovered[nb] = 0
    return [idx_to_word(i) for i in code]
