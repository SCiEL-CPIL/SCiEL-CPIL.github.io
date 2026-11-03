import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q != 3:
        # Fallback for non-ternary
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]
        
    if n != 7:
        # Fallback for other n, use direct sum or trivial
        if n <= 4:
             # Perfect code for n=4? Tetracode size 9
             if n == 4:
                 return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             else:
                 import itertools
                 return [list(t) for t in itertools.product(range(3), repeat=n)]
        else:
             # Direct sum construction
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             import itertools
             base_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             return base_code

    # Specific optimization for n=7, q=3, radius=1
    # Total words = 3^7 = 2187
    N = 3**n
    
    # Precompute neighbors for all words in the space
    # Map word (tuple) to int index
    def word_to_int(w):
        idx = 0
        for d in w:
            idx = idx * 3 + d
        return idx

    def int_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute neighbors (including self) for all N words
    # neighbors[i] is a list of N neighbors? No, just the 15 neighbors.
    neighbors = [None] * N
    for i in range(N):
        w = int_to_word(i)
        nb = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(3):
                if val == orig: continue
                w[pos] = val
                nb.append(word_to_int(w))
            w[pos] = orig
        neighbors[i] = nb

    # Initialize code with the 243-word direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    base_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    code_set = set(word_to_int(w) for w in base_words)
    code_list = list(code_set)
    
    # Initialize coverage counts
    cover = np.zeros(N, dtype=np.int32)
    for c in code_list:
        for nb in neighbors[c]:
            cover[nb] += 1
            
    best_size = len(code_list)
    best_code = code_list[:]
    
    random.seed(42)
    
    # Local Search Loop
    # We try to remove codewords. If stuck, we swap.
    # To avoid infinite loops or getting stuck, we track progress.
    
    # We will run multiple restarts or a long single run.
    # Given the speed, a single long run with swaps should work.
    
    stuck_count = 0
    max_swaps_without_removal = 1000
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline:
        if stuck_count > max_swaps_without_removal:
            # Restart with a fresh random order or fresh code?
            # Let's just shuffle the code list and reset stuck counter
            random.shuffle(code_list)
            stuck_count = 0
            max_swaps_without_removal = int(max_swaps_without_removal * 1.5)
            # Occasionally do a full restart
            if max_swaps_without_removal > 10000:
                 # Full restart
                 code_set = set(word_to_int(w) for w in base_words)
                 code_list = list(code_set)
                 cover = np.zeros(N, dtype=np.int32)
                 for c in code_list:
                     for nb in neighbors[c]:
                         cover[nb] += 1
                 stuck_count = 0
                 max_swaps_without_removal = 1000
                 restarts += 1
                 if restarts > max_restarts:
                     break
            continue

        # Phase 1: Try to remove
        removed = False
        # Shuffle to avoid bias
        random.shuffle(code_list)
        for i in range(len(code_list)):
            c = code_list[i]
            # Check if c can be removed
            # min(cover[nb] for nb in neighbors[c]) > 1
            # Using numpy for speed? neighbors[c] is a list.
            # cover is a numpy array.
            # min_val = np.min(cover[neighbors[c]])
            # This creates a temporary array. 
            # Alternatively:
            can_remove = True
            for nb in neighbors[c]:
                if cover[nb] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for nb in neighbors[c]:
                    cover[nb] -= 1
                code_list.pop(i)
                removed = True
                break
        
        if removed:
            stuck_count = 0
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = code_list[:]
            # After a removal, we might be able to remove more, so loop back to top
            continue
        
        # Phase 2: No removal possible. Try a swap.
        # Pick a random codeword to remove
        idx_remove = random.randint(0, len(code_list) - 1)
        c_remove = code_list[idx_remove]
        
        # Pick a random word to add
        # Bias towards words with low coverage? Or just random?
        # Random is fine.
        c_add = random.randint(0, N - 1)
        
        if c_add == c_remove:
            continue
        if c_add in code_set: # code_set needs to be updated if we remove/add
             # We are maintaining code_list, let's maintain a set for O(1) lookup
             # Actually, we don't strictly need the set if we just check list, but list is O(N).
             # Let's maintain a set `current_code_set`
             pass
        
        # Let's maintain a set for membership
        current_code_set = set(code_list)
        
        if c_add in current_code_set:
            continue
            
        # Tentative change
        # Remove c_remove
        for nb in neighbors[c_remove]:
            cover[nb] -= 1
            
        # Check validity: all cover[nb] >= 1?
        valid = True
        # We only changed neighbors of c_remove. But adding c_add changes neighbors of c_add.
        # It's easier to just apply both and check global min?
        # No, global min check is O(N) = 2187. That's fast.
        
        # Add c_add
        for nb in neighbors[c_add]:
            cover[nb] += 1
            
        # Check if any cover[nb] < 1
        # Actually, we only need to check if the removal of c_remove broke coverage
        # for any word NOT covered by c_add.
        # But c_add might cover some of the same words.
        # Simplest: check np.min(cover) >= 1
        if np.min(cover) < 1:
            # Revert
            for nb in neighbors[c_remove]:
                cover[nb] += 1
            for nb in neighbors[c_add]:
                cover[nb] -= 1
            stuck_count += 1
            continue
        else:
            # Accept swap
            code_list[idx_remove] = c_add
            current_code_set.remove(c_remove)
            current_code_set.add(c_add)
            stuck_count += 1
            # After a swap, we might be able to remove, so loop back to top
            continue
            
    # Return the best code found
    # Convert ints back to lists
    return [int_to_word(c) for c in best_code]
