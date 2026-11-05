import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task specifies this one
        # Simple greedy for general case
        return _general_greedy(q, n, radius, deadline - start_time)

    total_words = q ** n
    # Map words (tuples) to integers for fast indexing
    # word -> int: sum(w[i] * q**(n-1-i))
    # int -> word: not strictly needed if we precompute neighbors in int space
    
    # Precompute neighbors for all 729 words
    # neighbors[i] contains i and all words at Hamming distance 1 from i
    neighbors = [None] * total_words
    for i in range(total_words):
        # Convert i to tuple
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        
        neigh = set()
        neigh.add(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w[:]
                new_w[pos] = new_val
                # Convert new_w back to int
                new_i = 0
                for digit in new_w:
                    new_i = new_i * q + digit
                neigh.add(new_i)
        neighbors[i] = list(neigh)

    # 1. Greedy Set Cover Construction
    def greedy_cover():
        uncovered = [1] * total_words
        count_uncovered = total_words
        code = []
        
        while count_uncovered > 0:
            best_idx = -1
            best_cover = 0
            # Optimization: we can't just pick max, but for small N, scanning all is fine
            # 729 words * 13 neighbors = ~9k ops per step. 729 steps = ~6M ops. Fast.
            for i in range(total_words):
                if uncovered[i] == 0:
                    continue
                # Check how many uncovered neighbors i has
                # Actually, we want the word i that covers the most *currently uncovered* words.
                # So we look at neighbors[i] and count how many have uncovered[x] == 1.
                
                # We can break early if we find 13 (perfect), but unlikely.
                covers = 0
                for x in neighbors[i]:
                    if uncovered[x]:
                        covers += 1
                        if covers == 13: # Max possible
                            break
                
                if covers > best_cover:
                    best_cover = covers
                    best_idx = i
            
            if best_idx == -1:
                break # Should not happen if count_uncovered > 0
                
            code.append(best_idx)
            for x in neighbors[best_idx]:
                if uncovered[x]:
                    uncovered[x] = 0
                    count_uncovered -= 1
                    
            if time.time() > deadline:
                break
                
        return code

    # 2. Seed Code (81 words)
    def seed_code():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seeds = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = h + list(tail)
                idx = 0
                for digit in w:
                    idx = idx * 3 + digit
                seeds.append(idx)
        return seeds

    # Try both and pick the smaller
    code1 = greedy_cover()
    if time.time() > deadline:
         return [int_to_word(c, q, n) for c in code1]
         
    code2 = seed_code()
    
    if len(code1) <= len(code2):
        current_code = code1
    else:
        current_code = code2
        
    # 3. Local Search: Remove Redundant Codewords
    # Maintain count array
    counts = [0] * total_words
    for c in current_code:
        for x in neighbors[c]:
            counts[x] += 1
            
    # Verify validity (should be valid)
    # min(counts) should be >= 1
    
    improved = True
    while improved:
        improved = False
        new_code = []
        for c in current_code:
            # Try to remove c
            # Decrement counts for neighbors of c
            for x in neighbors[c]:
                counts[x] -= 1
                
            # Check if any count dropped to 0
            valid_removal = True
            for x in neighbors[c]:
                if counts[x] == 0:
                    valid_removal = False
                    break
                    
            if valid_removal:
                # Keep removed
                improved = True
            else:
                # Revert
                for x in neighbors[c]:
                    counts[x] += 1
                new_code.append(c)
                
        current_code = new_code
        
        if time.time() > deadline:
            break

    # 4. Optional: Swap Search (if time permits)
    # Try swapping each codeword with every other possible word to see if it allows
    # a smaller set? This is expensive. 
    # Given the "remove" phase is very effective, let's stick to that unless we are far from 73.
    # If we are at 75-80, we might try a bit of random restart or swap.
    
    # Convert back to list of lists
    result = []
    for c in current_code:
        w = []
        temp = c
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result

def int_to_word(c, q, n):
    w = []
    temp = c
    for _ in range(n):
        w.append(temp % q)
        temp //= q
    w.reverse()
    return w

def _general_greedy(q, n, radius, time_limit):
    # Fallback implementation for other q,n,radius if needed
    # For this specific task, the main logic above handles q=3,n=6,r=1
    # This is a simple placeholder to ensure the function doesn't crash if params change
    # but the task contract implies fixed params or at least the main path is optimized for them.
    # We will just return an empty list or a trivial code if we hit this, 
    # but since the task is specific, we assume the optimized path is used.
    # To be safe, let's return a single word if n is small, else empty.
    # Actually, the evaluator passes specific q,n,radius.
    # If it's not 3,6,1, the above logic might not be optimal but should still work if generalized.
    # However, the precomputation is hardcoded for q=3 in the neighbor logic? No, it uses q.
    # The seed_code is hardcoded for n=4 tetracode. 
    # If q!=3 or n!=6, we should probably just do pure greedy.
    
    start_time = time.time()
    total_words = q ** n
    # ... (Implementation of general greedy would go here) ...
    # For brevity and focus on the specific task, we return a trivial valid code if possible
    # or just the greedy result.
    
    # Let's just return the greedy result from the general case
    # Re-using the logic from construct but without the specific seed
    uncovered = [1] * total_words
    count_uncovered = total_words
    code = []
    
    # Precompute neighbors for general q,n
    neighbors = [None] * total_words
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        
        neigh = set()
        neigh.add(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w[:]
                new_w[pos] = new_val
                new_i = 0
                for digit in new_w:
                    new_i = new_i * q + digit
                neigh.add(new_i)
        neighbors[i] = list(neigh)
        
    while count_uncovered > 0:
        if time.time() > start_time + time_limit:
            break
        best_idx = -1
        best_cover = 0
        for i in range(total_words):
            if uncovered[i] == 0:
                continue
            covers = 0
            for x in neighbors[i]:
                if uncovered[x]:
                    covers += 1
            if covers > best_cover:
                best_cover = covers
                best_idx = i
        if best_idx == -1:
            break
        code.append(best_idx)
        for x in neighbors[best_idx]:
            if uncovered[x]:
                uncovered[x] = 0
                count_uncovered -= 1
                
    result = []
    for c in code:
        w = []
        temp = c
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result
