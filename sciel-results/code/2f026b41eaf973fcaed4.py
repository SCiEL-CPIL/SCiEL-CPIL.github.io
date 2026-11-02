import itertools
import time
def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precompute
    words = []
    word_to_idx = {}
    for w in itertools.product(range(q), repeat=n):
        word_to_idx[w] = len(words)
        words.append(w)
    N = len(words)
    
    balls = [0] * N
    for i, w in enumerate(words):
        mask = 1 << i
        balls[i] = mask
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                # Create neighbor
                neigh = list(w)
                neigh[pos] = val
                neigh_idx = word_to_idx[tuple(neigh)]
                balls[i] |= (1 << neigh_idx)
                
    # 2. Heuristic Loop
    best_code = None
    best_size = float('inf')
    
    # Initial good start: Tetracode product (from seed)
    # But we will run randomized greedy.
    
    while time.time() < deadline:
        # Randomized Greedy
        uncovered = (1 << N) - 1
        code = []
        
        while uncovered:
            best_val = -1
            best_idx = -1
            # To speed up, we could maintain a priority queue, 
            # but N=2187 is small. Linear scan is 2187 * popcount.
            # popcount is fast.
            # 2187 * (iterations of greedy)
            # Greedy iterations ~ 100-200.
            # 2187 * 200 * (cost of popcount) -> very fast.
            
            # Optimization: Only check candidates that might be good?
            # No, just scan all.
            
            for i in range(N):
                if not (uncovered & (1 << i)): # If i is already covered? 
                    # Wait, we select i to cover others. 
                    # We don't skip i based on if i is covered. 
                    # We select i based on how many UNCOVERED words it covers.
                    pass
                
                # Count how many uncovered words are in ball[i]
                # val = (uncovered & balls[i]).bit_count()
                # This is the objective.
                
                # To avoid recalculating for all i every step, 
                # we can just do it.
                pass
            
            # Actually, let's just do the linear scan.
            max_cov = 0
            best_i = -1
            # Random tie breaking
            # We can iterate in random order or just pick first max.
            # Picking first max is deterministic. 
            # To randomize, we can shuffle the list of indices or use random tie-breaking.
            
            # Efficient approach:
            # We want max (uncovered & balls[i]).bit_count()
            
            # Let's just loop.
            for i in range(N):
                cov = (uncovered & balls[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_i = i
                    # Optional: random tie breaking
                    # if cov == max_cov and random.random() < 0.5: best_i = i
            if max_cov == 0:
                break # Should not happen if logic is correct
            
            code.append(best_i)
            uncovered &= ~balls[best_i]
            
        # Redundancy Removal
        # Compute cover counts
        cover_counts = [0] * N
        for c in code:
            b = balls[c]
            # Add 1 to all bits set in b
            # This is slow if done bit by bit.
            # But N=2187.
            # We can use a trick or just iterate?
            # Iterating 2187 * len(code) ~ 2187 * 200 = 400k ops. Fast.
            
            # To update cover_counts efficiently:
            # We can't easily "add 1 to bits set in b" without looping.
            # Let's just loop over set bits.
            # Or, since we do this for each c in code, and code size is small,
            # we can just do:
            # for u in range(N):
            #    if (b >> u) & 1: cover_counts[u] += 1
            # This is N * len(code).
            
            # Alternative: Keep a list of covered words for each c?
            # No, just recompute counts from scratch for the whole code?
            # That would be len(code) * N.
            # Let's do that. It's simpler and fast enough.
            
        cover_counts = [0] * N
        for c in code:
            b = balls[c]
            # Iterate set bits
            # b is an int.
            # while b:
            #    lsb = b & -b
            #    idx = lsb.bit_length() - 1
            #    cover_counts[idx] += 1
            #    b ^= lsb
            
            # This is efficient.
            
            temp_b = b
            while temp_b:
                lsb = temp_b & -temp_b
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                temp_b ^= lsb
                
        # Remove redundant
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Check if c is redundant
                # Decrement counts for balls[c]
                b = balls[c]
                temp_b = b
                redundant = True
                # We need to know if any count becomes 0
                # Store old counts? No, just check.
                # If any count in balls[c] is 1, then removing c makes it 0.
                # So c is NOT redundant if any word in its ball has count 1.
                
                # Optimization:
                # If min(cover_counts[u] for u in ball[c]) > 1, then redundant.
                
                temp_b = b
                is_redundant = True
                while temp_b:
                    lsb = temp_b & -temp_b
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        is_redundant = False
                        break
                    temp_b ^= lsb
                    
                if is_redundant:
                    # Remove c
                    # Decrement counts
                    temp_b = b
                    while temp_b:
                        lsb = temp_b & -temp_b
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        temp_b ^= lsb
                    changed = True
                else:
                    new_code.append(c)
            
            code = new_code
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Convert indices back to words
    return [words[i] for i in best_code]
