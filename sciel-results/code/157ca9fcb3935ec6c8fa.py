import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    N = len(words)
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # 2. Precompute coverage masks
    # masks[i] is an integer bitmask where bit j is 1 if words[j] is within radius of words[i]
    masks = [0] * N
    
    # Helper to get neighbors within radius
    def get_neighbors(word):
        # For efficiency, we can generate all words within distance radius
        # For small radius, BFS or combinatorial expansion is good.
        # Here we use a simple distance check for all words if N is small, 
        # or specific generation. 
        # Given N=2187, N^2 is ~4.7M. Distance calc is cheap.
        # But let's do it smarter:
        pass
        
    # Let's just do O(N^2) distance checks for simplicity and robustness against radius changes,
    # but optimized with early exit or vectorization? No, pure Python loops are slow.
    # Let's use the neighbor generation for R=1 specifically if possible, or a generic one.
    
    # Generic neighbor generation for small radius:
    # We can iterate through all possible words and check distance.
    # To speed up, we can precompute distance? No.
    # Let's try to optimize the mask generation.
    
    # If N is small (e.g., < 10000), O(N^2) is fine.
    # 2187^2 = 4.7M.
    # Each distance check is sum(1 for a,b in zip(w1,w2) if a!=b).
    # This is ~7 operations. 4.7M * 7 = 33M ops. In Python, this is ~1-2 seconds. Acceptable.
    
    for i in range(N):
        w_i = words[i]
        mask = 0
        for j in range(N):
            w_j = words[j]
            # Calculate Hamming distance
            dist = 0
            for k in range(n):
                if w_i[k] != w_j[k]:
                    dist += 1
                    if dist > radius:
                        break
            if dist <= radius:
                mask |= (1 << j)
        masks[i] = mask
        
    # 3. Greedy Set Cover with multiple restarts
    best_code = None
    best_size = float('inf')
    
    num_restarts = 20 # Tune based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        uncovered = (1 << N) - 1
        code = []
        
        while uncovered:
            # Find word covering max uncovered points
            best_idx = -1
            best_count = -1
            
            # Randomize iteration order to escape local optima?
            # Or just iterate 0..N-1.
            # To allow different paths, we can shuffle indices?
            # Shuffling N=2187 is fast.
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                # Check if this word covers any uncovered points
                # If mask[i] & uncovered == 0, skip
                # We want max bit_count(mask[i] & uncovered)
                coverage = masks[i] & uncovered
                if coverage:
                    count = coverage.bit_count()
                    if count > best_count:
                        best_count = count
                        best_idx = i
                        # Early exit if we found a perfect cover? Unlikely.
                        
            if best_idx == -1:
                break # Should not happen if uncovered is not empty and masks are correct
                
            code.append(words[best_idx])
            uncovered &= ~masks[best_idx]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    return best_code
