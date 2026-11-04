import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Heuristic to find a large Sidon-like set in [0, N)
    # A Sidon set has distinct pairwise sums.
    # This minimizes the peak of the autoconvolution relative to the L2 norm.
    
    S = [0]
    sums = set([0]) # 0+0
    
    # Greedy construction
    # Try to add numbers in a way that avoids collisions
    # Randomized greedy might be better to avoid local minima, 
    # but deterministic is safer for reproducibility.
    # We iterate through candidates and pick the one that causes fewest collisions?
    # Or just first fit?
    
    # Let's use a simple greedy: try x from 1 to N-1, if valid, add it.
    # This might not be optimal size, but good enough.
    
    # Better: Randomized greedy with restarts?
    # Given 120s limit, we can afford a bit of search.
    
    best_S = [0]
    
    for _ in range(100): # 100 restarts
        S = [0]
        sums = set([0])
        # Shuffle candidates?
        # To get larger sets, we should pick smaller numbers first?
        # Or random?
        # Small numbers create small sums, which are more likely to collide.
        # Large numbers create large sums, less collision.
        # But we are limited to [0, N).
        
        # Let's try greedy with random order
        candidates = np.arange(1, N)
        np.random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added
            # New sums: x+x, x+s for s in S
            new_sums = [x+x] + [x+s for s in S]
            
            # Check for collisions
            valid = True
            for ns in new_sums:
                if ns in sums:
                    valid = False
                    break
            
            if valid:
                S.append(x)
                sums.update(new_sums)
                
        if len(S) > len(best_S):
            best_S = S
            
    # Create the step function
    f = [0.0] * N
    for i in best_S:
        if i < N:
            f[i] = 1.0
            
    return f
# EVOLVE-BLOCK-END
