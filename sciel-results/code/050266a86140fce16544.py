import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (though task specifies 3,8,1)
        # Use a greedy or seed-like approach
        return seed_construct(q, n, radius)

    # 1. Generate Ternary Golay Code
    # Generator Matrix (6x11)
    G = [
        [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 1, 2, 1, 2, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 2, 1, 2],
        [0, 0, 0, 1, 0, 0, 2, 1, 1, 2, 1],
        [0, 0, 0, 0, 1, 0, 1, 2, 2, 1, 1],
        [0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 2]
    ]
    
    codewords = []
    for coeffs in itertools.product(range(3), repeat=6):
        cw = [0]*11
        for i in range(6):
            c = coeffs[i]
            if c == 1:
                for j in range(11):
                    cw[j] = (cw[j] + G[i][j]) % 3
            elif c == 2:
                for j in range(11):
                    cw[j] = (cw[j] + 2*G[i][j]) % 3
        codewords.append(cw)
        
    # 2. Project to 8 coordinates
    # Drop indices 0, 1, 2
    keep_indices = [3, 4, 5, 6, 7, 8, 9, 10]
    projected = set()
    for cw in codewords:
        proj = tuple(cw[i] for i in keep_indices)
        projected.add(proj)
        
    result = [list(p) for p in projected]
    
    # 3. Verify Coverage
    # Check if every word in {0,1,2}^8 is within dist 1 of some codeword
    covered = [False] * (3**8)
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx
        
    # Precompute neighbors for each codeword?
    # Or just iterate all words and check against codewords?
    # 6561 words * 486 codewords * 8 dist = ~25M ops.
    
    # Optimization: For each codeword, mark all words within dist 1 as covered.
    # Number of words within dist 1 of a center: 1 + 8*2 = 17.
    # 486 codewords * 17 neighbors = ~8262 marks.
    # This is MUCH faster than checking all 6561 words against all 486 codewords.
    
    covered = [False] * (3**8)
    for cw in result:
        # Mark cw
        idx = word_to_idx(cw)
        covered[idx] = True
        
        # Mark neighbors (dist 1)
        for pos in range(8):
            orig_val = cw[pos]
            for new_val in range(3):
                if new_val == orig_val: continue
                # Create neighbor
                # Instead of creating list, compute index directly?
                # Or just create list, it's small.
                neighbor = cw[:]
                neighbor[pos] = new_val
                n_idx = word_to_idx(neighbor)
                covered[n_idx] = True
                
    if not all(covered):
        # Fallback if construction failed (should not happen)
        return seed_construct(q, n, radius)
        
    return result

def seed_construct(q, n, radius):
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
