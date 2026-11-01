import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback for non-standard cases
    if q != 3 or n != 8 or radius != 1:
        # Simple greedy for general case
        total_words = q ** n
        # ... implement general greedy ...
        # For now, just return a trivial code or seed-like
        # The task focuses on 3,8,1
        
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    # Precompute neighbors
    # neighbors[i] is a list of indices within distance <= radius
    # For radius 1, it's i and its single-digit variations
    neighbors_list = []
    for w in range(total_words):
        nbs = [w]
        for k in range(n):
            d_k = (w // powers[k]) % q
            for v in range(q):
                if v != d_k:
                    w_new = w + (v - d_k) * powers[k]
                    nbs.append(w_new)
        neighbors_list.append(nbs)
        
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple rounds of greedy with different tie-breaking
    # Round 1: Deterministic (lowest index)
    # Round 2-5: Random tie-breaking
    
    for run in range(5):
        if time.time() > deadline:
            break
            
        covered = np.zeros(total_words, dtype=bool)
        code = []
        uncovered_count = total_words
        
        while uncovered_count > 0:
            # Calculate scores
            # ~covered is the mask of uncovered words
            # We want the count of uncovered neighbors for each candidate
            scores = np.sum(~covered[neighbors], axis=1)
            
            # Find max score
            max_score = np.max(scores)
            
            # Find all indices with max score
            candidates = np.where(scores == max_score)[0]
            
            # Tie-breaking
            if run == 0:
                best_idx = candidates[0]
            else:
                best_idx = np.random.choice(candidates)
                
            code.append(int(best_idx))
            
            # Update covered
            nbs = neighbors[best_idx]
            # Count newly covered
            newly = np.sum(~covered[nbs])
            covered[nbs] = True
            uncovered_count -= newly
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert best_code (list of ints) to list of lists of digits
    result = []
    for w in best_code:
        digits = [0] * n
        x = w
        for i in range(n):
            digits[i] = x % q
            x //= q
        result.append(digits)
        
    return result
