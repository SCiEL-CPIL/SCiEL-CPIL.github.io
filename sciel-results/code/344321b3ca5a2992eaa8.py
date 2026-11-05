import numpy as np
import random
import time

def _square_nonzero_count(coeffs):
    """Compute number of nonzero terms in Q^2 given dense coefficient list of Q."""
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def construct():
    best_count = float('inf')
    best_coeffs = None
    
    def try_candidate(coeffs):
        nonlocal best_count, best_coeffs
        if any(c == 0 for c in coeffs):
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        count = _square_nonzero_count(coeffs)
        if count < best_count:
            best_count = count
            best_coeffs = list(coeffs)

    random.seed(42)
    
    # Improved Local Search
    def local_search(c, max_iter=100):
        c = list(c)
        current_count = _square_nonzero_count(c)
        for _ in range(max_iter):
            improved = False
            # Try perturbing each coefficient
            for i in range(12):
                # Generate a diverse set of candidates
                # Include small steps, large jumps, and negation
                candidates = set()
                candidates.add(c[i] - 1)
                candidates.add(c[i] + 1)
                candidates.add(-c[i])
                candidates.add(0) # Will be filtered
                candidates.add(1)
                candidates.add(-1)
                # Random large jumps
                for _ in range(5):
                    candidates.add(random.randint(-64, 64))
                    
                for new_c in candidates:
                    if new_c == 0 or new_c == c[i]:
                        continue
                    if abs(new_c) > 64:
                        continue
                    new_c_list = list(c)
                    new_c_list[i] = new_c
                    new_count = _square_nonzero_count(new_c_list)
                    if new_count < current_count:
                        c = new_c_list
                        current_count = new_count
                        improved = True
                        break
                if improved:
                    break
            
            # Try swapping two coefficients
            if not improved:
                i, j = random.sample(range(12), 2)
                if c[i] != c[j]:
                    c[i], c[j] = c[j], c[i]
                    new_count = _square_nonzero_count(c)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
            
            if not improved:
                break
        
        try_candidate(c)
        return c

    # Run multiple restarts
    for restart in range(500):
        # Initialize with random small integers
        c = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        # Occasionally use larger initial values
        if restart % 10 == 0:
             c = [random.randint(-10, 10) for _ in range(12)]
             # Ensure non-zero
             for i in range(12):
                 if c[i] == 0: c[i] = 1
        
        local_search(c)

    # Try specific patterns again
    patterns = [
        [1]*12,
        [-1]*12,
        [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
        [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
        [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
        [1,2,-2,-1,-1,2,1,-2,2,-1,1,-1],
        [1,-2,3,-4,5,-6,7,-8,9,-10,11,-12],
        [1,1,1,1,1,1,1,1,1,1,-1,-1],
        [1,1,1,1,-1,-1,-1,-1,1,1,1,1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    ]
    for pat in patterns:
        if any(c == 0 for c in pat):
            continue
        try_candidate(pat)

    # Build result
    result = [0] * 12
    for i, c in enumerate(best_coeffs):
        result[i] = c
        
    if result[-1] == 0:
        result[-1] = 1
        
    return result
