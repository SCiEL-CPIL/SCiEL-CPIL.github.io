def construct():
    import numpy as np
    import random
    import time

    def count_nonzero_square(coeffs):
        # coeffs is list in ascending degree
        # compute square
        n = len(coeffs)
        square = [0] * (2*n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(n):
                if coeffs[j] == 0:
                    continue
                square[i+j] += coeffs[i] * coeffs[j]
        return sum(1 for c in square if c != 0)
    
    best_count = float('inf')
    best_candidate = None
    
    # Heuristic candidates
    candidates = []
    # All ones
    candidates.append([1]*12)
    # Alternating
    candidates.append([1 if i%2==0 else -1 for i in range(12)])
    # Half pos, half neg
    candidates.append([1]*6 + [-1]*6)
    # Specific pattern: 1, 1, -1, -1 ...
    candidates.append([1,1,-1,-1,1,1,-1,-1,1,1,-1,-1])
    # Pattern: 1, -1, 1, -1 ...
    candidates.append([1 if i%2==0 else -1 for i in range(12)])
    
    # Add some random initial candidates
    for _ in range(100):
        # Random positions for 12 terms
        # We need exactly 12 nonzero terms.
        # Let's assume the polynomial is dense in a small window or sparse.
        # The problem says "twelve-term integer polynomials", which usually means exactly 12 non-zero coefficients.
        # The output list length <= 41.
        # So we can place the 12 non-zero terms at any positions 0..40.
        
        # Strategy: Pick 12 distinct indices from 0..40
        indices = random.sample(range(41), 12)
        indices.sort()
        coeffs = [0]*41
        for idx in indices:
            # Random coefficient in [-64, 64], non-zero
            # Small coefficients are more likely to cause exact cancellations?
            # Actually, large coefficients make exact cancellation less likely unless structured.
            # Let's try small integers first.
            coeffs[idx] = random.randint(-10, 10)
            if coeffs[idx] == 0:
                coeffs[idx] = 1
        
        # Ensure exactly 12 nonzero terms (random.randint might hit 0, we handled that, 
        # but let's be strict)
        non_zero_count = sum(1 for c in coeffs if c != 0)
        if non_zero_count != 12:
            continue
            
        candidates.append(coeffs)
        
    # Time-bounded random search
    start_time = time.time()
    timeout = 55.0 # Leave some buffer for evaluation
    
    # Local search / Random restart
    # We will mutate the best candidate found so far or generate new random ones
    
    # First evaluate initial candidates
    for c in candidates:
        if len(c) > 41:
            continue
        if sum(1 for x in c if x != 0) != 12:
            continue
        if c[-1] == 0:
            # Trim trailing zeros? The problem says "nonzero final coefficient".
            # So the last element of the list must be non-zero.
            # If we have a list of length 41 with trailing zeros, we should trim it 
            # to satisfy "nonzero final coefficient" if the evaluator checks the list directly.
            # However, the evaluator likely checks the property of the polynomial.
            # "Return coefficient list... nonzero final coefficient".
            # This implies the list itself must end with a non-zero.
            # So we must trim trailing zeros.
            while len(c) > 0 and c[-1] == 0:
                c.pop()
            if len(c) == 0:
                continue
        cnt = count_nonzero_square(c)
        if cnt < best_count:
            best_count = cnt
            best_candidate = c[:]
            
    while time.time() - start_time < timeout:
        # Generate a new random candidate
        # Mix of dense (indices close) and sparse (indices far)
        mode = random.random()
        if mode < 0.5:
            # Dense cluster
            base = random.randint(0, 29)
            indices = random.sample(range(base, base+12), 12)
        else:
            # Sparse
            indices = random.sample(range(41), 12)
            
        indices.sort()
        coeffs = [0]*41
        for idx in indices:
            # Bias towards small coefficients for cancellation
            # -1, 1, -2, 2, etc.
            if random.random() < 0.8:
                coeffs[idx] = random.choice([-1, 1, -2, 2, -3, 3])
            else:
                coeffs[idx] = random.randint(-64, 64)
                if coeffs[idx] == 0:
                    coeffs[idx] = 1
            
        # Trim trailing zeros
        while len(coeffs) > 0 and coeffs[-1] == 0:
            coeffs.pop()
            
        if len(coeffs) == 0:
            continue
            
        if sum(1 for x in coeffs if x != 0) != 12:
            continue
            
        cnt = count_nonzero_square(coeffs)
        if cnt < best_count:
            best_count = cnt
            best_candidate = coeffs[:]
            
            # Optional: Local search around best_candidate
            # Try flipping signs or changing magnitudes of a few terms
            # This might be too slow if not careful, but let's try a simple version
            # Actually, random restarts are safer and often sufficient for small N.
            
    if best_candidate is None:
        # Fallback
        return [1]*12
        
    return best_candidate
