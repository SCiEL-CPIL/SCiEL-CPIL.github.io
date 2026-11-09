import numpy as np

def construct():
    N = 10_000_000
    # Precompute sigma using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    best_ratio = float('inf')
    best_a, best_b = 1, 1 # Initialize
    
    # We search for small |d| where 2a - 3b = d
    # Error = |d| / (2b)
    # To minimize error, we want large b and small |d|
    # We iterate d from -D to D
    # For each d, we find the largest b such that sigma(a) == sigma(b)
    # a = (3b + d) / 2
    
    D = 200 # Search range for d
    
    for d in range(-D, D + 1):
        # 3b + d must be even => b % 2 == (-d) % 2
        parity = (-d) % 2
        
        # Bounds for b:
        # 1 <= (3b + d) / 2 <= N
        # 2 - d <= 3b <= 2N - d
        # b_min = ceil((2 - d) / 3)
        # b_max = floor((2N - d) / 3)
        
        b_min_raw = (2 - d)
        b_max_raw = (2 * N - d)
        
        b_min = int(np.ceil(b_min_raw / 3.0))
        b_max = int(np.floor(b_max_raw / 3.0))
        
        if b_min < 1:
            b_min = 1
        if b_max > N:
            b_max = N
            
        if b_min > b_max:
            continue
            
        # Adjust b_min to match parity
        if b_min % 2 != parity:
            b_min += 1
            
        if b_min > b_max:
            continue
            
        # Create array of b values with correct parity
        # Step is 2
        B = np.arange(b_min, b_max + 1, 2, dtype=np.int64)
        
        if len(B) == 0:
            continue
            
        # Calculate A
        A = (3 * B + d) // 2
        
        # Check bounds for A (should be valid by construction, but safe)
        # A >= 1 and A <= N
        
        # Check sigma equality
        matches = sigma[A] == sigma[B]
        
        if np.any(matches):
            # We want the largest B, which corresponds to the last index
            # Find the index of the last True
            idxs = np.where(matches)[0]
            if len(idxs) > 0:
                last_idx = idxs[-1]
                b_val = B[last_idx]
                a_val = A[last_idx]
                
                # Calculate exact ratio error
                # |a/b - 1.5| = |2a - 3b| / (2b) = |d| / (2b)
                err = abs(d) / (2.0 * b_val)
                
                if err > 0 and err < best_ratio:
                    best_ratio = err
                    best_a = int(a_val)
                    best_b = int(b_val)
                    
    return [best_a, best_b]
