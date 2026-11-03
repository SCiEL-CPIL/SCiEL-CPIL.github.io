import numpy as np
def construct(q, n, radius):
    # 1. Generate all words and indices
    total = q ** n
    # We don't even need all_words array for the logic, just for output.
    # But we need to convert index -> word for output.
    
    # 2. Build coverage matrix (total, cov_size)
    # cov_size = 1 + n * (q - 1)
    cov_size = 1 + n * (q - 1)
    coverage_matrix = np.zeros((total, cov_size), dtype=np.int32)
    
    powers = [q ** (n - 1 - pos) for pos in range(n)]
    
    for i in range(total):
        # Get word digits from index i
        # Or just iterate i and compute neighbors
        # i is the index.
        # We need the digits to know what values to change.
        # We can extract digits of i in base q.
        digits = [0] * n
        tmp = i
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
            
        k = 0
        coverage_matrix[i, k] = i
        k += 1
        
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_idx = i + (new_val - orig_val) * powers[pos]
                coverage_matrix[i, k] = new_idx
                k += 1
                
    # 3. Multiple Greedy Passes
    best_selected = None
    best_len = total + 1
    
    # Estimate time per pass
    # Let's run a fixed number of passes, e.g., 50, and see.
    # Or run until time limit.
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    passes = 0
    while time.time() - start_time < time_limit:
        uncovered = np.ones(total, dtype=bool)
        selected = []
        
        while uncovered.any():
            # Count uncovered words covered by each candidate
            # uncovered[coverage_matrix] is boolean matrix (total, cov_size)
            # Sum along axis 1
            counts = uncovered[coverage_matrix].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
                
            # Find candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            uncovered[coverage_matrix[best_idx]] = False
            
        if len(selected) < best_len:
            best_len = len(selected)
            best_selected = selected.copy()
            
        passes += 1
        # If we found the theoretical lower bound or a very good solution, maybe stop?
        # Lower bound is 403. If we hit 403, stop.
        if best_len <= 403:
            break
            
    # 4. Convert indices to words
    codewords = []
    for idx in best_selected:
        digits = [0] * n
        tmp = idx
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
        codewords.append(digits)
        
    return codewords
