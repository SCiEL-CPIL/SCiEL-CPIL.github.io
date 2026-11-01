import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices 0..total-1
    # Word i in base-q representation (most significant digit first)
    # This matches the index calculation if we treat index as mixed-radix.
    # Let's verify:
    # base[i] = q^(n-1-i)
    # index = sum(v_i * base[i])
    # This is standard.
    
    # Precompute balls for all words
    # balls[i] is an array of indices of words within distance 1 of i
    balls = np.zeros((total, ball_size), dtype=np.int32)
    
    # Generate all words
    # Use int32 to avoid overflow issues with larger q/n, though int16 might suffice for q=3, n=7
    # But let's stick to a safe integer type.
    words = np.array([list(map(int, np.base_repr(i, q).zfill(n))) for i in range(total)], dtype=np.int32)
    
    for i in range(total):
        w = words[i]
        balls[i, 0] = i
        idx = 1
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                # New index
                new_idx = i + (v - orig) * base[coord]
                balls[i, idx] = new_idx
                idx += 1
                
    # Helper to get counts of coverage
    def get_counts(code_indices):
        counts = np.zeros(total, dtype=np.int32)
        for c in code_indices:
            counts[balls[c]] += 1
        return counts
            
    def is_valid_code(code_indices):
        counts = get_counts(code_indices)
        return np.all(counts > 0)
        
    def greedy_build(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            # Calculate new coverage for each word
            new_cov = np.sum(~covered[balls], axis=1)
            
            # Find best candidates
            max_cov = np.max(new_cov)
            if max_cov == 0:
                break # Should not happen
            
            candidates = np.where(new_cov == max_cov)[0]
            
            if randomize and len(candidates) > 1:
                # Pick randomly from top candidates
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
                
            # Add to code
            code.append(best_idx)
            # Update covered
            covered[balls[best_idx]] = True
            
        return code
        
    def reduce_code(code_set):
        """Reduces the code set by removing redundant codewords."""
        code_set = set(code_set)
        counts = np.zeros(total, dtype=np.int32)
        for c in code_set:
            counts[balls[c]] += 1
            
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                # Check if c is redundant: all points in its ball are covered by >= 2 codewords
                if np.min(counts[balls[c]]) >= 2:
                    code_set.remove(c)
                    counts[balls[c]] -= 1
                    changed = True
        return code_set

    best_code = None
    best_size = total
    
    # Run multiple greedy restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        # Alternate between deterministic and randomized greedy
        if restarts % 3 == 0:
            code = greedy_build(randomize=True, seed=restarts)
        else:
            code = greedy_build(randomize=False)
            
        # Reduce code
        code_set = reduce_code(code)
                    
        current_size = len(code_set)
        if current_size < best_size:
            best_size = current_size
            best_code = list(code_set)
            
    # Final reduction on best_code to squeeze out more
    if best_code is not None and time.time() < deadline - 5.0:
        code_set = reduce_code(best_code)
        best_size = len(code_set)
        best_code = list(code_set)
        
    # Convert indices back to words
    # We need to return list of lists of ints
    result = []
    for idx in best_code:
        # Convert index to base-q digits
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
