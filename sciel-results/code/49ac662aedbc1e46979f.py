import numpy as np
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for general case, though task is specific
        # Return a trivial covering if possible or empty
        return []

    size = q ** n
    
    # Precompute neighbors
    # Using vectorized base conversion for speed
    # Digits for all numbers 0 to size-1
    # We can generate neighbors more efficiently
    
    # Generate all numbers
    nums = np.arange(size)
    
    # Convert to base q digits
    # This is a bit tricky to vectorize fully without loops, but n=8 is small
    # Let's stick to the loop for precomputation as it's one-time cost
    
    def to_base_q(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    def from_base_q(digits, n, q):
        x = 0
        for i in range(n):
            x = x * q + digits[i]
        return x

    neighbors = np.zeros((size, 1 + radius * (q - 1) * n), dtype=np.int32)
    
    for x in range(size):
        digits = to_base_q(x, n, q)
        neighbors[x, 0] = x
        idx = 1
        for i in range(n):
            for val in range(q):
                if val == digits[i]:
                    continue
                new_digits = digits.copy()
                new_digits[i] = val
                neighbors[x, idx] = from_base_q(new_digits, n, q)
                idx += 1

    covered = np.zeros(size, dtype=np.int32)
    codewords = []
    
    start_time = time.time()
    deadline = 110.0
    
    # Phase 1: Greedy Set Cover
    while np.any(covered == 0):
        if time.time() - start_time > deadline - 10: # Leave 10s for reduction
            break
        
        gains = (covered[neighbors] == 0).sum(axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
            
        codewords.append(best_idx)
        covered[neighbors[best_idx]] += 1

    # Phase 2: Reduction
    # Try to remove codewords if possible
    removed_any = True
    while removed_any and (time.time() - start_time < deadline):
        removed_any = False
        # Iterate in reverse to handle removals safely? 
        # Or just iterate and skip if already removed.
        # We store indices in a list.
        
        new_codewords = []
        for c in codewords:
            # Tentatively remove c
            covered[neighbors[c]] -= 1
            if np.any(covered == 0):
                # Cannot remove
                covered[neighbors[c]] += 1
                new_codewords.append(c)
            else:
                # Can remove, so we don't add it back to new_codewords
                # and we keep covered decremented
                removed_any = True
        
        codewords = new_codewords

    # Convert back
    result = []
    for c in codewords:
        result.append(to_base_q(c, n, q))
        
    return result
