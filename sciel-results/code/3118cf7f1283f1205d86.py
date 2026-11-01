import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe
    
    # Fallback: Tetracode construction (729 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    fallback_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    best_code = fallback_code
    
    # Greedy search to find a smaller code
    # We represent codewords as tuples for hashing
    # Space size is 3^8 = 6561
    # We maintain a set of covered indices
    # To speed up, we precompute the neighbors of each index
    # Neighbor: change 1 position to 2 other values
    
    indices = list(range(3**n))
    neighbors = [[] for _ in indices]
    for idx in indices:
        digits = [int(d) for d in np.base_repr(idx, base=q).zfill(n)]
        # Reverse to match construct order? 
        # np.base_repr gives most significant first.
        # Our construct uses most significant first?
        # itertools.product(range(3), repeat=4) gives (0,0,0,0) to (2,2,2,2).
        # idx = d0*3^7 + d1*3^6 + ...
        # So digits[0] is d0.
        for i in range(n):
            for new_val in range(q):
                if new_val != digits[i]:
                    new_digits = digits[:]
                    new_digits[i] = new_val
                    new_idx = 0
                    for d in new_digits:
                        new_idx = new_idx * q + d
                    neighbors[idx].append(new_idx)
    
    # Greedy Set Cover
    covered = [False] * len(indices)
    remaining = set(indices)
    code = []
    
    while remaining and time.time() < deadline:
        # Find the codeword that covers the most uncovered words
        best_idx = -1
        best_count = -1
        for idx in remaining:
            count = 0
            for nb in neighbors[idx]:
                if not covered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = idx
            if time.time() > deadline:
                break
        if best_idx == -1:
            break
        code.append(best_idx)
        for nb in neighbors[best_idx]:
            covered[nb] = True
        remaining.discard(best_idx)
        # Remove from remaining any index that is now covered?
        # No, remaining is the set of candidate centers.
        # We should remove centers that are no longer useful?
        # A center is useful if it covers at least one uncovered word.
        # Since we mark covered, we can just iterate over remaining.
        
    # Convert indices back to codewords
    if len(code) < len(best_code):
        best_code = []
        for idx in code:
            digits = [int(d) for d in np.base_repr(idx, base=q).zfill(n)]
            best_code.append(digits)
            
    return best_code
