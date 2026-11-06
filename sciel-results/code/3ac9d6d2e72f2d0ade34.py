import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    if limit < 1:
        return [1]
    
    powerful = set()
    powerful.add(1)
    
    # b^3 <= limit => b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit / b^3 => a <= sqrt(limit / b^3)
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # Try different limits
    # 10^12 is the sweet spot for M^2 memory
    limits_to_try = [10**12, 10**11, 10**10]
    
    for limit in limits_to_try:
        if time.time() - start_time > deadline - 30:
            break
            
        try:
            powerful = _generate_powerful(limit)
            M = len(powerful)
            
            # If M is too large, skip to avoid memory issues
            if M > 50000:
                continue
                
            P = np.array(powerful, dtype=np.int64)
            
            # Compute all sums
            # This creates an M x M matrix
            sums = P[:, None] + P[None, :]
            
            # Find the sum with the most occurrences
            unique_sums, counts = np.unique(sums, return_counts=True)
            
            max_idx = int(np.argmax(counts))
            candidate_N = int(unique_sums[max_idx])
            candidate_count = int(counts[max_idx])
            
            # The count includes (a,b) and (b,a).
            # We want distinct pairs a <= b.
            # If candidate_N is even and candidate_N//2 is powerful, 
            # the diagonal element (N/2, N/2) is counted once.
            # Others are counted twice.
            # So actual pairs = (candidate_count - 1) // 2 + 1 if N/2 in set else candidate_count // 2
            # But it's safer to just reconstruct and count.
            
            # Reconstruct pairs
            powerful_set = set(powerful)
            pairs = []
            for a in powerful:
                b = candidate_N - a
                if a <= b and b in powerful_set:
                    pairs.append([a, b])
            
            actual_count = len(pairs)
            
            if actual_count > best_count:
                best_count = actual_count
                best_result = {'N': candidate_N, 'pairs': pairs}
                
        except MemoryError:
            continue
        except Exception as e:
            continue
            
        if time.time() - start_time > deadline - 30:
            break
            
    return best_result
