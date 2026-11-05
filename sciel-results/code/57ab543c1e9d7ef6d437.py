import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = reduce(lambda a, b: a * b // gcd(a, b), range(1, n + 1), 1)
    
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    size1 = 3 ** mid
    size2 = 3 ** (n - mid)
    
    def generate_sums_and_map(weights):
        # Returns (list_of_sums, dict_sum_to_idx)
        current_map = {0: 0}
        power = 1
        for w in weights:
            new_map = {}
            for s, idx in current_map.items():
                # digit 0 (-1)
                new_s = s - w
                new_idx = idx
                if new_s not in new_map:
                    new_map[new_s] = new_idx
                    
                # digit 1 (0)
                new_s = s
                new_idx = idx + power
                if new_s not in new_map:
                    new_map[new_s] = new_idx
                    
                # digit 2 (+1)
                new_s = s + w
                new_idx = idx + 2 * power
                if new_s not in new_map:
                    new_map[new_s] = new_idx
                    
            current_map = new_map
            power *= 3
            
        return list(current_map.keys()), current_map

    # Generate sums and maps for both halves
    sums1_vals, map1 = generate_sums_and_map(w1)
    sums1_vals.sort()
    
    if time.time() - start_time > timeout:
        return [0] * 28 + [1, -1]
        
    sums2_vals, map2 = generate_sums_and_map(w2)
    
    if time.time() - start_time > timeout:
        return [0] * 28 + [1, -1]

    best_diff = None
    best_s1 = 0
    best_s2 = 0
    
    # Iterate through half 2 sums
    for s2 in sums2_vals:
        if time.time() - start_time > timeout:
            break
            
        target = -s2
        # Find insertion point in sorted sums1
        pos = bisect.bisect_left(sums1_vals, target)
        
        # Check neighbors
        candidates = []
        if pos < size1:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
            
        for p in candidates:
            s1 = sums1_vals[p]
            diff = s1 + s2
            if diff == 0:
                continue
            if best_diff is None or abs(diff) < abs(best_diff):
                best_diff = diff
                best_s1 = s1
                best_s2 = s2
                
    if best_diff is None:
        # Fallback
        return [0] * 28 + [1, -1]
        
    # Get indices from maps
    idx1 = map1.get(best_s1)
    idx2 = map2.get(best_s2)
    
    if idx1 is None or idx2 is None:
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients
    coeffs = [0] * n
    
    t = idx1
    for i in range(mid):
        digit = t % 3
        t //= 3
        if digit == 0:
            coeffs[i] = -1
        elif digit == 2:
            coeffs[i] = 1
        else:
            coeffs[i] = 0
            
    t = idx2
    for i in range(n - mid):
        digit = t % 3
        t //= 3
        if digit == 0:
            coeffs[mid + i] = -1
        elif digit == 2:
            coeffs[mid + i] = 1
        else:
            coeffs[mid + i] = 0
            
    return coeffs
