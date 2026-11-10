import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Calculate LCM of 2..48
    L = 1
    for i in range(2, 49):
        L = L * i // gcd(L, i)
        
    # 2. Weights
    # Indices 0..46 correspond to denominators 2..48
    weights = [L // (i + 2) for i in range(47)]
    
    # Split
    mid = 23
    weights_A = weights[:mid]
    weights_B = weights[mid:]
    
    # 3. Generate (sum, mask) pairs
    def gen_sums_masks(weights_list):
        # Returns list of (sum, mask) tuples
        results = [(0, 0)]
        n = len(weights_list)
        for k in range(n):
            w = weights_list[k]
            bit = 1 << k
            new_results = []
            for s, m in results:
                ns = s + w
                if ns < L:
                    new_results.append((ns, m | bit))
            results.extend(new_results)
            if time.time() > deadline:
                break
        return results

    results_A = gen_sums_masks(weights_A)
    if time.time() > deadline:
        # Fallback: return a valid subset
        return [0]
    
    results_B = gen_sums_masks(weights_B)
    if time.time() > deadline:
        return [0]
    
    # 4. Sort by sum
    results_A.sort(key=lambda x: x[0])
    results_B.sort(key=lambda x: x[0])
    
    # 5. Two-pointer search
    best_sum = -1
    best_mask_A = 0
    best_mask_B = 0
    
    i = 0
    j = len(results_B) - 1
    
    while i < len(results_A) and j >= 0:
        current_sum = results_A[i][0] + results_B[j][0]
        if current_sum < L:
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask_A = results_A[i][1]
                best_mask_B = results_B[j][1]
            i += 1
        else:
            j -= 1
    
    # 6. Construct indices
    indices = []
    for k in range(len(weights_A)):
        if best_mask_A & (1 << k):
            indices.append(k)
            
    for k in range(len(weights_B)):
        if best_mask_B & (1 << k):
            indices.append(mid + k)
            
    if not indices:
        indices = [0]
        
    return sorted(indices)
