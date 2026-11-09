import time
from math import gcd
import bisect

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    # 1. Precompute LCM and Weights
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    # weights[i] corresponds to coefficient of 1/(i+1)
    # Sum is sum(c[i] / (i+1)) = (1/L) * sum(c[i] * L/(i+1))
    # We minimize |sum(c[i] * w[i])| where w[i] = L // (i+1)
    weights = [L // (i + 1) for i in range(30)]
    
    n1 = 15
    n2 = 15
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    # 2. Enumerate sums for first half
    # We store (sum_value, index_in_base3)
    # Index allows reconstruction: digits of index in base 3 give coeffs (0->0, 1->1, 2->-1)
    
    def enumerate_sums(weights_subset):
        n = len(weights_subset)
        total_combos = 3 ** n
        sums = [0] * total_combos
        # We can compute sums incrementally or via base-3 decoding
        # Incremental is faster.
        # Start with [0]
        # For each weight, expand the list: new_sums = [s, s+w, s-w] for s in sums
        # This creates 3^n elements.
        
        current_sums = [0]
        for w in weights_subset:
            new_sums = []
            # Pre-allocate? List append is okay.
            # To speed up, we can use list comprehension or extend
            # new_sums = []
            # for s in current_sums:
            #    new_sums.append(s)
            #    new_sums.append(s + w)
            #    new_sums.append(s - w)
            
            # Optimization: use extend
            # Note: order matters for index mapping if we want to reconstruct.
            # If we just want *a* solution, we can use backtracking on the final sum.
            # So we don't need to store the index if we use backtracking for reconstruction.
            # Backtracking is O(3^n) worst case, but usually much faster for finding *one* path.
            # Given we have the target sum, we can just do a DFS.
            
            # Let's just store the sums.
            # We will reconstruct using DFS later.
            
            # To save memory and time, let's just generate the list of sums.
            # 14M integers is ~112MB. Acceptable.
            
            # Using list multiplication/extension might be faster?
            # Let's stick to the loop but optimized.
            
            # Actually, Python list expansion is slow.
            # Let's try a different approach:
            # Generate all base-3 numbers from 0 to 3^n - 1.
            # For each, compute sum.
            # This is 14M iterations of 15 steps.
            
            # Is there a faster way?
            # NumPy?
            # Let's try to do it in pure Python first, it might be fast enough.
            
            pass # Placeholder for logic below
            
        return current_sums

    # Let's implement the enumeration properly.
    # We will generate sums for w1 and w2.
    
    def get_sums(weights_subset):
        sums = [0]
        for w in weights_subset:
            # Create new list
            # s, s+w, s-w
            # We can do:
            # new_sums = []
            # for s in sums:
            #   new_sums.append(s)
            #   new_sums.append(s+w)
            #   new_sums.append(s-w)
            # sums = new_sums
            
            # Faster:
            # Use list comprehension? No, depends on previous list.
            # Use extend?
            new_sums = [0] * (len(sums) * 3)
            # Fill it?
            # This is tricky to vectorize in pure python without numpy.
            
            # Let's just use the loop. 15 iterations of expanding list.
            # 1 -> 3 -> 9 -> ... -> 14M.
            # The last step is the most expensive (4.7M -> 14M).
            
            # Let's use a recursive generator? No, stack depth.
            
            # Let's try to use `array` or just lists.
            
            # I will use the iterative expansion.
            new_sums = []
            # Pre-allocation helps?
            # new_sums = [0] * (len(sums)*3)
            # idx = 0
            # for s in sums:
            #   new_sums[idx] = s; idx+=1
            #   new_sums[idx] = s+w; idx+=1
            #   new_sums[idx] = s-w; idx+=1
            
            # This is O(N).
            
            for s in sums:
                new_sums.append(s)
                new_sums.append(s + w)
                new_sums.append(s - w)
            sums = new_sums
        return sums

    # Check time for this step
    t0 = time.time()
    sums1 = get_sums(w1)
    t1 = time.time()
    # print(f"Sum1 gen: {t1-t0}s")
    
    if time.time() > DEADLINE:
        # Fallback
        return [0]*28 + [1, -1]
        
    sums2 = get_sums(w2)
    t2 = time.time()
    # print(f"Sum2 gen: {t2-t1}s")
    
    if time.time() > DEADLINE:
        return [0]*28 + [1, -1]

    # Sort sums2
    sums2.sort()
    t3 = time.time()
    # print(f"Sort: {t3-t2}s")
    
    # Search for best pair
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    len2 = len(sums2)
    
    # Iterate through sums1
    for s1 in sums1:
        if time.time() > DEADLINE:
            break
        target = -s1
        pos = bisect.bisect_left(sums2, target)
        
        # Check neighbors
        # We want min |s1 + s2| != 0
        candidates = []
        for p in [pos - 1, pos, pos + 1, pos - 2, pos + 2]:
            if 0 <= p < len2:
                s2 = sums2[p]
                total = s1 + s2
                if total != 0:
                    candidates.append((abs(total), total, s1, s2))
        
        if candidates:
            min_abs, min_total, cur_s1, cur_s2 = min(candidates, key=lambda x: x[0])
            if best_abs is None or min_abs < best_abs:
                best_abs = min_abs
                best_s1 = cur_s1
                best_s2 = cur_s2
                
    if best_abs is None:
        return [0]*28 + [1, -1]
        
    # Reconstruct coefficients
    # We need c1 such that sum(c1[i]*w1[i]) = best_s1
    # and c2 such that sum(c2[i]*w2[i]) = best_s2
    
    def find_coeffs(weights_subset, target):
        n = len(weights_subset)
        # Backtracking
        # Pruning: if remaining weights cannot reach target, prune.
        # Max possible sum from remaining weights: sum(abs(w[i:]) for i in range(idx, n))
        
        # Precompute suffix sums of abs weights for pruning
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i+1] + abs(weights_subset[i])
            
        def backtrack(idx, current_sum, coeffs):
            if time.time() > DEADLINE:
                return None
            if idx == n:
                if current_sum == target:
                    return list(coeffs)
                return None
            
            # Pruning
            if abs(target - current_sum) > suffix_sums[idx]:
                return None
                
            w = weights_subset[idx]
            # Try 0, 1, -1
            # Heuristic: try the one that gets closer to target first?
            # diff = target - current_sum
            # if diff > 0, try 1, 0, -1
            # if diff < 0, try -1, 0, 1
            # if diff == 0, try 0, 1, -1
            
            diff = target - current_sum
            if diff > 0:
                order = [1, 0, -1]
            elif diff < 0:
                order = [-1, 0, 1]
            else:
                order = [0, 1, -1]
                
            for c in order:
                coeffs.append(c)
                result = backtrack(idx + 1, current_sum + c * w, coeffs)
                coeffs.pop()
                if result is not None:
                    return result
            return None
            
        return backtrack(0, 0, [])
        
    coeffs1 = find_coeffs(w1, best_s1)
    coeffs2 = find_coeffs(w2, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0]*28 + [1, -1]
        
    result = coeffs1 + coeffs2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0]*28 + [1, -1]
        
    # Local Search to ensure we didn't miss a better one due to time cut in search?
    # The MIM search is exact if it completes.
    # If it was cut off by deadline, the result might not be optimal.
    # But we check deadline in the loop.
    
    # Just in case, do a few hill climbing steps
    current_total = total
    improved = True
    while improved and time.time() < DEADLINE - 2:
        improved = False
        for i in range(30):
            old_c = result[i]
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue
                new_total = current_total + (new_c - old_c) * weights[i]
                if new_total == 0:
                    continue
                if abs(new_total) < abs(current_total):
                    result[i] = new_c
                    current_total = new_total
                    improved = True
                    break
            if improved:
                break
                
    return result
