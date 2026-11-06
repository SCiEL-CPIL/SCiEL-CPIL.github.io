import time
from fractions import Fraction
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    ONE = Fraction(1, 1)

    # Known strong candidate: 1/2+1/3+1/7+1/43 = 1805/1806, deficit 1/1806
    # Indices: 2->0, 3->1, 7->5, 43->41
    best_set = [0, 1, 5, 41]
    best_deficit = Fraction(1, 1806)

    # Strategy: Structural change to meet-in-the-middle with exact rational arithmetic.
    # We split denominators into two sets. However, since n=47 is too large for full MITM,
    # we exploit the structure: the sum must be < 1.
    # The largest terms dominate. If we include 1/2, we have 1/2 left.
    # If we include 1/2 and 1/3, we have 1/6 left.
    
    # Let's do a focused search on the "core" set of small denominators where the sum is close to 1.
    # We will enumerate subsets of {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} 
    # that sum to less than 1, and for each, try to add one or two large denominators 
    # to minimize the deficit.
    
    # Actually, a better structural approach:
    # The deficit is 1 - sum. We want to minimize this.
    # This is equivalent to maximizing the sum subject to sum < 1.
    
    # Let's use a branch and bound / DFS on the first K denominators, 
    # and for the remaining ones, we can greedily pick the largest possible 
    # that fits, or use a simple check.
    
    # Since the reciprocals decrease, if we fix a prefix, the best completion 
    # for the remaining items to get close to 1 is to pick the largest available 
    # ones that fit. But since we can pick any subset, the "largest that fit" 
    # greedy is not always optimal for the remainder, but it's a good heuristic.
    
    # However, for the tail (large denominators), the reciprocals are small.
    # If the current sum is S, we need to add terms from the tail to get as close to 1 as possible.
    # The maximum additional sum from the tail is sum(1/d for d in tail).
    # If S + max_tail_sum < 1, we should take all of them.
    # If S >= 1, invalid.
    # If S < 1 <= S + max_tail_sum, we need to find the subset of the tail that gets closest to 1-S.
    
    # The tail is denominators 17..48. There are 32 of them.
    # 2^32 is too big for brute force.
    # But note: 1/17 + ... + 1/48 is small.
    # Let's check the sum of 1/17 to 1/48.
    # H_48 - H_16. H_48 ~ ln(48)+gamma ~ 3.87+0.577 ~ 4.45. H_16 ~ 3.38. Diff ~ 1.07.
    # So the tail can sum to > 1.
    
    # Let's split: Core = 2..16 (15 items). Tail = 17..48 (32 items).
    # Enumerate all subsets of Core (2^15 = 32768).
    # For each subset C with sum S_C < 1:
    #   We want to find a subset T of Tail such that S_C + S_T < 1 and S_C + S_T is maximized.
    #   This is a subset sum problem on the tail with target 1 - S_C.
    #   Since tail size is 32, we can do meet-in-the-middle on the tail.
    #   Split Tail into T1 (16 items) and T2 (16 items).
    #   Precompute all sums of T1 and T2.
    #   For each S_C, we need to find t1 in sums_T1, t2 in sums_T2 such that t1+t2 < 1-S_C and t1+t2 is max.
    #   This can be done by sorting sums_T2 and for each t1, binary searching in sums_T2 for the largest value < 1-S_C-t1.
    
    # This is a solid structural approach.
    
    # Step 1: Precompute tail sums
    tail_denoms = denominators[15:]  # 17 to 48
    mid = len(tail_denoms) // 2
    tail1 = tail_denoms[:mid]
    tail2 = tail_denoms[mid:]
    
    sums_tail1 = []
    for mask in range(1 << len(tail1)):
        s = Fraction(0, 1)
        for i in range(len(tail1)):
            if mask & (1 << i):
                s += recip[15 + i]
        sums_tail1.append(s)
        
    sums_tail2 = []
    for mask in range(1 << len(tail2)):
        s = Fraction(0, 1)
        for i in range(len(tail2)):
            if mask & (1 << i):
                s += recip[15 + mid + i]
        sums_tail2.append(s)
        
    # Sort sums_tail2 for binary search
    # We need to retrieve the mask or indices later? No, just the value is enough for the deficit.
    # But we need to return the indices.
    # We can store (sum, mask) pairs.
    pairs_tail2 = [(s, mask) for s, mask in zip(sums_tail2, range(1 << len(tail2)))]
    pairs_tail2.sort(key=lambda x: x[0])
    vals_tail2 = [p[0] for p in pairs_tail2]
    
    # Step 2: Iterate over core subsets
    core_denoms = denominators[:15]  # 2 to 16
    core_len = len(core_denoms)
    
    best_sum = Fraction(0, 1)
    best_mask_core = 0
    best_mask_t1 = 0
    best_mask_t2 = 0
    
    for mask_core in range(1 << core_len):
        if time.time() > deadline - 10:
            break
            
        s_core = Fraction(0, 1)
        for i in range(core_len):
            if mask_core & (1 << i):
                s_core += recip[i]
                
        if s_core >= 1:
            continue
            
        # We want to maximize s_core + s_t1 + s_t2 subject to < 1
        # So we want to maximize s_t1 + s_t2 subject to < 1 - s_core
        target = 1 - s_core
        
        best_t_sum = Fraction(0, 1)
        best_t1_mask = 0
        best_t2_mask = 0
        
        for mask_t1 in range(1 << len(tail1)):
            s_t1 = sums_tail1[mask_t1]
            if s_t1 >= target:
                continue
                
            rem = target - s_t1
            
            # Binary search in vals_tail2 for largest value < rem
            # Use bisect_left on rem, then take index-1
            import bisect
            idx = bisect.bisect_left(vals_tail2, rem)
            if idx > 0:
                s_t2 = vals_tail2[idx-1]
                if s_t1 + s_t2 > best_t_sum:
                    best_t_sum = s_t1 + s_t2
                    best_t1_mask = mask_t1
                    best_t2_mask = pairs_tail2[idx-1][1]
                    
        total_sum = s_core + best_t_sum
        if total_sum > best_sum:
            best_sum = total_sum
            best_mask_core = mask_core
            best_mask_t1 = best_t1_mask
            best_mask_t2 = best_mask_t2
            
    # Reconstruct indices
    result = []
    for i in range(core_len):
        if best_mask_core & (1 << i):
            result.append(i)
            
    for i in range(len(tail1)):
        if best_mask_t1 & (1 << i):
            result.append(15 + i)
            
    for i in range(len(tail2)):
        if best_mask_t2 & (1 << i):
            result.append(15 + mid + i)
            
    # Final check
    if not result:
        result = [0, 1, 5, 41]
        
    final_sum = sum((recip[i] for i in result), Fraction(0, 1))
    if final_sum >= 1:
        # Fallback to known valid
        return [0, 1, 5, 41]
        
    return sorted(result)
