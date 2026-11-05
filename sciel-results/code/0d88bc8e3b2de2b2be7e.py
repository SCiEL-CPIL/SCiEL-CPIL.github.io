import time
from math import gcd
from functools import reduce
import bisect
from fractions import Fraction

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses a meet-in-the-middle approach with exact integer arithmetic for the core search,
    followed by local search refinements.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators to work with integers
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    # Values in integer scale: V[i] = L // denominators[i]
    V = [L // d for d in denominators]
    
    # Split into two halves for meet-in-the-middle
    # n=47. Let's split 23 and 24.
    n1 = 23
    n2 = n - n1
    V1 = V[:n1]
    V2 = V[n1:]
    
    # Generate all subset sums and masks for V1
    # There are 2^23 = 8,388,608 subsets. This is manageable.
    def gen_sums(vals):
        sums = [0]
        masks = [0]
        for i, v in enumerate(vals):
            if time.time() > deadline:
                return sums, masks
            # Optimization: only extend if we have time
            new_sums = [s + v for s in sums]
            new_masks = [m | (1 << i) for m in masks]
            sums.extend(new_sums)
            masks.extend(new_masks)
        return sums, masks

    sums1, masks1 = gen_sums(V1)
    
    if time.time() > deadline:
        return [0] 

    sums2, masks2 = gen_sums(V2)
    
    if time.time() > deadline:
        return [0]

    # Sort sums2 and masks2 together by sum
    pairs2 = list(zip(sums2, masks2))
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sorted sums for bisect
    sorted_sums2 = [p[0] for p in pairs2]
    
    best_total = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through sums1
    for i in range(len(sums1)):
        s1 = sums1[i]
        if s1 >= L:
            continue
            
        # We want s2 such that s1 + s2 < L
        # s2 < L - s1
        # s2 <= L - 1 - s1
        target = L - 1 - s1
        
        # Find rightmost index in sorted_sums2 <= target
        idx = bisect.bisect_right(sorted_sums2, target)
        
        if idx > 0:
            s2 = sorted_sums2[idx - 1]
            m2 = pairs2[idx - 1][1]
            
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_mask1 = masks1[i]
                best_mask2 = m2
                
        if time.time() > deadline:
            break

    # Reconstruct indices
    indices = []
    for j in range(n1):
        if best_mask1 & (1 << j):
            indices.append(j)
            
    for j in range(n2):
        if best_mask2 & (1 << j):
            indices.append(n1 + j)
            
    if not indices:
        return [0]

    # Local Search Refinement
    # The meet-in-the-middle finds the global optimum for the partitioned problem,
    # which is the global optimum for the whole problem since it's a subset sum problem.
    # However, due to potential floating point or integer overflow issues (though LCM is huge, Python handles big ints),
    # or if the search was cut short, we can try to improve.
    # Actually, meet-in-the-middle with full enumeration of both halves IS exact.
    # 2^23 * log(2^24) operations is roughly 8M * 24 = 192M ops. In Python this might be slow.
    # Let's check if it finished in time. If it did, the result is optimal.
    # If it didn't, we have a partial result.
    
    # Let's verify the result is strictly below 1 and try to improve with local search if time permits.
    current_set = set(indices)
    current_sum_frac = Fraction(0, 1)
    for idx in indices:
        current_sum_frac += Fraction(1, denominators[idx])
    
    if current_sum_frac >= 1:
        # Should not happen if logic is correct, but fallback
        # Find a valid subset
        return [0]
        
    best_sum_frac = current_sum_frac
    best_indices = indices[:]

    # Local search: try to swap or add/remove
    # Since MITM is exact if it completes, this is just a safety net or for partial completions.
    # But MITM is likely to be the bottleneck.
    # Let's add a quick local search just in case.
    
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try adding any single item
            for i in range(n):
                if i in current_set:
                    continue
                new_sum = best_sum_frac + Fraction(1, denominators[i])
                if new_sum < 1 and new_sum > best_sum_frac:
                    best_sum_frac = new_sum
                    best_indices = list(current_set | {i})
                    current_set = set(best_indices)
                    improved = True
                    break
            
            # Try swapping
            if not improved:
                for remove_idx in list(current_set):
                    for add_idx in range(n):
                        if add_idx in current_set:
                            continue
                        new_sum = best_sum_frac - Fraction(1, denominators[remove_idx]) + Fraction(1, denominators[add_idx])
                        if new_sum < 1 and new_sum > best_sum_frac:
                            best_sum_frac = new_sum
                            new_set = (current_set - {remove_idx}) | {add_idx}
                            best_indices = list(new_set)
                            current_set = new_set
                            improved = True
                            break
                    if improved:
                        break

    return best_indices

if __name__ == "__main__":
    res = construct()
    # print(res)
    pass
