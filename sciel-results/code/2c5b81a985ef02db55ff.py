"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
from bisect import bisect_left, bisect_right

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    weights = [L // d for d in denominators]
    
    deadline = time.time() + 55
    
    # Split items into two halves
    mid = n // 2
    half1 = weights[:mid]
    half2 = weights[mid:]
    
    # Generate all subset sums and masks for a half
    def gen_sums(weights):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights):
            current_len = len(sums)
            for j in range(current_len):
                sums.append(sums[j] + w)
                masks.append(masks[j] | (1 << i))
        return sums, masks
    
    # Generate for half1
    sums1, masks1 = gen_sums(half1)
    
    # Generate for half2
    sums2, masks2 = gen_sums(half2)
    
    # Sort half2 by sum, keeping masks
    pairs2 = list(zip(sums2, masks2))
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sorted_sums2 = [p[0] for p in pairs2]
    
    # We want to find max s1 + s2 < L
    # For each s1, find largest s2 < L - s1
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0
    
    # Precompute: for each possible s1, we need to check if time allows
    # Use binary search on sorted_sums2
    
    for i in range(len(sums1)):
        s1 = sums1[i]
        if s1 >= L:
            continue
        
        target = L - s1
        # Find largest s2 < target
        idx = bisect_left(sorted_sums2, target)
        
        if idx > 0:
            s2 = sorted_sums2[idx - 1]
            total_sum = s1 + s2
            deficit = L - total_sum
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = masks1[i]
                best_mask2 = pairs2[idx - 1][1]
                
                if best_deficit == 1:
                    break
    
    # Reconstruct indices from masks
    final_indices = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            final_indices.append(i)
    for i in range(len(half2)):
        if best_mask2 & (1 << i):
            final_indices.append(mid + i)
    
    # Fallback if no valid subset
    if not final_indices:
        # Return index of smallest denominator (largest reciprocal)
        final_indices = [0]
    
    # Local search refinement: try to improve by swapping or adding/removing elements
    # Current sum in units of 1/L
    current_sum = sum(weights[i] for i in final_indices)
    current_deficit = L - current_sum
    
    # Build set of selected indices for fast lookup
    selected_set = set(final_indices)
    
    # Try single element swaps: remove one selected, add one unselected
    # Or add/remove single elements
    improved = False
    
    # First, try removing elements to see if we can get closer to L from below
    # Actually, we want to maximize sum < L, so we want to try adding elements
    # that don't push us over L, or swapping to get closer
    
    # Try adding unselected elements that keep sum < L
    for i in range(n):
        if i not in selected_set:
            new_sum = current_sum + weights[i]
            if new_sum < L:
                new_deficit = L - new_sum
                if new_deficit < current_deficit:
                    final_indices.append(i)
                    selected_set.add(i)
                    current_sum = new_sum
                    current_deficit = new_deficit
                    improved = True
    
    # Try single swaps: remove one selected, add one unselected
    if time.time() < deadline - 5:
        for i in list(final_indices):
            for j in range(n):
                if j not in selected_set:
                    # Remove i, add j
                    new_sum = current_sum - weights[i] + weights[j]
                    if 0 < new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < current_deficit:
                            # Perform the swap
                            final_indices.remove(i)
                            final_indices.append(j)
                            selected_set.remove(i)
                            selected_set.add(j)
                            current_sum = new_sum
                            current_deficit = new_deficit
                            improved = True
                            break
    
    # Try double swaps if time permits
    if time.time() < deadline - 10 and len(final_indices) >= 2:
        for i in range(len(final_indices)):
            for j in range(i + 1, len(final_indices)):
                idx_i = final_indices[i]
                idx_j = final_indices[j]
                for k in range(n):
                    if k not in selected_set:
                        for l in range(k + 1, n):
                            if l not in selected_set:
                                # Remove idx_i, idx_j; add k, l
                                new_sum = current_sum - weights[idx_i] - weights[idx_j] + weights[k] + weights[l]
                                if 0 < new_sum < L:
                                    new_deficit = L - new_sum
                                    if new_deficit < current_deficit:
                                        # Perform the double swap
                                        final_indices.remove(idx_i)
                                        final_indices.remove(idx_j)
                                        final_indices.append(k)
                                        final_indices.append(l)
                                        selected_set.remove(idx_i)
                                        selected_set.remove(idx_j)
                                        selected_set.add(k)
                                        selected_set.add(l)
                                        current_sum = new_sum
                                        current_deficit = new_deficit
                                        improved = True
                                        break
                if improved:
                    break
            if improved:
                break
    
    # Ensure nonempty
    if not final_indices:
        final_indices = [0]
    
    return final_indices
# EVOLVE-BLOCK-END
