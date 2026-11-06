import time
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    # A: 0 to 21 (22 items)
    # B: 22 to 46 (25 items)
    mid = 22
    
    A = denominators[:mid]
    B = denominators[mid:]
    
    # 1. Generate all subsets of A
    # We store (sum, mask)
    sums_A = []
    n_A = len(A)
    # Precompute reciprocals for A
    recips_A = [Fraction(1, d) for d in A]
    
    for mask in range(1 << n_A):
        s = Fraction(0, 1)
        # Calculate sum for this mask
        # Optimization: use DP? No, 4M * 22 steps is ~100M ops.
        # Let's use a simple loop.
        for i in range(n_A):
            if mask & (1 << i):
                s += recips_A[i]
        if s < 1:
            sums_A.append((s, mask))
            
    if time.time() > deadline:
         # Fallback to greedy if A generation is too slow (unlikely)
         selected, total = [], Fraction()
         for i, d in enumerate(denominators):
             if total + Fraction(1, d) < 1:
                 selected.append(i)
                 total += Fraction(1, d)
         return selected

    # Sort sums_A by sum
    sums_A.sort(key=lambda x: x[0])
    
    # Extract values and masks for binary search
    vals_A = [x[0] for x in sums_A]
    masks_A = [x[1] for x in sums_A]
    
    # 2. Generate subsets of B with pruning
    # Sort B in descending order to prune earlier?
    # Larger reciprocals first.
    # Indices in B correspond to original indices mid + j
    B_sorted = sorted([(d, i) for i, d in enumerate(B)], key=lambda x: x[0], reverse=True)
    recips_B = [Fraction(1, d) for d, _ in B_sorted]
    # Map sorted index j to original index in B (which is mid + j)
    # Actually, we just need the mask relative to B.
    # When we find a best mask in B, we need to map it back to original indices.
    # Let's keep track of the original indices in B.
    # B_orig_indices = list(range(mid, n))
    # B_sorted_indices = [idx for _, idx in B_sorted]
    
    best_sum = Fraction(0, 1)
    best_mask_A = 0
    best_mask_B = 0 # Mask relative to sorted B order
    
    n_B = len(B)
    
    # Recursive function for B
    # idx: current index in B_sorted
    # curr_sum: current sum
    # curr_mask: current mask (bits 0 to n_B-1)
    
    def search_B(idx, curr_sum, curr_mask):
        nonlocal best_sum, best_mask_A, best_mask_B
        
        if time.time() > deadline:
            return
            
        if idx == n_B:
            # Check if curr_sum < 1
            if curr_sum < 1:
                target = 1 - curr_sum
                # Find largest val in vals_A < target
                # bisect_left returns index where target would be inserted
                # We want the element before it
                pos = bisect.bisect_left(vals_A, target)
                if pos > 0:
                    candidate_sum = vals_A[pos - 1] + curr_sum
                    if candidate_sum > best_sum:
                        best_sum = candidate_sum
                        best_mask_A = masks_A[pos - 1]
                        best_mask_B = curr_mask
            return
            
        # Option 1: Skip B[idx]
        search_B(idx + 1, curr_sum, curr_mask)
        
        # Option 2: Take B[idx]
        new_sum = curr_sum + recips_B[idx]
        if new_sum < 1:
            search_B(idx + 1, new_sum, curr_mask | (1 << idx))
            
    search_B(0, Fraction(0, 1), 0)
    
    # Reconstruct indices
    # best_mask_A corresponds to A (original indices 0 to mid-1)
    # best_mask_B corresponds to B_sorted (needs mapping back to original B indices)
    
    selected = []
    
    # Process A
    for i in range(n_A):
        if best_mask_A & (1 << i):
            selected.append(i) # Original index is i
            
    # Process B
    # B_sorted[j] has original index in B: B_sorted_indices[j]
    # Which corresponds to global index: mid + B_sorted_indices[j]
    for j in range(n_B):
        if best_mask_B & (1 << j):
            # j is index in sorted B
            # Find original index in B
            orig_idx_in_B = B_sorted[j][1]
            global_idx = mid + orig_idx_in_B
            selected.append(global_idx)
            
    # Ensure non-empty
    if not selected:
        # Fallback: pick index 0 (denominator 2)
        return [0]
        
    return selected
