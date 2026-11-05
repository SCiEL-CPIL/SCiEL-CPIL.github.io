import numpy as np
from fractions import Fraction
import time

def construct():
    start = time.time()
    deadline = start + 80  # 80 second budget, 10s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    weights = np.array([1.0/d for d in denominators])
    
    # Split into two halves
    mid = n // 2  # 23
    A = weights[:mid]  # 23 items
    B = weights[mid:]  # 24 items
    
    # Generate all subset sums for A
    # 2^23 = 8388608
    nA = 1 << mid
    sums_A = np.zeros(nA)
    for i, w in enumerate(A):
        # For all subsets that include item i, add w
        # Items 0 to i-1 define the pattern
        block = 1 << i
        for start_idx in range(0, nA, 2*block):
            sums_A[start_idx+block:start_idx+2*block] += w
    
    # Sort A sums and keep track of original indices
    order_A = np.argsort(sums_A)
    sorted_sums_A = sums_A[order_A]
    sorted_idx_A = order_A
    
    # Generate all subset sums for B
    nB = 1 << (n - mid)
    sums_B = np.zeros(nB)
    for i, w in enumerate(B):
        block = 1 << i
        for start_idx in range(0, nB, 2*block):
            sums_B[start_idx+block:start_idx+2*block] += w
    
    # For each B subset, find best A subset
    best_deficit = 1.0
    best_A_mask = 0
    best_B_mask = 0
    
    # Use binary search: for each B sum, find largest A sum < 1 - B_sum
    # We need A_sum + B_sum < 1, so A_sum < 1 - B_sum
    # Use np.searchsorted
    
    # Vectorize: for each B_sum, find the largest A_sum < 1 - B_sum
    targets = 1.0 - sums_B
    # Find insertion point for each target in sorted_sums_A
    # We want the largest A_sum strictly less than target
    # np.searchsorted gives the index where target would be inserted
    # So the element at index-1 is the largest <= target
    # But we need strictly less, so if sums_A[idx] == target, we need idx-1
    
    insert_pos = np.searchsorted(sorted_sums_A, targets, side='left')
    # The candidate A sum is at insert_pos - 1 (if >= 0)
    valid = insert_pos > 0
    cand_pos = np.where(valid, insert_pos - 1, 0)
    cand_sums_A = sorted_sums_A[cand_pos]
    
    # But we need to ensure cand_sums_A < 1 - sums_B (strictly)
    # Check: cand_sums_A + sums_B < 1
    total_sums = cand_sums_A + sums_B
    valid_strict = total_sums < 1.0
    valid = valid & valid_strict
    
    if np.any(valid):
        # Find the minimum total_sum (which gives minimum deficit)
        # Actually we want to MINIMIZE deficit = 1 - total_sum
        # So we want to MAXIMIZE total_sum subject to total_sum < 1
        # So find the maximum total_sum among valid entries
        max_total = np.max(total_sums[valid])
        # Find which B index gives this
        # But there might be ties, pick any
        b_idx = np.argmax(np.where(valid, total_sums, -1))
        
        # Get the A index
        a_pos = cand_pos[b_idx]
        a_mask = sorted_idx_A[a_pos]
        b_mask = b_idx
        
        # Convert masks to indices
        selected = []
        for i in range(mid):
            if a_mask & (1 << i):
                selected.append(i)
        for i in range(n - mid):
            if b_mask & (1 << i):
                selected.append(mid + i)
        
        # Verify with exact rationals
        total_frac = sum(Fraction(1, denominators[i]) for i in selected)
        if total_frac < 1:
            deficit = 1 - total_frac
            return selected
    
    # Fallback to greedy
    selected, total = [], Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
