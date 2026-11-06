"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    from math import gcd
    from fractions import Fraction
    import bisect

    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit (using 55s to be safe)

    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators for exact integer arithmetic
    L = 1
    for d in denominators:
        L = (L * d) // gcd(L, d)
        
    # Weights are L // d
    weights = [L // d for d in denominators]
    
    # We want to maximize sum of weights subject to sum < L.
    # Deficit is L - sum. We want to minimize deficit.
    
    best_sum = 0
    best_indices = []
    
    def evaluate_and_update(indices):
        nonlocal best_sum, best_indices
        if not indices:
            return
        s = sum(weights[i] for i in indices)
        if s < L and s > best_sum:
            best_sum = s
            best_indices = list(indices)

    # --- Phase 1: Greedy Initialization ---
    # Try multiple greedy orders to get good starting points.
    
    # Order 1: Standard ascending denominator (largest weight first)
    curr_sum = 0
    curr_indices = []
    for i in range(n):
        if curr_sum + weights[i] < L:
            curr_sum += weights[i]
            curr_indices.append(i)
    evaluate_and_update(curr_indices)

    # Order 2: Descending denominator (smallest weight first)
    curr_sum = 0
    curr_indices = []
    for i in range(n - 1, -1, -1):
        if curr_sum + weights[i] < L:
            curr_sum += weights[i]
            curr_indices.append(i)
    evaluate_and_update(curr_indices)

    # --- Phase 2: Meet-in-the-Middle (MITM) ---
    # This is the structural change. Instead of random restarts + local search,
    # we use MITM which is more systematic for subset sum problems.
    # Split indices into two halves.
    # Generate all subset sums for first half.
    # Generate all subset sums for second half.
    # Sort second half sums.
    # For each sum in first half, binary search in second half for largest sum < L - sum1.
    
    # Optimization: Prune sums that are already >= L.
    # Also, we can prune dominated sums? No, for subset sum, we just need all valid sums.
    # However, 2^23 is ~8M. Storing 8M integers is feasible.
    # But we need to reconstruct indices. Storing masks is also feasible (64-bit int).
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    n1 = len(w1)
    n2 = len(w2)
    
    # Generate subset sums for first half
    # sums1: list of (sum, mask)
    sums1 = [(0, 0)]
    for i in range(n1):
        w = w1[i]
        # Add w to existing sums
        new_sums = []
        for s, m in sums1:
            ns = s + w
            if ns < L:
                new_sums.append((ns, m | (1 << i)))
        sums1.extend(new_sums)
        
    if time.time() > deadline - 10:
        return best_indices

    # Generate subset sums for second half
    sums2 = [(0, 0)]
    for i in range(n2):
        w = w2[i]
        new_sums = []
        for s, m in sums2:
            ns = s + w
            if ns < L:
                new_sums.append((ns, m | (1 << i)))
        sums2.extend(new_sums)
        
    if time.time() > deadline - 5:
        return best_indices

    # Sort sums2 by sum value to allow binary search
    sums2.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    s2_vals = [s for s, m in sums2]
    
    # Iterate through sums1
    # For each s1, we want max s2 such that s1 + s2 < L
    # i.e., s2 < L - s1
    # Binary search for rightmost index in s2_vals < L - s1
    
    for s1, m1 in sums1:
        if time.time() > deadline - 2:
            break
            
        target = L - s1
        # We want s2 < target
        # bisect_left gives first index >= target. We want the one before it.
        idx = bisect.bisect_left(s2_vals, target) - 1
        
        if idx >= 0:
            s2, m2 = sums2[idx]
            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                # Reconstruct indices
                curr_indices = []
                # First half
                for i in range(n1):
                    if m1 & (1 << i):
                        curr_indices.append(i)
                # Second half
                for i in range(n2):
                    if m2 & (1 << i):
                        curr_indices.append(mid + i)
                best_indices = curr_indices

    # --- Phase 3: Local Search Refinement ---
    # Use the best solution found by MITM as a starting point for local search.
    # This can help if MITM missed a slightly better combination due to pruning or if we want to polish.
    
    def local_search(indices_set):
        """
        Perform local search on a set of indices.
        Moves:
        1. Add an unselected item if it fits.
        2. Swap a selected item with an unselected item if the swap improves the sum and keeps it < L.
        """
        curr_sum = sum(weights[i] for i in indices_set)
        improved = True
        while improved:
            improved = False
            curr_indices_list = list(indices_set)
            
            # Try adding
            for i in range(n):
                if i not in indices_set:
                    new_sum = curr_sum + weights[i]
                    if new_sum < L:
                        indices_set.add(i)
                        curr_sum = new_sum
                        improved = True
                        break
            
            # Try swapping
            if not improved:
                for i in list(indices_set):
                    for j in range(n):
                        if j in indices_set:
                            continue
                        # We want to increase the sum, so weights[j] should be > weights[i]
                        if weights[j] <= weights[i]:
                            continue
                        
                        new_sum = curr_sum - weights[i] + weights[j]
                        if new_sum < L:
                            # This swap increases sum and is valid.
                            indices_set.remove(i)
                            indices_set.add(j)
                            curr_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
                        
        return indices_set, curr_sum

    if time.time() < deadline - 5 and best_indices:
        best_set = set(best_indices)
        best_sum_val = best_sum
        
        # Try removing each item in the best set and refilling
        for i in list(best_set):
            if time.time() > deadline - 2:
                break
                
            temp_set = best_set.copy()
            temp_set.remove(i)
            temp_sum = best_sum_val - weights[i]
            
            remaining_items = [j for j in range(n) if j not in temp_set]
            remaining_items.sort(key=lambda x: -weights[x])
            
            for j in remaining_items:
                if temp_sum + weights[j] < L:
                    temp_sum += weights[j]
                    temp_set.add(j)
                    
            temp_set, temp_sum = local_search(temp_set)
            
            if temp_sum > best_sum_val:
                best_sum_val = temp_sum
                best_indices = list(temp_set)
                best_set = set(best_indices)
                best_sum = best_sum_val

    # Final Verification
    if not best_indices:
        # Fallback to simple greedy if something went wrong
        curr_sum = 0
        for i in range(n):
            if curr_sum + weights[i] < L:
                curr_sum += weights[i]
                best_indices.append(i)
        best_sum = curr_sum

    # Ensure strict inequality and distinct indices
    final_sum = sum(weights[i] for i in best_indices)
    if final_sum >= L:
        for i in best_indices[:]:
            if final_sum - weights[i] < L:
                best_indices.remove(i)
                final_sum -= weights[i]
                break
        if not best_indices:
            best_indices = [0]
            final_sum = weights[0]

    return best_indices
# EVOLVE-BLOCK-END
