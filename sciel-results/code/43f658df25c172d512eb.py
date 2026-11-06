import time
import random
from math import gcd
from functools import reduce
from operator import mul

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    # LCM(a, b) = abs(a*b) // gcd(a, b)
    L = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    
    # Pre-compute integer numerators for reciprocals
    # recips_int[i] = L / denominators[i]
    recips_int = [L // d for d in denominators]
    
    # Suffix sums of recips_int
    # suffix[i] = sum of recips_int from i to n-1
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recips_int[i]
    
    # State variables
    best_sum_int = 0
    best_indices = []
    
    def update_best(indices, sum_int):
        nonlocal best_sum_int, best_indices
        if sum_int > best_sum_int and sum_int < L:
            best_sum_int = sum_int
            best_indices = list(indices)

    # --- Phase 1: Fast Local Search to get a good initial bound ---
    # Greedy initialization: Add items in order if they fit
    current_indices = []
    current_sum = 0
    for i in range(n):
        if current_sum + recips_int[i] < L:
            current_indices.append(i)
            current_sum += recips_int[i]
    
    update_best(current_indices, current_sum)
    
    # Local Search Loop (Hill Climbing)
    # We use integer arithmetic for speed
    improved = True
    while improved and (time.time() - start_time < DEADLINE - 10):
        improved = False
        current_set = set(current_indices)
        
        # 1. Try adding a single element
        for i in range(n):
            if i not in current_set:
                new_sum = current_sum + recips_int[i]
                if new_sum < L:
                    # Since we are adding a positive value, new_sum > current_sum
                    # So deficit decreases
                    current_indices.append(i)
                    current_sum = new_sum
                    current_set.add(i)
                    update_best(current_indices, current_sum)
                    improved = True
                    break # Restart scan to ensure no other better add is missed immediately
        
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # 2. Try swapping: remove one, add one
        # We want to increase the sum. 
        # If we remove i (val vi) and add j (val vj), new_sum = current_sum - vi + vj
        # We need new_sum < L and new_sum > current_sum => vj > vi
        # Also j must not be in set, i must be in set.
        for i in list(current_indices):
            if time.time() - start_time > DEADLINE - 10:
                break
            vi = recips_int[i]
            for j in range(n):
                if j in current_set or j == i:
                    continue
                vj = recips_int[j]
                if vj <= vi:
                    continue # No gain
                new_sum = current_sum - vi + vj
                if new_sum < L:
                    # Perform swap
                    current_indices = [j if x == i else x for x in current_indices]
                    current_sum = new_sum
                    current_set.remove(i)
                    current_set.add(j)
                    update_best(current_indices, current_sum)
                    improved = True
                    break
            if improved:
                break
        
        # 3. Try 2-for-1 swap: remove i, add j, k
        # new_sum = current_sum - vi + vj + vk
        # Need new_sum < L and new_sum > current_sum => vj + vk > vi
        # This is more expensive O(N^3), so we only do it if time permits
        if time.time() - start_time < DEADLINE - 15:
             for i in list(current_indices):
                if time.time() - start_time > DEADLINE - 15:
                    break
                vi = recips_int[i]
                base_sum = current_sum - vi
                for j in range(n):
                    if j in current_set or j == i:
                        continue
                    vj = recips_int[j]
                    if base_sum + vj >= L:
                        continue
                    for k in range(n):
                        if k in current_set or k == i or k == j:
                            continue
                        vk = recips_int[k]
                        new_sum = base_sum + vj + vk
                        if new_sum < L and new_sum > current_sum:
                            # Perform swap
                            new_indices = [x for x in current_indices if x != i]
                            new_indices.append(j)
                            new_indices.append(k)
                            current_indices = new_indices
                            current_sum = new_sum
                            current_set.remove(i)
                            current_set.add(j)
                            current_set.add(k)
                            update_best(current_indices, current_sum)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # --- Phase 2: Branch and Bound for Exact Optimization ---
    # Use the best found so far to prune.
    # We search for the maximum sum < L.
    # Since N=47 is small, B&B with strong suffix pruning should be very fast if the bound is tight.
    
    def branch_and_bound(idx, current_sum, current_indices):
        nonlocal best_sum_int, best_indices
        
        if idx == n:
            if current_sum > best_sum_int:
                best_sum_int = current_sum
                best_indices = list(current_indices)
            return
        
        # Pruning 1: Even if we take all remaining, can we beat best?
        # Max possible sum from here is current_sum + suffix[idx]
        if current_sum + suffix[idx] <= best_sum_int:
            return
            
        # Pruning 2: If we take item idx, does it exceed L?
        # If current_sum + recips_int[idx] >= L, we cannot take it.
        # Note: We must strictly be below L.
        
        # Option 1: Include idx
        new_sum = current_sum + recips_int[idx]
        if new_sum < L:
            current_indices.append(idx)
            branch_and_bound(idx + 1, new_sum, current_indices)
            current_indices.pop()
            
        # Option 2: Exclude idx
        branch_and_bound(idx + 1, current_sum, current_indices)

    # Run B&B
    # To make B&B more effective, we can sort indices by value descending?
    # No, the indices are fixed to denominators 2..48. 
    # However, we can reorder the search order to try high-value items first to find good solutions quickly.
    # But the indices returned must be original indices.
    # Let's stick to original order for simplicity, but the suffix sum pruning is strong.
    # Actually, sorting by value descending helps B&B find large sums faster.
    # Let's create a search order: indices sorted by recips_int descending.
    search_order = sorted(range(n), key=lambda i: recips_int[i], reverse=True)
    
    # We need to map search_order back to original indices for the result.
    # The B&B function above assumes we are iterating 0..n-1 in the search_order array.
    # Let's adapt B&B to work on the sorted order.
    
    def branch_and_bound_sorted(pos, current_sum, current_indices):
        nonlocal best_sum_int, best_indices
        
        if pos == n:
            if current_sum > best_sum_int:
                best_sum_int = current_sum
                best_indices = list(current_indices)
            return
            
        # Pruning
        # We need suffix sums based on the *search order*
        # Let's pre-compute suffix_search
        # suffix_search[pos] = sum of recips_int[search_order[k]] for k from pos to n-1
        
        # Calculate max possible sum
        # We can precompute this array once outside.
        
        if current_sum + suffix_search[pos] <= best_sum_int:
            return
            
        # Option 1: Include search_order[pos]
        orig_idx = search_order[pos]
        val = recips_int[orig_idx]
        new_sum = current_sum + val
        if new_sum < L:
            current_indices.append(orig_idx)
            branch_and_bound_sorted(pos + 1, new_sum, current_indices)
            current_indices.pop()
            
        # Option 2: Exclude
        branch_and_bound_sorted(pos + 1, current_sum, current_indices)
        
    # Precompute suffix_search
    suffix_search = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_search[i] = suffix_search[i + 1] + recips_int[search_order[i]]
        
    # Run B&B with time check
    # B&B can be exponential, so we need a time guard inside recursion.
    # Since Python recursion is limited, and N=47, depth 47 is safe.
    # But the number of calls can be huge.
    # We will rely on the tight bound from Phase 1 to prune heavily.
    
    # To handle time limits in recursion, we can't easily break out cleanly without exceptions or global flags.
    # Given N=47 and strong pruning, it should be fast. 
    # If Phase 1 found a very good solution, B&B will prune most branches.
    
    # Let's try running it.
    # We wrap it in a try-catch just in case, though we expect it to finish.
    
    # One issue: If Phase 1 didn't find the global optimum, B&B will find it.
    # If Phase 1 found the global optimum, B&B will verify it quickly.
    
    # To be safe against timeout in B&B, we can limit the recursion depth or use a counter?
    # No, 47 is small. The pruning is the key.
    
    # However, if the bound is weak, B&B might explode.
    # Phase 1 local search is quite strong for this problem size.
    
    # Let's execute B&B
    # We use a global time check in the recursion? 
    # It's safer to just run it. If it takes too long, the program will be killed by the evaluator.
    # To mitigate, we can run B&B only if we have > 5 seconds left.
    
    if time.time() - start_time < DEADLINE - 5:
        branch_and_bound_sorted(0, 0, [])
        
    # --- Phase 3: Random Restarts + Local Search (if time remains) ---
    # If B&B didn't finish or if we want to double check with different seeds
    # Actually, B&B is exact. If it finishes, best_indices is optimal.
    # If it didn't finish (timeout), we return the best found so far.
    
    # Since we can't easily interrupt B&B, we rely on it finishing.
    # For N=47, with good initial bound, it should be very fast.
    
    return best_indices

# Pre-compute suffix_search for the sorted order used in B&B
# Note: search_order and suffix_search are defined inside construct in the draft above, 
# but they need to be available. I will move the definition inside construct to ensure scope correctness.

# Corrected construct function with all variables local
def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    recips_int = [L // d for d in denominators]
    
    # Suffix sums for original order (not strictly needed if we use sorted order for B&B, 
    # but useful for other heuristics if we add them)
    
    best_sum_int = 0
    best_indices = []
    
    def update_best(indices, sum_int):
        nonlocal best_sum_int, best_indices
        if sum_int > best_sum_int and sum_int < L:
            best_sum_int = sum_int
            best_indices = list(indices)

    # --- Phase 1: Local Search ---
    current_indices = []
    current_sum = 0
    for i in range(n):
        if current_sum + recips_int[i] < L:
            current_indices.append(i)
            current_sum += recips_int[i]
    
    update_best(current_indices, current_sum)
    
    improved = True
    while improved and (time.time() - start_time < DEADLINE - 10):
        improved = False
        current_set = set(current_indices)
        
        # Add
        for i in range(n):
            if i not in current_set:
                new_sum = current_sum + recips_int[i]
                if new_sum < L:
                    current_indices.append(i)
                    current_sum = new_sum
                    current_set.add(i)
                    update_best(current_indices, current_sum)
                    improved = True
                    break
        
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Swap 1-1
        for i in list(current_indices):
            if time.time() - start_time > DEADLINE - 10:
                break
            vi = recips_int[i]
            for j in range(n):
                if j in current_set or j == i:
                    continue
                vj = recips_int[j]
                if vj <= vi:
                    continue
                new_sum = current_sum - vi + vj
                if new_sum < L:
                    current_indices = [j if x == i else x for x in current_indices]
                    current_sum = new_sum
                    current_set.remove(i)
                    current_set.add(j)
                    update_best(current_indices, current_sum)
                    improved = True
                    break
            if improved:
                break
        
        # Swap 2-1
        if time.time() - start_time < DEADLINE - 15:
             for i in list(current_indices):
                if time.time() - start_time > DEADLINE - 15:
                    break
                vi = recips_int[i]
                base_sum = current_sum - vi
                for j in range(n):
                    if j in current_set or j == i:
                        continue
                    vj = recips_int[j]
                    if base_sum + vj >= L:
                        continue
                    for k in range(n):
                        if k in current_set or k == i or k == j:
                            continue
                        vk = recips_int[k]
                        new_sum = base_sum + vj + vk
                        if new_sum < L and new_sum > current_sum:
                            new_indices = [x for x in current_indices if x != i]
                            new_indices.append(j)
                            new_indices.append(k)
                            current_indices = new_indices
                            current_sum = new_sum
                            current_set.remove(i)
                            current_set.add(j)
                            current_set.add(k)
                            update_best(current_indices, current_sum)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # --- Phase 2: Branch and Bound ---
    # Sort by value descending for better B&B performance
    search_order = sorted(range(n), key=lambda i: recips_int[i], reverse=True)
    
    # Precompute suffix sums for the search order
    suffix_search = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_search[i] = suffix_search[i + 1] + recips_int[search_order[i]]
        
    def branch_and_bound_sorted(pos, current_sum, current_indices):
        nonlocal best_sum_int, best_indices
        
        if pos == n:
            if current_sum > best_sum_int:
                best_sum_int = current_sum
                best_indices = list(current_indices)
            return
            
        # Pruning: Can we beat the best?
        if current_sum + suffix_search[pos] <= best_sum_int:
            return
            
        # Option 1: Include
        orig_idx = search_order[pos]
        val = recips_int[orig_idx]
        new_sum = current_sum + val
        if new_sum < L:
            current_indices.append(orig_idx)
            branch_and_bound_sorted(pos + 1, new_sum, current_indices)
            current_indices.pop()
            
        # Option 2: Exclude
        branch_and_bound_sorted(pos + 1, current_sum, current_indices)
        
    if time.time() - start_time < DEADLINE - 5:
        branch_and_bound_sorted(0, 0, [])
        
    return best_indices
