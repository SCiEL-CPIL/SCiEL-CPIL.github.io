import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin for 60s limit
    
    # Denominators 2 to 48 inclusive
    # Indices 0..46 correspond to denominators 2..48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Sort denominators in descending order to pick largest fractions first.
    # This helps in filling the sum towards 1 quickly, which is crucial for 
    # minimizing the deficit (getting as close to 1 as possible).
    # We need to map back to original indices.
    # Original index i corresponds to denominator i+2.
    # Let's create a list of (denominator, original_index)
    items = [(d, i) for i, d in enumerate(denominators)]
    # Sort by denominator descending
    items.sort(key=lambda x: -x[0])
    
    # Precompute suffix sums of reciprocals for pruning.
    # suffix_sum[i] = sum of 1/d for items[i:]
    # This allows us to know the maximum additional sum we can get from i to end.
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + Fraction(1, items[i][0])
    
    best_deficit = Fraction(1)  # Initial best deficit is 1 (empty set sum 0)
    best_mask = 0  # Bitmask of selected items in the sorted order
    
    # We use a recursive DFS with Branch and Bound.
    # To avoid recursion depth issues or overhead, we could use an explicit stack,
    # but Python recursion limit is usually 1000, and n=47, so depth is fine.
    # However, Python function call overhead can be high. 
    # Given the tight time limit, an iterative stack or optimized recursion is needed.
    # Let's try recursion first with aggressive pruning. If it's too slow, we'll switch.
    # Actually, for n=47, the tree is huge. Pruning is key.
    
    # Optimization: Use float for quick bounds check in the inner loop if possible,
    # but keep Fraction for exactness of the best candidate.
    # Let's stick to Fraction for correctness, but optimize the loop.
    
    # To speed up, we can precompute the float values of suffix sums for faster comparison.
    suffix_sums_float = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums_float[i] = suffix_sums_float[i + 1] + 1.0 / items[i][0]
    
    # Also precompute float reciprocals for items
    item_recs_float = [1.0 / d for d, _ in items]
    
    # We'll use a stack for iterative DFS to manage state and allow early termination checks.
    # State: (index, current_sum_frac, current_sum_float, mask)
    # We process items from 0 to n-1.
    
    # Initial state
    # We want to explore the branch that INCLUDES the current item first, 
    # because larger fractions help reach 1 faster.
    
    # Stack elements: (idx, curr_sum_frac, curr_sum_float, mask)
    stack = [(0, Fraction(0), 0.0, 0)]
    
    while stack:
        if time.time() > deadline:
            break
        
        idx, curr_sum_frac, curr_sum_float, mask = stack.pop()
        
        if idx == n:
            # End of list, update best if better
            if curr_sum_frac < 1:
                deficit = 1 - curr_sum_frac
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_mask = mask
            continue
        
        # Pruning 1: If current sum + max possible remaining sum <= current best sum
        # i.e., if curr_sum + suffix_sums[idx] <= 1 - best_deficit
        # Then we cannot improve.
        # Note: suffix_sums[idx] includes item idx.
        
        # Pruning 2: If current sum >= 1, this path is invalid (strictly below 1 required).
        # Since we add positive numbers, any extension will also be >= 1.
        if curr_sum_frac >= 1:
            continue
        
        # Check if we can even improve with the remaining items
        # Max potential sum from here is curr_sum_frac + suffix_sums[idx]
        # If this max potential sum is <= (1 - best_deficit), we can't beat best_deficit.
        # Wait, we want to MINIMIZE deficit, which means MAXIMIZE sum.
        # So if max_possible_sum <= best_sum_found, prune.
        # best_sum_found = 1 - best_deficit
        best_sum_found = 1 - best_deficit
        
        # Use float for quick check first to avoid Fraction overhead
        if curr_sum_float + suffix_sums_float[idx] <= best_sum_found + 1e-12:
            continue
            
        # Exact check for the bound
        if curr_sum_frac + suffix_sums[idx] <= best_sum_found:
            continue
            
        d, orig_idx = items[idx]
        rec = Fraction(1, d)
        rec_float = item_recs_float[idx]
        
        # Option 1: Include item idx
        # New sum = curr_sum + rec
        # We push the "Exclude" option first so that "Include" is processed next (LIFO)
        # This prioritizes larger fractions (which are earlier in the sorted list).
        
        # Push Exclude
        # Before pushing, we can check if excluding this item still allows improvement.
        # Max sum if we exclude idx is curr_sum + suffix_sums[idx+1]
        if curr_sum_frac + suffix_sums[idx + 1] > best_sum_found:
            stack.append((idx + 1, curr_sum_frac, curr_sum_float, mask))
            
        # Push Include
        new_sum_frac = curr_sum_frac + rec
        new_sum_float = curr_sum_float + rec_float
        
        # If new sum is still < 1, we can continue exploring this branch
        # If new sum >= 1, this branch is dead (unless we don't add any more, but we are at a leaf or continuing)
        # Actually, if new_sum >= 1, we cannot add ANY more items. 
        # So we should check if this specific combination (mask with bit idx set) is a valid candidate.
        # But we are at idx, so we haven't decided on idx+1..n-1.
        # If new_sum >= 1, we must stop adding items. 
        # So we evaluate the candidate with mask | (1<<idx) and sum new_sum_frac.
        # We do NOT push further states for this branch.
        
        if new_sum_frac < 1:
            # We can still add more items
            # Check if including this item could lead to a better result
            # The max sum from here is new_sum + suffix_sums[idx+1]
            if new_sum_frac + suffix_sums[idx + 1] > best_sum_found:
                stack.append((idx + 1, new_sum_frac, new_sum_float, mask | (1 << idx)))
            else:
                # Even with max remaining, we can't beat best. 
                # But wait, maybe this partial sum itself is the best?
                # No, we want the MAX sum < 1. 
                # If we can't beat best_sum_found by adding more, maybe the current sum is better?
                # Current sum is new_sum_frac.
                # If new_sum_frac > best_sum_found, update best.
                if new_sum_frac > best_sum_found:
                    best_sum_found = new_sum_frac
                    best_deficit = 1 - best_sum_found
                    best_mask = mask | (1 << idx)
        else:
            # new_sum_frac >= 1. This combination is invalid as a final answer.
            # And since all subsequent items are positive, any extension is also invalid.
            # So this branch is completely dead. Do nothing.
            pass

    # Decode best_mask to original indices
    # best_mask has bits set for the sorted items.
    # We need to map back to original zero-based indices.
    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            # items[i] is (denominator, original_index)
            orig_idx = items[i][1]
            result_indices.append(orig_idx)
            
    if not result_indices:
        # Fallback: should not happen, but return a valid subset
        # Pick the largest denominator (smallest fraction) to ensure sum < 1?
        # Or just index 0 (denom 2, sum 0.5)
        result_indices = [0]
        
    return result_indices
