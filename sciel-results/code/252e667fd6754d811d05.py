import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s budget for 60s limit (plus overhead)
    
    N = 256
    # Candidates are 2..256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Precompute lcm conflicts
    # conflict_mask[i] is a bitset (int) where bit j is 1 if candidates[i] and candidates[j] conflict
    # We use 0-indexed indices. candidates[i] = i + 2.
    conflict_mask = [0] * n
    for i in range(n):
        a = candidates[i]
        mask = 0
        for j in range(i + 1, n):
            b = candidates[j]
            if math.lcm(a, b) <= N:
                mask |= (1 << j)
                conflict_mask[j] |= (1 << i)
        conflict_mask[i] = mask

    # We want to maximize sum(1/x).
    # Since 1/x decreases as x increases, we sort candidates by value descending (x ascending).
    # But the indices in conflict_mask are tied to the original order (2..256).
    # Let's keep the original order for bitmasks, but iterate in order of "value" (x).
    # Actually, for Branch and Bound, we usually pick the next element to consider.
    # To get good bounds quickly, we should consider elements with highest weight (smallest x) first?
    # Or does it matter? The bound is sum of all *remaining* elements.
    # If we process in order of index (2, 3, 4...), the "remaining" set shrinks from the right.
    # Let's stick to processing indices in order 0 to n-1 (values 2 to 256).
    
    # Precompute suffix sums of 1/x for the upper bound.
    # bound[i] = sum(1/candidates[k] for k in range(i, n))
    # This is the maximum possible sum we can add if we were to pick ALL elements from i to n-1.
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + (1.0 / candidates[i])

    best_sum = 0.0
    best_selection = []
    
    # State:
    # idx: current index to consider (0 to n)
    # current_sum: sum of selected so far
    # selected_mask: bitset of selected indices
    # available_mask: bitset of indices that are still compatible with current selection
    # Note: available_mask should only have bits set for indices >= idx? 
    # Actually, if we decide to skip index idx, it's gone. If we take it, it conflicts with others.
    # Let's use a recursive approach:
    # solve(idx, current_sum, available_mask)
    # available_mask contains bits for all j >= idx such that j is compatible with all currently selected items.
    
    # To make it faster, we can use a global variable for best and prune aggressively.
    
    def solve(idx, current_sum, available_mask):
        nonlocal best_sum, best_selection
        
        # Time check
        if time.time() > deadline:
            return
            
        # Upper bound: current_sum + sum of 1/x for all available elements from idx onwards
        # Wait, available_mask might have holes. 
        # The tightest simple bound is current_sum + suffix_sum[idx] is WRONG because some are unavailable.
        # A better bound: current_sum + sum(1/candidates[j] for j in range(idx, n) if (available_mask >> j) & 1)
        # Computing this sum every time is O(n). 
        # Optimization: Precompute prefix/suffix sums? No, the mask changes arbitrarily.
        # However, n=255. O(n) per node is acceptable if tree is small.
        # Let's calculate the bound exactly for the available bits.
        
        # Quick check: if current_sum + suffix_sum[idx] <= best_sum, prune.
        # This is a valid (though loose) bound. If even taking ALL remaining (ignoring conflicts) doesn't beat best, prune.
        if current_sum + suffix_sum[idx] <= best_sum + 1e-12:
            return

        # Calculate exact potential sum of available items from idx onwards
        # To speed this up, we can iterate through set bits in available_mask starting from idx.
        potential = 0.0
        # Iterate through bits in available_mask
        m = available_mask
        # We only care about bits >= idx
        # Shift right by idx to simplify? Or just iterate.
        # Since n is small (255), iterating is fine.
        for j in range(idx, n):
            if (m >> j) & 1:
                potential += 1.0 / candidates[j]
                
        if current_sum + potential <= best_sum + 1e-12:
            return

        # Base case: no more items to consider
        if idx == n:
            if current_sum > best_sum + 1e-12:
                best_sum = current_sum
                # Reconstruct selection from available_mask? No, we need to track selection.
                # We can track selected indices in the recursion stack or pass a list.
                # Passing a list is expensive (copying).
                # Alternative: Use a global list `current_path` and backtrack.
                # Let's switch to using a global path list.
                pass
            return

        # Option 1: Skip idx
        # Is idx available?
        if (available_mask >> idx) & 1:
            # If we skip idx, it remains unavailable? No, if we skip it, we just don't pick it.
            # The "available_mask" for the next step (idx+1) should be the same, 
            # EXCEPT we must clear the bit for idx (since we are moving past it).
            # Actually, the recursion moves idx to idx+1. 
            # The available_mask passed to the next level represents "which future indices are compatible with current selection".
            # Since we didn't select idx, the compatibility of future indices doesn't change.
            # So next_available = available_mask (bits for idx+1..n-1 are unchanged).
            # But we must ensure we don't consider idx again. The `idx` parameter handles that.
            
            # However, we must be careful: if we skip idx, do we remove it from available?
            # Yes, effectively. But since we increment idx, we just look at bits >= idx+1.
            
            solve(idx + 1, current_sum, available_mask)
            
            # Option 2: Take idx
            # Select idx
            new_sum = current_sum + (1.0 / candidates[idx])
            
            # Update available_mask:
            # Remove idx from available (bit idx is 0 in next mask, or we just ignore it since idx increases)
            # Remove all j that conflict with idx from available.
            # conflict_mask[idx] has bits for j where lcm(a,b) <= N.
            # We need to clear these bits in available_mask.
            
            # Note: conflict_mask[idx] might contain bits < idx. We don't care about those for future, 
            # but bitwise AND handles it.
            
            next_available = available_mask & ~conflict_mask[idx]
            # Also clear the bit for idx itself just to be clean (though idx increments)
            next_available &= ~(1 << idx)
            
            solve(idx + 1, new_sum, next_available)
        else:
            # idx is not available, must skip
            solve(idx + 1, current_sum, available_mask)

    # The standard recursive B&B above doesn't track the *path* for the best solution.
    # Let's modify to track the path.
    
    current_path = []
    
    def solve_tracked(idx, current_sum, available_mask):
        nonlocal best_sum, best_selection
        
        if time.time() > deadline:
            return
            
        # Prune with loose bound first (fast)
        if current_sum + suffix_sum[idx] <= best_sum + 1e-12:
            return

        # Calculate tight bound (slower)
        # Optimization: If the loose bound is very close to best, we might still prune with tight.
        # But let's just do the tight bound check.
        potential = 0.0
        m = available_mask
        # Iterate set bits in m starting from idx
        # Efficient way: 
        # temp = m & ~((1 << idx) - 1)  # mask out bits < idx
        # Then iterate set bits in temp.
        temp = m & ~((1 << idx) - 1)
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            potential += 1.0 / candidates[j]
            temp ^= lsb
            
        if current_sum + potential <= best_sum + 1e-12:
            return

        if idx == n:
            if current_sum > best_sum + 1e-12:
                best_sum = current_sum
                best_selection = current_path[:]
            return

        if (available_mask >> idx) & 1:
            # Try taking idx
            current_path.append(idx)
            val = 1.0 / candidates[idx]
            new_sum = current_sum + val
            next_available = available_mask & ~conflict_mask[idx]
            next_available &= ~(1 << idx)
            
            solve_tracked(idx + 1, new_sum, next_available)
            
            current_path.pop()
            
            # Try skipping idx
            solve_tracked(idx + 1, current_sum, available_mask)
        else:
            # Must skip
            solve_tracked(idx + 1, current_sum, available_mask)

    # Initial available mask: all bits 0 to n-1 set
    initial_mask = (1 << n) - 1
    
    # Start recursion
    solve_tracked(0, 0.0, initial_mask)
    
    # If best_selection is empty (shouldn't happen as 2 is always valid alone), return [2]
    if not best_selection:
         return [2]
         
    result = [candidates[i] for i in best_selection]
    return result
