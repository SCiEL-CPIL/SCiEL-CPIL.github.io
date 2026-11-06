"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    from functools import lru_cache
    
    N = 512
    # Vertices are integers 1..511
    # Weight of vertex a is 1/(512 - a)
    # We want to maximize sum of weights subject to pairwise coprime constraint.
    
    # Step 1: Precompute coprime relationships
    # For each a in 1..511, determine which other numbers are coprime to it.
    # We will build a conflict graph where an edge exists if gcd(a, b) > 1.
    # An independent set in this graph is a set of pairwise coprime numbers.
    
    # To use branch and bound effectively, we order vertices by weight descending.
    # Higher weights (larger a) are more valuable.
    
    # Let's represent the set of available vertices as a bitset (Python int).
    # Bit i is 1 if vertex (i+1) is available. Note: vertex 1 is index 0.
    # Actually, let's map a -> index a-1.
    # So a=1 is bit 0, a=511 is bit 510.
    
    # Precompute for each vertex a, the bitmask of vertices that are NOT coprime to a.
    # If we pick a, we must remove all non-coprime vertices from the available set.
    
    # n = 512
    # vertices 1 to 511
    
    # Let's create an array `conflict_mask[a]` for a in 1..511.
    # conflict_mask[a] will be a bitmask of all b in 1..511 such that gcd(a, b) > 1.
    # Note: this includes a itself? gcd(a,a)=a>1 for a>1. For a=1, gcd(1,1)=1.
    # We need to be careful. If we pick a, we cannot pick any b where gcd(a,b)>1.
    
    # Let's compute prime factors for each number to speed up gcd or just use math.gcd.
    # Since 511 is small, math.gcd is fast enough for precomputation.
    
    conflict = [0] * 512  # index 1 to 511
    for a in range(1, 512):
        mask = 0
        for b in range(1, 512):
            if gcd(a, b) > 1:
                mask |= (1 << (b - 1))
        conflict[a] = mask
    
    # Weights
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
    
    # Sort vertices by weight descending to help pruning?
    # In branch and bound, we usually pick the most valuable vertex first to get high scores early.
    # However, the bitset approach relies on indices.
    # Let's stick to the natural order but use upper bound pruning.
    
    # Upper bound calculation:
    # Given a set of available vertices (bitset), what is the maximum possible sum of weights?
    # A simple upper bound is the sum of the top k weights where k is the size of the maximum independent set?
    # Or simply the sum of all weights in the available set? No, that's too loose.
    # A better bound: The maximum number of vertices we can pick is bounded by the chromatic number or clique cover?
    # Actually, a simple greedy upper bound: sort available vertices by weight desc, sum them all? No.
    
    # Let's use a standard Branch and Bound for Maximum Weight Independent Set.
    # State: current_sum, available_mask, current_set
    # We process vertices in a specific order.
    
    # To make it efficient, let's reorder the vertices by weight descending.
    # Let `order` be the list of vertices sorted by weight descending.
    # weight[511] = 1/1 = 1.0
    # weight[510] = 1/2 = 0.5
    # ...
    # weight[1] = 1/511
    
    order = sorted(range(1, 512), key=lambda x: weights[x], reverse=True)
    
    # We need to map the new order to bit positions?
    # It's easier to keep the original bit positions and just iterate through `order`.
    
    # Global best
    best_sum = 0.0
    best_set = []
    
    # Recursive B&B
    # To avoid recursion depth issues (though 511 is deep), we can use iterative or just increase limit.
    # Python default recursion limit is 1000. 511 is fine.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Precompute suffix sums of weights in `order` for upper bound?
    # If we are at index i in `order`, the maximum we can add is the sum of weights of all remaining vertices in `order` that are still available.
    # But we can't pick all of them due to conflicts.
    # A tighter bound: For each remaining vertex, it might conflict with others.
    # Simple bound: sum of weights of all available vertices. This is valid but loose.
    
    # Let's try a slightly better bound:
    # The maximum independent set size is at most the size of the largest clique in the complement? No.
    # Let's just use the sum of all available weights as the upper bound for the remaining part.
    # If current_sum + sum(available_weights) <= best_sum, prune.
    
    # We need to quickly compute sum of weights of available vertices.
    # We can precompute weight_sum_mask? No, mask is 512 bits, too big for array.
    # We can iterate through the bits of the mask to sum weights. This is O(N) per node, which might be slow.
    # Alternative: Precompute for each vertex its weight. In the recursion, we maintain a running sum?
    # No, the available set changes non-uniformly.
    
    # Optimization:
    # Instead of summing all available weights, we can note that the weights are decreasing with index in `order`.
    # Let's store weights in an array `w` indexed by position in `order`.
    # w[0] = weight[511], w[1] = weight[510], etc.
    
    w_order = [weights[a] for a in order]
    
    # Let's map original vertex a to its position in `order`.
    pos_in_order = [0] * 512
    for i, a in enumerate(order):
        pos_in_order[a] = i
    
    # Conflict masks in terms of `order` indices?
    # If we pick vertex a at position i, we must remove all vertices b that conflict with a.
    # Let's precompute `conflict_in_order[i]` which is a bitmask of positions j such that order[j] conflicts with order[i].
    
    conflict_order = [0] * 512
    for i, a in enumerate(order):
        mask = 0
        for j, b in enumerate(order):
            if gcd(a, b) > 1:
                mask |= (1 << j)
        conflict_order[i] = mask
    
    # Now the problem is:
    # Maximize sum w_order[i] for a subset S of {0..510} such that for all i,j in S, i is not in conflict_order[j] (and vice versa, symmetric).
    # Actually, if i in conflict_order[j], then j in conflict_order[i].
    
    # B&B Algorithm:
    # We iterate through positions i = 0 to 510.
    # At step i, we decide to include or exclude order[i].
    # If we include it, we remove all its conflicts from the available set for future steps.
    
    # State:
    # i: current index in order
    # available_mask: bitmask of positions >= i that are still available.
    # current_sum: sum of weights of selected vertices so far.
    
    # Upper Bound:
    # The maximum additional sum is the sum of w_order[j] for all j in available_mask.
    # We can compute this on the fly or maintain a running sum of available weights.
    # Let's maintain `avail_sum` as the sum of weights of all vertices in available_mask.
    
    def solve(i, available_mask, current_sum, selected):
        nonlocal best_sum, best_set
        
        if i == 511:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = selected[:]
            return
        
        # Upper bound check
        # Calculate sum of weights in available_mask
        # To make this fast, we can precompute prefix sums? No, mask is arbitrary.
        # We can just iterate bits. 512 bits is ~8 64-bit words.
        # Or, we can pass `avail_sum` as a parameter and update it.
        
        # Let's pass avail_sum.
        # If current_sum + avail_sum <= best_sum, prune.
        if current_sum + avail_sum <= best_sum:
            return
            
        # Option 1: Exclude order[i]
        # The vertex i is not picked. It is no longer available for future steps (since we move to i+1).
        # But wait, `available_mask` only contains vertices >= i.
        # If we exclude i, we just move to i+1. The mask for i+1 should not contain i.
        # Our `available_mask` is defined such that bit j is 1 if vertex j is available.
        # We iterate i from 0 to 510.
        # At step i, we consider vertex i.
        # If we exclude it, we proceed to i+1 with the same available_mask (but bit i is effectively gone because we won't look at it again).
        # Actually, it's cleaner to define `available_mask` as the set of vertices from i to 510 that are still selectable.
        
        # Let's refine the state.
        # i: next vertex to consider.
        # available_mask: bitmask where bit j is 1 if vertex j (j>=i) is available.
        # Note: bits < i are 0.
        
        # If i is not in available_mask, we just skip to i+1.
        if not (available_mask & (1 << i)):
            solve(i + 1, available_mask, current_sum, selected)
            return
        
        # Option 1: Exclude i
        # We don't pick i. We move to i+1.
        # The available mask for i+1 is the same, but we ignore bit i.
        # We can just call solve(i+1, available_mask, current_sum, selected)
        # However, we should clear bit i to be clean, or just rely on the loop index.
        # Since we only look at bits >= i, and we move to i+1, bit i is never checked again.
        
        solve(i + 1, available_mask, current_sum, selected)
        
        # Option 2: Include i
        # We pick i.
        # New sum = current_sum + w_order[i]
        # New available mask: remove i and all vertices conflicting with i.
        # conflicts = conflict_order[i]
        # new_mask = available_mask & ~conflicts & ~(1 << i)  # i is in conflicts? gcd(a,a)>1 for a>1. For a=1, gcd(1,1)=1.
        # Wait, if a=1, conflict_order[pos[1]] should not have bit pos[1] set.
        # Let's check: gcd(1,1)=1. So 1 does not conflict with itself.
        # For a>1, gcd(a,a)=a>1. So a conflicts with itself.
        
        # We must ensure we don't pick i twice.
        # If i is in conflict_order[i], it will be removed.
        # If not (a=1), we must manually remove it.
        
        new_mask = available_mask & ~conflict_order[i]
        # Ensure i is removed
        new_mask &= ~(1 << i)
        
        # Calculate new avail_sum
        # We need to subtract weights of removed vertices.
        # Removed vertices are those in (available_mask & conflict_order[i]) plus i if not in conflict.
        # Actually, new_avail_sum = avail_sum - sum(w_order[j] for j in (available_mask & ~new_mask))
        
        removed_mask = available_mask & ~new_mask
        # Calculate sum of weights in removed_mask
        rem_sum = 0.0
        m = removed_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            rem_sum += w_order[idx]
            m ^= lsb
        
        new_avail_sum = avail_sum - rem_sum
        
        # Check bound before recursing?
        # The recursive call will do it, but we can do it here to avoid function call overhead.
        if current_sum + w_order[i] + new_avail_sum > best_sum:
            selected.append(order[i])
            solve(i + 1, new_mask, current_sum + w_order[i], selected)
            selected.pop()
            
    # Initialize
    # Initial available mask: all 511 bits set.
    initial_mask = (1 << 511) - 1
    initial_avail_sum = sum(w_order)
    
    solve(0, initial_mask, 0.0, [])
    
    # The best_set contains the vertices in the order they were selected.
    # Return them as a list of integers.
    return best_set

# EVOLVE-BLOCK-END
