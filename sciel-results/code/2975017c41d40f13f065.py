import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    n = 256
    # 1. Precompute conflict bitsets
    # conflict[i] is an integer bitset where bit j is 1 if i and j conflict
    # We index 0..255 corresponding to numbers 1..256
    conflict = [0] * n
    
    for i in range(n):
        a = i + 1
        bits = 0
        for j in range(i + 1, n):
            b = j + 1
            s = a + b
            p = a * b
            if p % s == 0:
                bits |= (1 << j)
                # Symmetry: conflict[j] will need bit i, but we can just build both or use a helper
        conflict[i] = bits
        # We need the reverse direction for quick lookup if we iterate differently, 
        # but since we check (mask & conflict[v]), we need conflict[v] to contain bits of S.
        # The loop above sets bit j in conflict[i] if i conflicts with j.
        # When checking if v can be added to S, we check if (mask & conflict[v]) == 0.
        # conflict[v] must contain bits for all u in S that conflict with v.
        # The loop above correctly populates this if we ensure symmetry or just populate fully.
        # Let's populate fully to be safe and simple.
        # Actually, the loop above only sets bits j > i in conflict[i].
        # It does NOT set bits i < j in conflict[j].
        # So conflict[j] is missing lower bits.
        # Let's rebuild or fix.
        
    # Re-do conflict computation to be fully symmetric or just one-directional correct for the mask check.
    # Mask check: `mask` has bits set for elements in S.
    # We want to know if any element in S conflicts with v.
    # So we need `conflict[v]` to have bit u set if u conflicts with v.
    # Let's compute `conflict[v]` properly.
    
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(n):
            if i == j: continue
            b = j + 1
            s = a + b
            p = a * b
            if p % s == 0:
                conflict[i] |= (1 << j)
                
    # 2. Initial Set: All Odds
    # Odds are 1, 3, ..., 255. Indices 0, 2, ..., 254.
    mask = 0
    for i in range(0, n, 2):
        mask |= (1 << i)
        
    best_mask = mask
    best_size = bin(mask).count('1')
    
    # 3. Local Search
    # We will try to improve the size.
    # Helper to update mask and size
    def try_add(mask, v):
        # v is index
        if mask & (1 << v):
            return mask, False
        if mask & conflict[v]:
            return mask, False
        return mask | (1 << v), True

    # Local search loop
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # Step A: Try to add any free vertex
        added_any = True
        while added_any:
            added_any = False
            # Iterate through all vertices not in mask
            for v in range(n):
                if not (mask & (1 << v)):
                    if not (mask & conflict[v]):
                        mask |= (1 << v)
                        improved = True
                        added_any = True
                        break # Restart loop to pick up new freedoms
            
        if (time.time() - start_time) > deadline:
            break
            
        # Step B: Try 1-for-1 swaps that enable additions
        # For each v not in S, check conflicts C_v = S & conflict[v]
        # If |C_v| == 1, let u be the element.
        # Try removing u and adding v.
        # Then check if any w not in S_new can be added.
        # If yes, net gain +1.
        
        # To speed up, only try this for a subset of v's or limit iterations
        swap_found = False
        for v in range(n):
            if mask & (1 << v):
                continue
            
            confs = mask & conflict[v]
            # If 0 conflicts, it would have been added in Step A.
            # So confs is non-zero.
            # Check if exactly one bit is set.
            if confs and (confs & (confs - 1)) == 0:
                u_bit = confs
                u_idx = u_bit.bit_length() - 1
                
                # Remove u, Add v
                new_mask = (mask & ~u_bit) | (1 << v)
                
                # Check if we can add any w to new_mask
                # We don't need to check w if it's u (removed) or v (added)
                # We want to see if size increases.
                # Current size: |S|
                # New size after swap: |S|
                # If we can add w, new size: |S| + 1.
                
                # Optimization: Only check w if there's a candidate.
                # Candidates are those not in new_mask.
                # Specifically, w must not conflict with new_mask.
                # (new_mask & conflict[w]) == 0.
                
                # We can just try to add greedily to new_mask to see if it grows.
                # But that's expensive. 
                # Heuristic: If we can add *any* w, we do it.
                
                # Let's check if there exists w such that (new_mask & conflict[w]) == 0 and w not in new_mask.
                # To make this fast, we can just do a quick scan.
                # But doing a full scan for every single swap candidate (256 * 256) might be slow if not optimized.
                # 256 * 256 = 65k checks. Each check is bit ops. Fast enough.
                
                can_add_w = False
                for w in range(n):
                    if new_mask & (1 << w):
                        continue
                    if not (new_mask & conflict[w]):
                        can_add_w = True
                        break
                        
                if can_add_w:
                    # Perform the swap and add the w we found
                    # We need to find which w. The loop above breaks on first one.
                    # Let's refine: just do the swap, then run Step A (add free vertices) on the new mask.
                    # If Step A adds anything, size increases.
                    mask = new_mask
                    improved = True
                    swap_found = True
                    break # Restart main loop to clean up with Step A

        if not swap_found:
            # Try 2-for-3 or deeper? 
            # For now, 1-for-1 is a good baseline.
            # If stuck, we can try removing 2 elements to add 1, if it allows adding 3?
            # Given time, let's stick to 1-for-1 and greedy add.
            # If no improvement in a full pass, break?
            # The `improved` flag handles this. If no add and no swap, improved remains False.
            # But wait, if `improved` was True from Step A, we loop again.
            # If Step A added nothing and Step B found no swap, improved is False, loop ends.
            
            # To escape local minima, we could add some randomness or try 2-for-k.
            # But let's ensure we at least run the loop until stable.
            
            # If we get stuck, try a different initial seed?
            # The current seed is Odds. It's very strong.
            # Let's try one more thing: 2-for-1 swap? (Remove 2, Add 1). Size decreases. Not useful unless it enables +3 later.
            # Too complex for remaining time.
            
            # Let's try random restarts if we have time?
            # No, the deterministic local search on Odds is likely optimal or near-optimal.
            
            # One more heuristic: If stuck, try removing a "high conflict" element and adding a "low conflict" element?
            # This is what the 1-for-1 swap effectively does if it enables addition.
            
            # If we are truly stuck, we break.
            break

    # Convert mask to list
    result = []
    for i in range(n):
        if mask & (1 << i):
            result.append(i + 1)
            
    return result

# Verification
# The logic seems sound.
# 1. Odds are independent.
# 2. Greedy addition finds any "free" spots.
# 3. 1-for-1 swap checks if removing a blocker allows adding the candidate AND potentially more.
#    Actually, my code just does the swap and sets `improved=True`, then breaks to restart the loop.
#    The restart will then run Step A (Greedy Add) on the new mask.
#    If the swap opened up a spot for `w`, Step A will find and add `w`.
#    This effectively implements "Swap u->v, then add w".
#    Net change: -u +v +w. Size change: +1.
#    This is correct.

# Potential Issue: Time limit.
# The inner loops in Step B iterate v (256) and for each, might iterate w (256).
# Worst case 65k iterations per pass.
# Number of passes? At most 256 improvements.
# 256 * 65k = 16.6M iterations.
# Python bit ops are fast, but 16M might take a few seconds.
# Should be well within 50s.

# Let's double check the conflict calculation.
# `conflict[i]` has bit `j` set if `i` and `j` conflict.
# `mask` has bits for elements in S.
# `mask & conflict[v]` gives the set of elements in S that conflict with v.
# This is correct.

# One small optimization: Precompute `conflict` as a list of ints.
# The current code does this.

# Final check on constraints:
# "Return a nonempty subset of [1, 256]" - Yes.
# "Maximize cardinality" - Yes, local search.
# "60s limit" - Yes, deadline guard.

# Is there a better initial set?
# What if we start empty and greedy add?
# Greedy add usually finds a smaller set than starting from a large known independent set and expanding.
# Odds (128) is a very large set. Expanding it is the right strategy.

# Can we improve the swap logic?
# Currently: If |conf(v)| == 1, swap.
# What if |conf(v)| == 2?
# We could remove both and add v. Size change: -2 + 1 = -1.
# Unless it allows adding 3 more? Unlikely to check efficiently.
# Sticking with |conf(v)| == 1 is safe and effective.

# What if we just do greedy add from a random permutation?
# Sometimes random greedy finds better solutions than structured ones if the structure is misleading.
# But Odds is structurally perfect (no internal edges).
# So it's a valid core. We just attach leaves (evens) to it.
# The graph of conflicts is likely sparse enough that this works well.

# One edge case: `confs & (confs - 1) == 0` checks for power of 2 (single bit).
# Correct.

# The code looks solid.
