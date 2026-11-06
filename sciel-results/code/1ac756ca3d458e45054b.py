import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s kill (leaving 35s buffer)

    N = 1000

    # State
    mask = np.zeros(N + 1, dtype=bool)
    counts = np.zeros(2 * N + 1, dtype=np.int32)
    current_size = 0
    
    # Best tracking
    best_mask = np.zeros(N + 1, dtype=bool)
    best_size = 0

    def add_element(x):
        nonlocal current_size
        if mask[x]:
            return
        mask[x] = True
        current_size += 1
        # Update counts: for each y in S, sum s = x+y
        # We need to iterate over present elements.
        # To be fast, we maintain a list of present elements? 
        # Or just iterate 1..N and check mask? 
        # Iterating 1..N is 1000 ops. Checking mask is O(1). 
        # Updating counts is O(1) per pair.
        # Total O(N) per add/remove.
        
        # Optimization: Precompute present list? 
        # Maintaining a list is tricky with removals.
        # Let's just iterate. 1000 iterations is very fast in Python/NumPy?
        # Actually, pure Python loop for 1000 items is ~50us. 
        # 1000 adds * 50us = 0.05s. This is fine.
        
        # Wait, we can vectorize the update?
        # counts[x+1 : x+N+1] += mask[1:N+1]
        # This is a numpy slice add. Much faster.
        
        # Let's try vectorized update.
        # mask is bool. We need int.
        # m = mask.astype(np.int32) # Expensive to cast every time?
        # Let's keep a separate int array for mask values? 
        # Or just cast once per call.
        
        m_int = mask.astype(np.int32)
        # Sums involving x are x + y where y in 1..N
        # Indices in counts: x+1 to x+N
        # Values to add: m_int[1..N]
        counts[x+1 : x+N+1] += m_int[1 : N+1]
        
        # Don't forget the diagonal x+x?
        # The convolution logic in the original code handled a<=b.
        # counts[s] = number of pairs (a,b) with a<=b, a+b=s.
        # When we add x, we add pairs (y, x) for y < x, and (x, x).
        # Wait, the original code used:
        # c = convolve(m, m) -> counts all pairs (a,b)
        # diag correction: if s even, and s/2 in set, add 1.
        # counts = (c + diag) // 2
        # This effectively counts (a,b) with a<=b.
        
        # My incremental update must match this definition.
        # Let's re-derive.
        # Let R(s) be count of pairs a<=b, a+b=s.
        # If we add x:
        # 1. Pairs (x, y) where y < x. Sum s = x+y. Since y<x, x>y, so pair is (y,x).
        #    These are new representations.
        # 2. Pair (x, x). Sum s = 2x.
        
        # So for y in S such that y < x:
        #    counts[x+y] += 1
        # For x itself:
        #    counts[2*x] += 1
        
        # How to do this vectorized?
        # We need to add 1 to counts[x+y] for all y in S with y < x.
        # This is not a simple contiguous slice add unless S is contiguous.
        
        # Alternative:
        # Maintain counts as "number of ordered pairs (a,b) with a<=b".
        # If we just maintain "number of pairs (a,b) with a,b in S" (ordered),
        # then adding x adds:
        # - (x, y) for all y in S
        # - (y, x) for all y in S
        # - (x, x)
        # Total 2*|S| + 1 increments? No.
        
        # Let's stick to the definition: count of a<=b.
        # When adding x:
        # - For each y in S with y < x: new pair (y, x). Sum s = x+y.
        # - Pair (x, x). Sum s = 2x.
        
        # So:
        # counts[2*x] += 1
        # For y in S where y < x: counts[x+y] += 1
        
        # Vectorized approach for the second part:
        # We need to add 1 to indices x + y for all y in S, y < x.
        # Let Y = [y for y in S if y < x].
        # This requires gathering indices.
        
        # Given N=1000, a Python loop over present elements might be acceptable if optimized?
        # Or we can use np.where(mask[:x]) to get indices.
        
        # Let's try a hybrid:
        # 1. counts[2*x] += 1
        # 2. present_smaller = np.where(mask[1:x])[0] + 1  # indices of y < x
        # 3. if len(present_smaller) > 0:
        #      counts[x + present_smaller] += 1
        
        # This is O(N) memory access, but vectorized.
        
        # Let's implement this.
        
        # Revert the previous vectorized slice attempt logic in my head, 
        # the one above was for ordered pairs or something else.
        # Let's implement the correct one.
        
        # Remove the previous incorrect update logic if I had written it.
        # I haven't written it to the array yet in this function body, 
        # I just described it. 
        # So I will implement the correct one now.
        
        # Correction: I already did `counts[x+1 : x+N+1] += m_int[1 : N+1]` in the draft above? 
        # No, I just thought about it. I need to write the actual code.
        
        # Let's rewrite add_element properly.
        
    # Let's restart the function definition with the correct logic.
    
    def add_element(x):
        nonlocal current_size
        if mask[x]:
            return
        mask[x] = True
        current_size += 1
        
        # 1. Diagonal
        counts[2 * x] += 1
        
        # 2. Pairs (y, x) where y < x
        # Find all y in S such that y < x
        # mask is 0-indexed, but we use 1..N.
        # mask[:x] covers indices 0..x-1. We want 1..x-1.
        present_smaller = np.where(mask[1:x])[0] + 1
        if len(present_smaller) > 0:
            counts[x + present_smaller] += 1

    def remove_element(x):
        nonlocal current_size
        if not mask[x]:
            return
        mask[x] = False
        current_size -= 1
        
        # 1. Diagonal
        counts[2 * x] -= 1
        
        # 2. Pairs (y, x) where y < x
        present_smaller = np.where(mask[1:x])[0] + 1
        if len(present_smaller) > 0:
            counts[x + present_smaller] -= 1

    def is_valid():
        return np.max(counts) <= 2

    # Initial Greedy Construction
    # Use the fast incremental update
    for x in range(1, N + 1):
        # Check if adding x is valid
        # We need to simulate the counts change.
        # Instead of modifying and reverting, we can check if any count would exceed 2.
        # Max count currently is M.
        # Adding x increases counts[2x] by 1.
        # And counts[x+y] by 1 for y < x in S.
        
        # If counts[2x] == 2, adding x makes it 3 -> Invalid.
        if counts[2 * x] >= 2:
            continue
            
        # Check sums x+y for y < x in S.
        present_smaller = np.where(mask[1:x])[0] + 1
        if len(present_smaller) > 0:
            sums = x + present_smaller
            # If any of these sums have count >= 2, adding x makes them 3.
            if np.max(counts[sums]) >= 2:
                continue
        
        # Safe to add
        add_element(x)
        
    best_mask = mask.copy()
    best_size = current_size

    # Local Search Phase
    # Strategy:
    # 1. Try to add a random element.
    # 2. If invalid, identify the "blocking" sums (those with count 2 that would become 3).
    # 3. Try to remove an element from one of the blocking pairs to allow the addition.
    #    This is a "swap" or "remove-then-add" heuristic.
    
    rng = random.Random(42)
    
    # To speed up sampling, maintain a list of present and absent?
    # Or just use np.where.
    
    # Time budget for local search
    t0 = time.time()
    ls_deadline = deadline - 2
    
    while time.time() < ls_deadline:
        # 1. Attempt Add
        # Pick random candidate
        b = rng.randint(1, N)
        if mask[b]:
            continue
            
        # Check validity of adding b
        # Simulate:
        # New counts would be counts + delta.
        # We just need to check if max(counts + delta) <= 2.
        
        # Delta is non-zero at:
        # 2*b: +1
        # b+y for y < b in S: +1
        
        valid = True
        
        # Check diagonal
        if counts[2 * b] >= 2:
            valid = False
        else:
            # Check off-diagonals
            present_smaller = np.where(mask[1:b])[0] + 1
            if len(present_smaller) > 0:
                sums = b + present_smaller
                if np.max(counts[sums]) >= 2:
                    valid = False
        
        if valid:
            add_element(b)
            if current_size > best_size:
                best_size = current_size
                best_mask = mask.copy()
            continue
        
        # 2. Invalid Add. Try to fix.
        # Find which sums are blocking.
        # A sum s is blocking if counts[s] == 2 and (s == 2*b or s in {b+y | y<b, y in S}).
        # Actually, we just need to find *any* pair (a, c) in S such that a+c is one of the affected sums,
        # and removing one of them makes the add valid.
        
        # Simplified: 
        # The sums that would exceed 2 are:
        # S_block = {s | s in affected_sums and counts[s] == 2}
        
        affected_sums = set()
        if counts[2 * b] == 2:
            affected_sums.add(2 * b)
        present_smaller = np.where(mask[1:b])[0] + 1
        if len(present_smaller) > 0:
            sums = b + present_smaller
            # Check which of these have count 2
            blocking_sums = sums[counts[sums] == 2]
            affected_sums.update(blocking_sums.tolist())
        
        if not affected_sums:
            # This shouldn't happen if valid was False, but just in case
            continue
            
        # For each blocking sum s, there are pairs (a, c) with a<=c, a+c=s, a,c in S.
        # If we remove a or c, the count of s drops to 1.
        # Then adding b might be valid (if no other sums are blocked).
        
        # Strategy: Pick a random blocking sum s.
        # Find all pairs (a, c) in S summing to s.
        # Pick a random pair. Remove one of them.
        # Then try to add b.
        # If b adds successfully, we net +1 size? No, we removed 1, added 1. Size same.
        # But we might have "unlocked" b. 
        # Wait, if we remove a, size drops by 1. Then add b, size goes back to original.
        # This is a swap.
        # To improve size, we need to add b *without* removing, or remove one and add *two*?
        # Since we can only add one b at a time, a swap keeps size constant.
        # However, the new configuration might allow *future* adds.
        
        # So, perform a swap: Remove a blocker, Add b.
        
        s = rng.choice(list(affected_sums))
        
        # Find pairs (a, c) in S such that a+c=s, a<=c.
        # a can range from max(1, s-N) to s//2.
        # c = s - a.
        # Check if mask[a] and mask[c].
        
        candidates = []
        a_min = max(1, s - N)
        a_max = s // 2
        for a in range(a_min, a_max + 1):
            c = s - a
            if mask[a] and mask[c]:
                candidates.append((a, c))
                
        if not candidates:
            continue
            
        a, c = rng.choice(candidates)
        
        # Remove a (or c)
        # Let's remove a.
        remove_element(a)
        
        # Now try to add b
        # Re-check validity of adding b
        valid = True
        if counts[2 * b] >= 2:
            valid = False
        else:
            present_smaller_new = np.where(mask[1:b])[0] + 1
            if len(present_smaller_new) > 0:
                sums_new = b + present_smaller_new
                if np.max(counts[sums_new]) >= 2:
                    valid = False
                    
        if valid:
            add_element(b)
            # Size is same as before swap (removed 1, added 1).
            # But state changed.
        else:
            # Swap failed to make b valid. 
            # We have removed a. Should we restore a?
            # If we restore a, we are back to square one.
            # If we keep a removed, we have a smaller set.
            # It's better to restore a to maintain size, unless we want to explore smaller sets.
            # Given the objective is max size, maintaining size is good.
            add_element(a)
            
    # Finalize
    if current_size > best_size:
        best_mask = mask.copy()
        best_size = current_size

    result = [int(x) for x in np.where(best_mask)[0] if x >= 1]
    return result
