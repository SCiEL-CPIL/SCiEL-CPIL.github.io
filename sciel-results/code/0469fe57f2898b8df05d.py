"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math

    # 1. Generate all primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # 2. Dynamic Programming with Pareto Frontiers
    # frontiers[i] is a list of (length, last_gap, predecessor_index)
    # We keep only non-dominated states: 
    # State A (l1, g1) dominates State B (l2, g2) if l1 >= l2 and g1 <= g2.
    
    frontiers = [[] for _ in range(n)]
    
    # Initialize the first prime
    frontiers[0].append((1, 0, -1))
    
    for i in range(1, n):
        pi = primes[i]
        new_states = []
        
        # We can always start a new sequence at pi
        # But we might extend from previous primes
        
        for j in range(i):
            pj = primes[j]
            gap = pi - pj
            
            # Check all states ending at j
            for l, g, pred in frontiers[j]:
                if g <= gap:
                    # We can extend this state
                    new_l = l + 1
                    new_g = gap
                    # The predecessor for this new state is j
                    # We need to track the full path or just the immediate predecessor?
                    # To reconstruct, we need to store the predecessor index j.
                    # But we also need to know which state in j's frontier led to this.
                    # Actually, since we store the state (l, g) at j, and we are adding (l+1, gap) at i,
                    # we can just store j as the immediate predecessor.
                    # However, if multiple states at j lead to the same (l+1, gap), it doesn't matter which j state we pick, 
                    # as long as we can trace back. 
                    # Wait, if we store (new_l, new_g, j), we can trace back to j. 
                    # Then at j, we need to find a state with length l and gap g.
                    # This might be ambiguous if there are multiple states with same l and g? 
                    # Unlikely but possible. 
                    # Better: store the index of the state in frontiers[j] as well? 
                    # Or simply, since we are building frontiers sequentially, 
                    # we can store the predecessor index j. 
                    # When reconstructing, at node i with state (L, G, pred_j), 
                    # we go to pred_j. At pred_j, we need to find a state with length L-1 and gap G_prev?
                    # No, the gap G in state at i is the gap between pred_j and i.
                    # The state at pred_j that we extended had gap G_prev <= G.
                    # We don't store G_prev in the state at i.
                    # So we need to be able to find the specific state at pred_j.
                    # If we store (l, g, j), we know we came from j with a state of length l-1? No, l is the new length.
                    # The state at j had length l-1.
                    # There might be multiple states at j with length l-1.
                    # We need to ensure we pick the one with gap <= current gap.
                    # Since we are storing the frontier, we can just pick any valid one during reconstruction?
                    # No, the validity is determined by the gap.
                    # Let's store the specific state index or just rely on the fact that if (l-1, g_prev) was in frontier j, 
                    # and g_prev <= gap, it's valid.
                    # To be safe, let's store the predecessor index j. 
                    # During reconstruction, we will look for a state in frontiers[j] with length l-1.
                    # If there are multiple, we pick one with gap <= current_gap.
                    
                    new_states.append((new_l, new_g, j))
        
        # Prune new_states to keep only Pareto optimal ones
        # Sort by length descending, then by gap ascending
        # Then iterate and keep a state if its gap is smaller than all previously kept states with >= length?
        # Actually, standard Pareto pruning:
        # 1. Sort by length descending.
        # 2. Iterate. Keep a state if its gap is strictly less than the minimum gap seen so far for lengths >= current length.
        
        # Combine with the trivial state (1, 0, -1) ? 
        # Actually, (1, 0, -1) is dominated by any state with length > 1. 
        # But if no extensions are possible, we need it.
        # Let's add it and prune.
        new_states.append((1, 0, -1))
        
        # Sort by length descending, then gap ascending
        new_states.sort(key=lambda x: (-x[0], x[1]))
        
        pruned = []
        min_gap_so_far = float('inf')
        # We want to keep states such that no other kept state has both >= length and <= gap.
        # Since we sort by length desc, the first state has max length.
        # For subsequent states, they have <= length. 
        # They are only useful if they have a strictly smaller gap than any state with higher (or equal) length.
        # Wait, if length is smaller, it can only be useful if its gap is smaller than all states with length >= its length.
        
        # Let's refine:
        # A state (l, g) is dominated if there exists (l', g') in the set such that l' >= l and g' <= g.
        # Sort by l desc.
        # Keep (l, g) if g < min_gap_seen_so_far.
        # Update min_gap_seen_so_far = min(min_gap_seen_so_far, g).
        
        for l, g, j in new_states:
            if g < min_gap_so_far:
                pruned.append((l, g, j))
                min_gap_so_far = g
            # If g == min_gap_so_far, it is dominated by the previous one (which has >= length and same gap). 
            # If g > min_gap_so_far, it is dominated.
            
        frontiers[i] = pruned

    # 3. Find the best state across all primes
    best_len = 0
    best_end_idx = -1
    best_state = None
    
    for i in range(n):
        for l, g, pred in frontiers[i]:
            if l > best_len:
                best_len = l
                best_end_idx = i
                best_state = (l, g, pred)
    
    if best_end_idx == -1:
        return primes[:1] # Fallback, though n>1 is guaranteed here

    # 4. Reconstruct the sequence
    seq = []
    cur_idx = best_end_idx
    cur_len = best_state[0]
    cur_gap = best_state[1]
    cur_pred = best_state[2]
    
    seq.append(primes[cur_idx])
    
    while cur_pred != -1:
        cur_idx = cur_pred
        seq.append(primes[cur_idx])
        
        # Find the state in frontiers[cur_idx] that corresponds to this step
        # The state must have length cur_len - 1
        # And its gap must be <= cur_gap (which is the gap from cur_idx to its successor)
        # Note: The gap stored in the state at cur_idx is the gap between its predecessor and cur_idx.
        # The gap used to extend to the successor is the successor - cur_idx.
        # The condition for extension was state_gap <= (successor - cur_idx).
        # So we need a state at cur_idx with length cur_len - 1 and gap <= cur_gap.
        
        target_len = cur_len - 1
        # Find a valid state
        found_pred = -1
        found_gap = 0
        for l, g, p in frontiers[cur_idx]:
            if l == target_len and g <= cur_gap:
                found_pred = p
                found_gap = g
                break
        
        if found_pred == -1:
            # This should not happen if logic is correct
            # But if it does, we break to avoid infinite loop
            break
            
        cur_pred = found_pred
        cur_gap = found_gap
        cur_len = target_len
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
