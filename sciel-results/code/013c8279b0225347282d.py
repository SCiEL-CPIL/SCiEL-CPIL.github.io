import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256
    all_nums = list(range(1, N + 1))

    # Precompute conflict masks: conflict_masks[i] has bit j set if (i*j)%(i+j)==0
    conflict_masks = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << j)
        conflict_masks[i] = mask

    def get_degree(v, mask):
        return (conflict_masks[v] & mask).bit_count()

    def greedy_min_degree(remaining_mask, rng):
        """
        Greedy minimum-degree independent set using bitmasks.
        Returns a mask of the selected vertices.
        """
        selected_mask = 0
        remaining = remaining_mask
        
        while remaining:
            # Find vertex with minimum degree in remaining
            min_deg = float('inf')
            best_v = None
            
            # Iterate through set bits in remaining
            r = remaining
            while r:
                # Get lowest set bit
                lsb = r & -r
                v = lsb.bit_length() - 1
                deg = get_degree(v, remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                r -= lsb
            
            if best_v is None:
                break
            
            selected_mask |= (1 << best_v)
            # Remove best_v and its neighbors from remaining
            # Neighbors are conflict_masks[best_v]
            # We want to remove best_v and all v such that conflict_masks[best_v] has bit v set
            # Also remove best_v itself
            to_remove = conflict_masks[best_v] | (1 << best_v)
            remaining &= ~to_remove
        
        return selected_mask

    def local_search_improve(selected_mask, rng, max_iters=50):
        """
        Local search: Try to remove 1 element and add as many as possible.
        """
        best_mask = selected_mask
        best_size = selected_mask.bit_count()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random element from selected to remove
            # Create list of set bits
            s_list = []
            m = selected_mask
            while m:
                lsb = m & -m
                s_list.append(lsb.bit_length() - 1)
                m -= lsb
            
            if not s_list:
                break
                
            remove_v = rng.choice(s_list)
            new_mask = selected_mask & ~(1 << remove_v)
            
            # Try to add elements greedily
            # Candidates are all numbers not in new_mask
            # To speed up, we can prioritize numbers with low degree in the "available" set
            # But for simplicity, let's just iterate through all candidates and add if compatible
            # To make it faster, we can shuffle candidates
            
            candidates = [i for i in range(1, N + 1) if not (new_mask >> i) & 1]
            rng.shuffle(candidates)
            
            added_any = False
            for v in candidates:
                # Check if v is compatible with new_mask
                if (conflict_masks[v] & new_mask) == 0:
                    new_mask |= (1 << v)
                    added_any = True
            
            if new_mask.bit_count() > best_size:
                best_size = new_mask.bit_count()
                best_mask = new_mask
                # If we improved, we can continue searching from this new state
                # Or reset to original? Usually, accept and continue.
                # But if we accept, the "remove" element changes.
                # Let's stick to best found so far to avoid getting stuck in a local loop
                # Actually, standard local search accepts improvements.
                # But we are modifying `selected_mask`? No, we use `new_mask`.
                # We should update `selected_mask` to `new_mask` if we want to chain improvements?
                # The current structure tries random removals from the *original* `selected_mask` each iteration?
                # No, `selected_mask` is constant in the loop.
                # So it tries 1-Remove, K-Add from the *current best*?
                # If we update `selected_mask`, we explore a new neighborhood.
                # Let's update `selected_mask` to `best_mask` if we found an improvement, to allow chaining.
                selected_mask = new_mask
            # Else, revert (do nothing, keep selected_mask as is)

        return best_mask

    # Main search
    best_solution_mask = 0
    best_size = 0

    # Strategy 1: Randomized Greedy Min-Degree
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(trial * 104729 + 17)
        # Start with all numbers
        remaining_mask = (1 << (N + 1)) - 2  # Bits 1..N set
        
        S_mask = greedy_min_degree(remaining_mask, rng)
        
        if S_mask.bit_count() > best_size:
            best_size = S_mask.bit_count()
            best_solution_mask = S_mask
        
        # Local search improvement
        if time.time() < deadline - 5:
            S_ls = local_search_improve(S_mask, rng, max_iters=20)
            if S_ls.bit_count() > best_size:
                best_size = S_ls.bit_count()
                best_solution_mask = S_ls

    # Strategy 2: Odds + Evens
    odds_mask = 0
    evens_mask = 0
    for i in range(1, N + 1):
        if i % 2 == 1:
            odds_mask |= (1 << i)
        else:
            evens_mask |= (1 << i)
            
    # Find evens compatible with ALL odds
    # An even e is safe if (conflict_masks[e] & odds_mask) == 0
    safe_evens_mask = 0
    e_list = []
    m = evens_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        if (conflict_masks[v] & odds_mask) == 0:
            safe_evens_mask |= lsb
            e_list.append(v)
        m -= lsb

    # Run greedy on safe_evens_mask
    # Note: The greedy function assumes it picks from `remaining_mask` and removes neighbors.
    # The neighbors of an even v might include other evens.
    # So we can run greedy_min_degree on safe_evens_mask.
    # However, greedy_min_degree removes neighbors from `remaining`.
    # The neighbors of v in the full graph are conflict_masks[v].
    # We only care about neighbors that are in `safe_evens_mask`.
    # The function `greedy_min_degree` uses `conflict_masks[best_v]` to remove.
    # `remaining` is updated as `remaining &= ~to_remove`.
    # Since `remaining` is initialized to `safe_evens_mask`, it will only remove evens.
    # So it works correctly.
    
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(trial * 31337 + 41)
        # Greedy on safe evens
        evens_S_mask = greedy_min_degree(safe_evens_mask, rng)
        
        # Combine with odds
        S_mask = odds_mask | evens_S_mask
        
        if S_mask.bit_count() > best_size:
            best_size = S_mask.bit_count()
            best_solution_mask = S_mask
            
        if time.time() < deadline - 5:
            S_ls = local_search_improve(S_mask, rng, max_iters=20)
            if S_ls.bit_count() > best_size:
                best_size = S_ls.bit_count()
                best_solution_mask = S_ls

    # Convert best mask to list
    best_solution = []
    m = best_solution_mask
    while m:
        lsb = m & -m
        best_solution.append(lsb.bit_length() - 1)
        m -= lsb

    # Fallback if empty (should not happen)
    if not best_solution:
        best_solution = [1]

    return best_solution
