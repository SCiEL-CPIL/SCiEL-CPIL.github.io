"""Fixed program: bitmask-accelerated 4-AP-free construction + Local Search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256

    # Precompute 4-AP masks and per-element 4-AP lists for fast checking
    # elem_4ap[x] contains masks of 4-APs that include x
    elem_4ap = [[] for _ in range(N + 1)]
    all_4ap_masks = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            a1, a2, a3, a4 = a, a + d, a + 2 * d, a + 3 * d
            m = (1 << a1) | (1 << a2) | (1 << a3) | (1 << a4)
            all_4ap_masks.append(m)
            elem_4ap[a1].append(m)
            elem_4ap[a2].append(m)
            elem_4ap[a3].append(m)
            elem_4ap[a4].append(m)

    # Precompute 3-AP masks and per-element 3-AP lists
    # elem_3ap[x] contains masks of 3-APs that include x
    elem_3ap = [[] for _ in range(N + 1)]
    all_3ap_masks = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            a1, a2, a3 = a, a + d, a + 2 * d
            m = (1 << a1) | (1 << a2) | (1 << a3)
            all_3ap_masks.append(m)
            elem_3ap[a1].append(m)
            elem_3ap[a2].append(m)
            elem_3ap[a3].append(m)

    def count_3aps(mask):
        c = 0
        for m in all_3ap_masks:
            if (mask & m) == m:
                c += 1
        return c

    def is_valid_4ap_free(mask):
        # Check if mask contains any 4-AP
        for m in all_4ap_masks:
            if (mask & m) == m:
                return False
        return True

    def check_add_valid(mask, x):
        # Check if adding x to mask creates a 4-AP
        # Only need to check 4-APs containing x
        bit_x = 1 << x
        new_mask = mask | bit_x
        for m in elem_4ap[x]:
            if (new_mask & m) == m:
                return False
        return True

    def get_3ap_gain(mask, x):
        # Calculate how many 3-APs are completed by adding x to mask
        # A 3-AP is completed if the other 2 elements are in mask
        bit_x = 1 << x
        gain = 0
        for m in elem_3ap[x]:
            # m contains x. Let m' be m without x.
            # If (mask & m') == m', then adding x completes it.
            # But we must ensure x is not already in mask (it isn't, we are adding it)
            # And the other two are in mask.
            if (mask & (m ^ bit_x)) == (m ^ bit_x):
                gain += 1
        return gain

    def get_3ap_loss(mask, x):
        # Calculate how many 3-APs are lost by removing x from mask
        bit_x = 1 << x
        loss = 0
        for m in elem_3ap[x]:
            # If the 3-AP was complete in mask, it will be lost.
            if (mask & m) == m:
                loss += 1
        return loss

    # ---------- Greedy Seed ----------
    def greedy_seed():
        mask = 0
        # Start with a random permutation to break symmetry
        elems = list(range(1, N + 1))
        random.shuffle(elems)
        
        # Greedy addition: add element with max 3-AP gain if valid
        # To speed up, we can just iterate and add if valid, prioritizing gain
        # A simple greedy: iterate 1..N, add if valid. Then refine.
        # Better greedy: try all, pick best gain.
        
        # Let's do a simple greedy pass first to get a decent base
        for x in elems:
            if check_add_valid(mask, x):
                mask |= (1 << x)
        
        # Refine the greedy set with local search
        return mask

    # ---------- Local Search (Hill Climbing) ----------
    def local_search(mask, score, time_limit):
        best_mask = mask
        best_score = score
        elems_in_mask = [i for i in range(1, N + 1) if (mask >> i) & 1]
        elems_out_mask = [i for i in range(1, N + 1) if not ((mask >> i) & 1)]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all swaps: remove r, add a
            # To avoid O(N^2) with heavy checks, we sample or iterate efficiently
            # Given N=256, N^2 = 65k. Each check is fast.
            # We can iterate over all r in elems_in and all a in elems_out
            
            # Optimization: Pre-compute gains/losses? 
            # Loss depends on mask. Gain depends on mask.
            # Let's just do a full scan of swaps if time permits.
            
            # To be safe on time, we'll do random swaps if the set is large, 
            # but for N=256, a full scan of ~ (100*150) = 15k swaps is fast.
            
            # Let's try all single additions first (if possible to increase size)
            # Actually, the problem asks to maximize 3-APs. Larger sets usually have more 3-APs 
            # but are harder to keep 4-AP free. 
            # The optimal size is likely around 1/3 of N or similar (Behrend's set size is ~ N/log^c N).
            # For N=256, maybe size 50-80.
            
            # Strategy: 
            # 1. Try to add elements that increase score.
            # 2. Try to swap elements to increase score.
            
            # Phase 1: Additions
            added = True
            while added and time.time() < time_limit:
                added = False
                for a in elems_out_mask:
                    if time.time() > time_limit: break
                    if check_add_valid(mask, a):
                        gain = get_3ap_gain(mask, a)
                        if gain > 0: # Only add if it improves score
                            mask |= (1 << a)
                            score += gain
                            if score > best_score:
                                best_score = score
                                best_mask = mask
                            added = True
                            # Update lists
                            elems_in_mask.append(a)
                            elems_out_mask.remove(a)
                            # Restart inner loop to re-evaluate
                            break # Restart to pick best next
            
            # Phase 2: Swaps
            swapped = True
            while swapped and time.time() < time_limit:
                swapped = False
                # Try to find a swap that improves score
                # To make it faster, we can randomize the order
                random.shuffle(elems_in_mask)
                for r in elems_in_mask:
                    if time.time() > time_limit: break
                    base_mask = mask & ~(1 << r)
                    loss = get_3ap_loss(mask, r)
                    
                    # Try to find an 'a' to add to base_mask
                    # We want gain > loss
                    # Sort candidates by potential? Just iterate.
                    # To speed up, we can check a subset or all.
                    # Let's iterate all 'a' in elems_out_mask
                    for a in elems_out_mask:
                        if time.time() > time_limit: break
                        # Check if a is valid in base_mask
                        # Note: check_add_valid checks against current mask. 
                        # We need to check against base_mask.
                        # We can reuse the logic: check if base_mask | (1<<a) has 4-AP.
                        # This is equivalent to check_add_valid(base_mask, a)
                        if check_add_valid(base_mask, a):
                            gain = get_3ap_gain(base_mask, a)
                            if gain > loss:
                                # Perform swap
                                mask = base_mask | (1 << a)
                                score += gain - loss
                                if score > best_score:
                                    best_score = score
                                    best_mask = mask
                                # Update lists
                                elems_in_mask.remove(r)
                                elems_in_mask.append(a)
                                elems_out_mask.remove(a)
                                elems_out_mask.append(r)
                                swapped = True
                                break # Restart to find next improvement
                    if swapped:
                        break
                        
        return best_mask, best_score

    # ---------- Main Loop ----------
    best_mask = 0
    best_score = 0
    
    # Run multiple restarts
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        mask = greedy_seed()
        score = count_3aps(mask)
        
        # Refine
        budget = deadline - 5.0
        mask, score = local_search(mask, score, budget)
        
        if score > best_score:
            best_score = score
            best_mask = mask

    result = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    if not result:
        # Fallback
        result = [1, 2, 4, 8, 16, 32, 64, 128]
    return sorted(result)
# EVOLVE-BLOCK-END
