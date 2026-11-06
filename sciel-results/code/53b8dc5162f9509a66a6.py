import time
import random

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Precompute all 4-APs
    aps = []
    ap_full_masks = []
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = b + d
            dap = c + d
            ap = (a, b, c, dap)
            aps.append(ap)
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << dap)
            ap_full_masks.append(mask)
            
    num_aps = len(aps)
    
    # 2. For each element, list the AP indices it belongs to
    aps_containing = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for elem in ap:
            aps_containing[elem].append(i)
            
    # 3. Greedy Construction
    def greedy_randomized():
        selected = 0  # Bitset
        size = 0
        order = list(range(N))
        random.shuffle(order)
        
        for x in order:
            valid = True
            for ap_idx in aps_containing[x]:
                others_mask = ap_full_masks[ap_idx] ^ (1 << x)
                if (selected & others_mask) == others_mask:
                    valid = False
                    break
            
            if valid:
                selected |= (1 << x)
                size += 1
                
        return selected, size

    # 4. Local Search
    def local_search(selected, size):
        best_set = selected
        best_size = size
        
        while time.time() < deadline - 5.0:
            improved = False
            # Try to add
            not_in_set = [i for i in range(N) if not (selected & (1 << i))]
            if not not_in_set:
                break
            
            # Try a few random candidates
            num_tries = min(50, len(not_in_set))
            candidates = random.sample(not_in_set, num_tries)
            
            for x in candidates:
                # Check if x can be added
                conflict_aps = []
                for ap_idx in aps_containing[x]:
                    others_mask = ap_full_masks[ap_idx] ^ (1 << x)
                    if (selected & others_mask) == others_mask:
                        conflict_aps.append(ap_idx)
                
                if not conflict_aps:
                    # Can add
                    selected |= (1 << x)
                    size += 1
                    if size > best_size:
                        best_size = size
                        best_set = selected
                    improved = True
                    break
                else:
                    # Try to resolve by swapping
                    # Pick a random conflict AP
                    ap_idx = random.choice(conflict_aps)
                    ap = aps[ap_idx]
                    # Elements in this AP that are in selected and not x
                    blockers = [e for e in ap if e != x and (selected & (1 << e))]
                    
                    if not blockers:
                        continue
                        
                    y = random.choice(blockers)
                    
                    # Try swap: remove y, add x
                    # Check if x is valid in (selected ^ (1<<y))
                    temp_selected = selected ^ (1 << y)
                    x_valid = True
                    for ap_idx2 in aps_containing[x]:
                        others_mask2 = ap_full_masks[ap_idx2] ^ (1 << x)
                        if (temp_selected & others_mask2) == others_mask2:
                            x_valid = False
                            break
                    
                    if x_valid:
                        # Perform swap
                        selected = temp_selected | (1 << x)
                        # Size unchanged
                        # Check if this allows an immediate add? 
                        # We loop back to the top of the while loop which tries to add.
                        # To avoid infinite swap loops, we can accept the swap only if it "improves" some metric?
                        # Or just accept it. The randomization helps escape.
                        improved = True
                        break
            
            if not improved:
                # If no add or swap succeeded, try a random perturbation
                # Remove a random element
                if size > 0:
                    in_set = [i for i in range(N) if (selected & (1 << i))]
                    if in_set:
                        y = random.choice(in_set)
                        selected ^= (1 << y)
                        size -= 1
                        # This might allow adds later
                        # Don't mark as improved to encourage trying again?
                        # Or mark as improved to avoid immediate repeat?
                        # Let's not mark, so we try to add in next iteration.
                
        return best_set, best_size

    # Run multiple restarts
    best_set_overall = 0
    best_size_overall = 0
    
    # Initial greedy runs
    # Do a few quick greedy runs
    for _ in range(100):
        if time.time() > deadline - 20.0:
            break
        s, sz = greedy_randomized()
        if sz > best_size_overall:
            best_size_overall = sz
            best_set_overall = s
            
    # Local search on the best found so far
    if time.time() < deadline - 5.0:
        s, sz = local_search(best_set_overall, best_size_overall)
        if sz > best_size_overall:
            best_size_overall = sz
            best_set_overall = s

    # Convert bitset to list of 1-based integers
    result = []
    for i in range(N):
        if best_set_overall & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
