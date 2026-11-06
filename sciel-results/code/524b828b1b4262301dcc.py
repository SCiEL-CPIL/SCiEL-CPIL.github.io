import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin

    # Generate all 4-element subsets of [16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    rng = random.Random(42)
    
    # Precompute pairwise intersections
    inter_table = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            ij = mi & all_masks[j]
            inter_table[i][j] = ij
            inter_table[j][i] = ij

    def is_valid_addition(fam_list, fam_set, new_idx):
        """
        Check if all_masks[new_idx] can be added to fam_list.
        fam_list: list of indices
        fam_set: set of indices (for fast membership, though not strictly needed for validity logic, 
                 but useful if we check 'in' inside, but we don't here. 
                 Actually, we need to ensure new_idx is not in fam_set, but caller ensures that.)
        """
        # Group existing masks by their intersection with new_idx
        # Intersections are 4-bit masks, so 0-15
        groups = [[] for _ in range(16)]
        
        for a in fam_list:
            inter = inter_table[a][new_idx]
            groups[inter].append(a)
        
        # For each group, check if any pair has intersection equal to the group key
        for k in range(16):
            g = groups[k]
            if len(g) < 2:
                continue
            # Check pairs
            n_g = len(g)
            for i in range(n_g):
                a = g[i]
                for j in range(i + 1, n_g):
                    b = g[j]
                    if inter_table[a][b] == k:
                        return False
        return True

    best_family = []
    best_size = 0

    # Phase 1: Multiple greedy trials with different orderings
    # We can do many restarts because greedy is fast
    max_greedy_restarts = 200
    for trial in range(max_greedy_restarts):
        if time.time() > deadline - 20:
            break
            
        order = list(range(n_total))
        rng.shuffle(order)
        
        family = []
        family_set = set()
        
        for idx in order:
            if is_valid_addition(family, family_set, idx):
                family.append(idx)
                family_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

    # Phase 2: Local Search (Swaps)
    # Start from the best greedy result
    cur_family = list(best_family)
    cur_set = set(cur_family)
    cur_size = len(cur_family)
    
    # Keep track of best in local search
    ls_best_family = list(cur_family)
    ls_best_size = cur_size
    
    # Time remaining for local search
    # We reserve 2s for finalization
    
    ls_start = time.time()
    max_ls_time = deadline - 2 - ls_start
    
    # Local Search Strategy:
    # 1. Try to add a new set without removing.
    # 2. Try to swap: remove one, add one.
    
    # To speed up finding a valid swap, we shuffle candidates.
    
    step = 0
    # We run until time runs out
    while time.time() < deadline - 2:
        step += 1
        
        # 1. Try to add a random set (if size < 250)
        if cur_size < 250:
            # Sample a few candidates
            num_samples = min(50, n_total)
            candidates = rng.sample(range(n_total), num_samples)
            added = False
            for idx in candidates:
                if idx in cur_set:
                    continue
                if is_valid_addition(cur_family, cur_set, idx):
                    cur_family.append(idx)
                    cur_set.add(idx)
                    cur_size += 1
                    added = True
                    if cur_size > ls_best_size:
                        ls_best_size = cur_size
                        ls_best_family = list(cur_family)
                    break
            if added:
                continue # Restart loop to try adding more or swapping
        
        # 2. Try to swap: remove a random element, try to add a better one
        if cur_family:
            rem_pos = rng.randrange(len(cur_family))
            rem_idx = cur_family[rem_pos]
            
            # Try to find a replacement
            # Sample candidates
            num_samples = min(100, n_total)
            candidates = rng.sample(range(n_total), num_samples)
            
            # To avoid checking the removed element itself or existing ones
            # We check validity against cur_family WITHOUT rem_idx
            
            found_swap = False
            for add_idx in candidates:
                if add_idx in cur_set:
                    continue
                
                # Temporarily simulate removal
                # We can just pass a filtered list, but that's O(N).
                # Optimization: 
                # is_valid_addition iterates over fam_list.
                # Let's just create the check list. N is ~250. 250 ops is fast.
                check_list = [x for x in cur_family if x != rem_idx]
                
                if is_valid_addition(check_list, cur_set, add_idx):
                    # Perform swap
                    cur_family[rem_pos] = add_idx
                    cur_set.discard(rem_idx)
                    cur_set.add(add_idx)
                    # cur_size remains same
                    found_swap = True
                    
                    # If size increased? No, swap keeps size same.
                    # But it might enable future additions.
                    break
            
            if not found_swap:
                # Occasionally, if we get stuck, try a "kick" or restart?
                # Or just continue.
                pass

    # Update best if local search found something better (unlikely to exceed greedy max if greedy is good, 
    # but swaps might allow reaching a larger size if greedy got stuck at a local maximum)
    if ls_best_size > best_size:
        best_size = ls_best_size
        best_family = ls_best_family

    # Final Polish: Try to add any remaining sets to the best family found
    final_family = list(best_family)
    final_set = set(final_family)
    
    # Iterate through all sets in random order to see if we can add more
    remaining_order = list(range(n_total))
    rng.shuffle(remaining_order)
    
    for idx in remaining_order:
        if idx in final_set:
            continue
        if is_valid_addition(final_family, final_set, idx):
            final_family.append(idx)
            final_set.add(idx)
            if len(final_family) > best_size:
                best_size = len(final_family)
                best_family = list(final_family)
            if len(final_family) >= 250:
                break

    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
