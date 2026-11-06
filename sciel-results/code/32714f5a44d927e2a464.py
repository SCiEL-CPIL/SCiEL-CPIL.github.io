"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

def _mask_to_list(m):
    return [i + 1 for i in range(16) if m & (1 << i)]

def _forms_sunflower_mask(a, b, c):
    """Check if three bitmask sets form a sunflower: all three pairwise intersections equal."""
    return (a & b) == (a & c) == (b & c)

def _greedy_with_order(all_masks, order, deadline):
    chosen_masks = []
    # conflict_count[i] = number of pairs (j,k) in chosen such that (chosen[j], chosen[k], candidate_i) forms sunflower
    # We track this incrementally: when we add a new mask, for each existing pair, check if it conflicts with each candidate
    # Actually, let's use a different approach: for each candidate, count how many pairs in chosen would form a sunflower with it
    # conflict[i] = number of pairs (j,k) in chosen such that sunflower(chosen[j], chosen[k], all_masks[i])
    
    n_total = len(all_masks)
    conflict = [0] * n_total
    
    for idx in order:
        if time.time() >= deadline:
            break
        m = all_masks[idx]
        if conflict[idx] > 0:
            continue
        
        # Add m to chosen
        chosen_masks.append(m)
        c_idx = len(chosen_masks) - 1
        
        # Update conflict counts for all other candidates
        for i in range(len(chosen_masks) - 1):
            ci = chosen_masks[i]
            for j in range(n_total):
                if j == idx or conflict[j] > 0:
                    continue
                cj = all_masks[j]
                if _forms_sunflower_mask(ci, cj, m):
                    conflict[j] += 1
                # Also need to check pairs (ci, m) with candidate j
                # and pairs (m, cj) with candidate j... 
                # Actually we need: for each pair (a,b) in chosen, check if (a,b,candidate_j) is sunflower
                # When we add m, new pairs are (m, existing_i) for each existing_i
                # So for each existing_i, for each candidate_j, check sunflower(existing_i, m, candidate_j)
                # That's what the inner loop does above (ci is existing_i, m is new, cj is candidate_j)
    
    return [_mask_to_list(m) for m in chosen_masks]

def _greedy_simple(all_masks, order, deadline):
    """Simple greedy with O(n²) check per candidate - used as fallback."""
    chosen = []
    for idx in order:
        if time.time() >= deadline:
            break
        m = all_masks[idx]
        ok = True
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(chosen)):
                if _forms_sunflower_mask(chosen[i], chosen[j], m):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(m)
    return [_mask_to_list(m) for m in chosen]

def _local_search(family_masks, all_masks, n_total, deadline):
    """Try to add sets that don't conflict with the current family."""
    current = list(family_masks)
    current_set = set(current)
    
    # Precompute conflicts: for each candidate not in current, count conflicting pairs
    # A candidate m conflicts if there exist i,j in current with sunflower(current[i], current[j], m)
    
    def can_add(m, current_list):
        for i in range(len(current_list)):
            for j in range(i + 1, len(current_list)):
                if _forms_sunflower_mask(current_list[i], current_list[j], m):
                    return False
        return True
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n_total):
            if time.time() >= deadline:
                break
            m = all_masks[idx]
            if m in current_set:
                continue
            if can_add(m, current):
                current.append(m)
                current_set.add(m)
                improved = True
    
    return current

def _greedy_conflict(all_masks, order, deadline):
    """Greedy with incremental conflict tracking."""
    n_total = len(all_masks)
    conflict = [0] * n_total
    chosen = []
    
    for idx in order:
        if time.time() >= deadline:
            break
        if conflict[idx] > 0:
            continue
        
        m = all_masks[idx]
        chosen.append(m)
        
        # For each existing element in chosen, update conflicts
        for i in range(len(chosen) - 1):
            ci = chosen[i]
            for j in range(n_total):
                if j == idx or conflict[j] > 0:
                    continue
                cj = all_masks[j]
                if _forms_sunflower_mask(ci, cj, m):
                    conflict[j] += 1
    
    return [_mask_to_list(m) for m in chosen]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for row in combinations(range(1, 17), 4):
        m = 0
        for x in row:
            m |= (1 << (x - 1))
        all_masks.append(m)
    n_total = len(all_masks)
    
    best = []
    best_size = 0
    
    # Try different strategies and random orders
    strategies = [
        ("conflict", _greedy_conflict),
        ("simple", _greedy_simple),
    ]
    
    # First: deterministic order with conflict tracking
    result = _greedy_conflict(all_masks, list(range(n_total)), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
        # Local search
        if time.time() < deadline:
            best_masks = [_to_mask(s) for s in best]
            extended = _local_search(best_masks, all_masks, n_total, deadline)
            if len(extended) > best_size:
                best = [_mask_to_list(m) for m in extended]
                best_size = len(extended)
    
    # Randomized restarts
    rng = random.Random(42)
    for restart in range(500):
        if time.time() >= deadline:
            break
        order = list(range(n_total))
        rng.shuffle(order)
        
        # Alternate between strategies
        if restart % 2 == 0:
            result = _greedy_conflict(all_masks, order, deadline)
        else:
            result = _greedy_simple(all_masks, order, deadline)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
            # Local search on improved solution
            if time.time() < deadline:
                best_masks = [_to_mask(s) for s in best]
                extended = _local_search(best_masks, all_masks, n_total, deadline)
                if len(extended) > best_size:
                    best = [_mask_to_list(m) for m in extended]
                    best_size = len(extended)
    
    return [sorted(s) for s in best]
# EVOLVE-BLOCK-END
