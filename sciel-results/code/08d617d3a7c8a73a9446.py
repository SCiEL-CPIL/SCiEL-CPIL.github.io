import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute all 4-subsets of [1,16]
    all_sets = [set(s) for s in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Precompute frozenset representations
    all_frozensets = [frozenset(s) for s in all_sets]
    
    # Precompute all bitmasks
    all_bitmasks = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        all_bitmasks.append(mask)
    
    # Precompute all pairwise intersection masks
    # inter_masks[i][j] = bitmask of all_sets[i] & all_sets[j]
    inter_masks = [[0] * n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        for j in range(i + 1, n_sets):
            im = all_bitmasks[i] & all_bitmasks[j]
            inter_masks[i][j] = im
            inter_masks[j][i] = im
    
    def is_valid_addition(cand_idx, family_indices):
        """Check if cand_idx can be added to family without creating a sunflower."""
        cand_mask = all_bitmasks[cand_idx]
        # Group family members by their intersection with cand
        groups = {}
        for fi in family_indices:
            inter = inter_masks[cand_idx][fi]
            if inter not in groups:
                groups[inter] = [fi]
            else:
                groups[inter].append(fi)
        
        # For each group with intersection S, check if any pair has intersection S
        for S, members in groups.items():
            if len(members) < 2:
                continue
            n = len(members)
            for a in range(n):
                for b in range(a + 1, n):
                    if inter_masks[members[a]][members[b]] == S:
                        return False
        return True
    
    def greedy_construct(order, family_indices):
        """Add sets from order to family, skipping invalid ones."""
        for idx in order:
            if is_valid_addition(idx, family_indices):
                family_indices.append(idx)
    
    def local_search_add(family_indices, rng, max_iters=500):
        """Try to add more sets via random sampling."""
        added = 0
        family_set = set(family_indices)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            idx = rng.randint(0, n_sets - 1)
            if idx in family_set:
                continue
            if is_valid_addition(idx, family_indices):
                family_indices.append(idx)
                family_set.add(idx)
                added += 1
        return added
    
    def local_search_swap(family_indices, rng, max_iters=200):
        """Try swap operations: remove one set, add up to two others."""
        improved = 0
        family_set = set(family_indices)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            # Pick a random set to remove
            rem_pos = rng.randint(0, len(family_indices) - 1)
            removed_idx = family_indices[rem_pos]
            
            # Temporarily remove
            family_indices[rem_pos] = -1
            family_set.discard(removed_idx)
            
            # Try to add up to 2 new sets
            added_new = []
            for _ in range(3):
                cand_idx = rng.randint(0, n_sets - 1)
                if cand_idx == removed_idx or cand_idx in family_set:
                    continue
                if is_valid_addition(cand_idx, [x for x in family_indices if x >= 0]):
                    family_indices.append(cand_idx)
                    family_set.add(cand_idx)
                    added_new.append(cand_idx)
            
            # Check if we improved
            new_size = len([x for x in family_indices if x >= 0])
            old_size = len(family_indices) - 1  # before we added new ones, minus the removed
            if new_size > len(family_indices) - 1 - len(added_new) + len(added_new):
                # We gained sets - keep the new ones
                improved += 1
                # Clean up: remove the placeholder
                family_indices = [x for x in family_indices if x >= 0]
            else:
                # Revert: remove the newly added, restore the removed
                for ci in added_new:
                    family_indices.remove(ci)
                    family_set.discard(ci)
                family_indices[rem_pos] = removed_idx
                family_set.add(removed_idx)
        
        return improved
    
    rng = random.Random(42)
    best_family_indices = []
    best_size = 0
    
    # Phase 1: Multiple greedy constructions with different orderings
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        if trial == 0:
            order = list(range(n_sets))
        elif trial == 1:
            order = list(range(n_sets))
            order.reverse()
        else:
            order = list(range(n_sets))
            rng.shuffle(order)
        
        family_indices = []
        greedy_construct(order, family_indices)
        
        # Local search to add more
        local_search_add(family_indices, rng, max_iters=400)
        
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family_indices = list(family_indices)
    
    # Phase 2: Local search swaps on the best family
    if time.time() < deadline - 5 and best_family_indices:
        family_indices = list(best_family_indices)
        
        for round_num in range(15):
            if time.time() > deadline - 5:
                break
            local_search_swap(family_indices, rng, max_iters=150)
            local_search_add(family_indices, rng, max_iters=300)
            
            if len(family_indices) > best_size:
                best_size = len(family_indices)
                best_family_indices = list(family_indices)
    
    # Phase 3: More aggressive local search if time permits
    if time.time() < deadline - 3 and best_family_indices:
        family_indices = list(best_family_indices)
        family_set = set(best_family_indices)
        
        for _ in range(1000):
            if time.time() > deadline - 3:
                break
            idx = rng.randint(0, n_sets - 1)
            if idx in family_set:
                continue
            if is_valid_addition(idx, family_indices):
                family_indices.append(idx)
                family_set.add(idx)
                if len(family_indices) > best_size:
                    best_size = len(family_indices)
                    best_family_indices = list(family_indices)
    
    return [sorted(all_sets[i]) for i in best_family_indices]
