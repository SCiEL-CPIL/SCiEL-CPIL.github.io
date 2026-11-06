"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute all 4-element subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)

    # Encode as bitmasks for fast intersection
    def encode_set(s):
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        return mask

    all_masks = [encode_set(s) for s in all_sets]

    def is_valid_family(family_indices):
        """Check if a family is sunflower-free."""
        n = len(family_indices)
        masks = [all_masks[i] for i in family_indices]
        for i in range(n):
            for j in range(i + 1, n):
                ab = masks[i] & masks[j]
                for k in range(j + 1, n):
                    ac = masks[i] & masks[k]
                    bc = masks[j] & masks[k]
                    if ab == ac == bc:
                        return False
        return True

    def can_add_mask(s_mask, chosen_masks):
        """Check if s_mask can be added to chosen_masks without creating a sunflower."""
        m = len(chosen_masks)
        if m < 2:
            return True
        
        # Group chosen sets by their intersection with s_mask
        groups = {}
        for i in range(m):
            inter = chosen_masks[i] & s_mask
            if inter not in groups:
                groups[inter] = [i]
            else:
                groups[inter].append(i)
        
        # For each group, check if any pair has intersection equal to the group key
        for inter, indices in groups.items():
            if len(indices) >= 2:
                n_idx = len(indices)
                for j in range(n_idx):
                    a_mask = chosen_masks[indices[j]]
                    for k in range(j + 1, n_idx):
                        b_mask = chosen_masks[indices[k]]
                        if (a_mask & b_mask) == inter:
                            return False
        return True

    def greedy_build(order):
        """Build a family using the given order of indices."""
        chosen = []
        chosen_masks = []
        for idx in order:
            s_mask = all_masks[idx]
            if can_add_mask(s_mask, chosen_masks):
                chosen.append(idx)
                chosen_masks.append(s_mask)
        return chosen

    def local_search(family, time_limit):
        """Improve the family using local search swaps and additions."""
        family = list(family)
        family_set = set(family)
        family_masks = [all_masks[i] for i in family]
        
        def try_add(idx):
            if idx in family_set:
                return False
            s_mask = all_masks[idx]
            if can_add_mask(s_mask, family_masks):
                family.append(idx)
                family_set.add(idx)
                family_masks.append(s_mask)
                return True
            return False

        def try_remove(idx):
            if idx not in family_set:
                return False
            pos = family.index(idx)
            family.pop(pos)
            family_set.discard(idx)
            family_masks.pop(pos)
            return True

        # Phase 1: Try to add any set that can be added
        added_any = True
        while added_any and time.time() < time_limit:
            added_any = False
            for idx in range(n_total):
                if time.time() > time_limit:
                    break
                if try_add(idx):
                    added_any = True
                    break
        
        # Phase 2: Try swaps: remove one set, try to add multiple others
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for remove_idx in list(family):
                if time.time() > time_limit:
                    break
                original_size = len(family)
                original_family = list(family)
                original_set = set(family_set)
                original_masks = list(family_masks)
                
                try_remove(remove_idx)
                
                # Try to add as many as possible
                for idx in range(n_total):
                    if time.time() > time_limit:
                        break
                    try_add(idx)
                
                new_size = len(family)
                if new_size > original_size:
                    improved = True
                else:
                    # Revert
                    family[:] = original_family
                    family_set.clear()
                    family_set.update(original_set)
                    family_masks[:] = original_masks

        return family

    best_family = []
    best_size = 0

    # 1. Deterministic greedy (lexicographic order) - this is the baseline that scored 1.0
    order_det = list(range(n_total))
    family_det = greedy_build(order_det)
    best_family = family_det
    best_size = len(family_det)

    # 2. Try randomized greedy restarts with local search
    max_trials = 30
    for trial in range(max_trials):
        if time.time() > deadline - 15.0:  # Leave 15s for final local search
            break
        
        random.seed(trial * 1000003 + 7)
        order_rand = list(range(n_total))
        random.shuffle(order_rand)
        
        family_rand = greedy_build(order_rand)
        
        # Apply local search to each candidate if time permits
        if time.time() < deadline - 5.0 and len(family_rand) >= best_size - 5:
            ls_limit = min(deadline - 5.0, time.time() + 2.0)
            family_rand = local_search(family_rand, ls_limit)
        
        if len(family_rand) > best_size:
            best_size = len(family_rand)
            best_family = family_rand

    # 3. Final local search on the best family found so far
    if time.time() < deadline - 5.0:
        ls_limit = deadline - 5.0
        family_ls = local_search(best_family, ls_limit)
        if len(family_ls) > best_size:
            best_size = len(family_ls)
            best_family = family_ls

    # 4. Final verification and conversion
    if not is_valid_family(best_family):
        # Fallback to deterministic greedy if something went wrong
        best_family = family_det

    result = [sorted(all_sets[i]) for i in best_family]
    return result
# EVOLVE-BLOCK-END
