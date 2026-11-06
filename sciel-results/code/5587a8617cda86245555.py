# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 16
    k = 4
    start_time = time.time()
    time_limit = 85.0  # Safety margin

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
    
    num_masks = len(all_masks)  # 1820

    # Precompute pairwise intersections
    # inter[i][j] is the bitmask of intersection of all_masks[i] and all_masks[j]
    inter = [[0] * num_masks for _ in range(num_masks)]
    for i in range(num_masks):
        mi = all_masks[i]
        for j in range(i + 1, num_masks):
            v = mi & all_masks[j]
            inter[i][j] = v
            inter[j][i] = v

    def check_conflict_fast(candidate_idx, family_indices):
        """
        Check if adding candidate_idx to family_indices creates a sunflower.
        Uses grouping by intersection with candidate to reduce O(M^2) to O(M + sum |G|^2).
        """
        if not family_indices:
            return False
        
        cm = all_masks[candidate_idx]
        
        # Group family members by their intersection with candidate
        groups = {}
        for f_idx in family_indices:
            inter_val = cm & all_masks[f_idx]
            if inter_val not in groups:
                groups[inter_val] = []
            groups[inter_val].append(f_idx)
        
        # Check for conflicts within groups
        for inter_val, group_members in groups.items():
            if len(group_members) < 2:
                continue
            # For any pair in this group, check if their mutual intersection is also inter_val
            # If yes, they form a sunflower with candidate.
            m = len(group_members)
            for a in range(m):
                i = group_members[a]
                for b in range(a + 1, m):
                    j = group_members[b]
                    if inter[i][j] == inter_val:
                        return True
        return False

    def greedy(order):
        """Run greedy in given order, return list of indices in family."""
        family = []
        for idx in order:
            if not check_conflict_fast(idx, family):
                family.append(idx)
        return family

    # 1. Deterministic greedy (lexicographic order)
    best_family = greedy(list(range(num_masks)))
    best_size = len(best_family)

    if time.time() - start_time > time_limit - 5:
        return [[i + 1 for i in range(n) if all_masks[best_family[j]] & (1 << i)] 
                for j in range(len(best_family))]

    # 2. Randomized greedy restarts
    random.seed(42)
    for restart in range(20):
        if time.time() - start_time > time_limit - 20:
            break
        order = list(range(num_masks))
        random.shuffle(order)
        family = greedy(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family

    # 3. Local Search (1-swap)
    # We try to add any non-family member. If it conflicts, we try to swap it with a family member.
    # We repeat until no improvement is found or time runs out.
    
    # Convert best_family to a set for fast lookup
    family_set = set(best_family)
    family_list = list(best_family)
    
    improved = True
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        # Shuffle non-family members to avoid bias
        candidates = [c for c in range(num_masks) if c not in family_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() - start_time > time_limit - 5:
                break
            
            # First, try to just add c
            if not check_conflict_fast(c, family_list):
                family_list.append(c)
                family_set.add(c)
                best_size += 1
                improved = True
                # Don't break, try to add more
                continue
            
            # If it conflicts, try swapping c with some member f in family_list
            # To speed up, we can try a few random swaps or all swaps.
            # Given the small size, trying all swaps is feasible if check is fast.
            # But check_conflict_fast is O(M). So M swaps is O(M^2).
            # M is ~100. M^2 = 10k. 1800 candidates * 10k = 18M ops.
            # This might be too slow for a full pass if we do many passes.
            # Let's try a limited number of swaps per candidate, or just one random swap.
            
            # Strategy: Try to find a valid swap.
            # Randomize family_list order for swaps to avoid bias
            swap_candidates = family_list[:]
            random.shuffle(swap_candidates)
            
            swapped = False
            for f in swap_candidates:
                # Remove f, check if c can be added to remaining
                # We need to check c against family_list \ {f}
                # This is slightly expensive to construct the list each time.
                # Optimization: Pass a list that skips f?
                # Or just construct the list. M is small.
                
                # Construct new family list without f
                # Instead of list comprehension, we can use a helper or just do it.
                # For speed, let's just check if c conflicts with family_list, 
                # but we know it does. The conflict is with some pair.
                # Removing f might resolve it.
                
                # Efficient check:
                # We can't easily reuse the group logic without the full list.
                # Let's just build the list.
                new_family_list = [x for x in family_list if x != f]
                
                if not check_conflict_fast(c, new_family_list):
                    # Valid swap
                    family_list = new_family_list
                    family_list.append(c)
                    family_set.remove(f)
                    family_set.add(c)
                    # Size remains same, but structure changed. 
                    # This might allow future additions.
                    # We mark improved to continue search.
                    improved = True
                    swapped = True
                    break
            
            # If we swapped, we break out of candidate loop to re-evaluate?
            # Or continue? Continuing might be better to find more swaps/additions.
            # But the state has changed. The `candidates` list is stale (contains c? No, c was not in family).
            # But family_set changed.
            # Let's just break and restart the outer loop to refresh candidates and try additions again.
            if swapped:
                break

    # Final polish: try to add any remaining candidates
    # This is essentially one more greedy pass on the current family
    family_set = set(family_list)
    for c in range(num_masks):
        if c not in family_set:
            if not check_conflict_fast(c, family_list):
                family_list.append(c)
                family_set.add(c)

    best_family = family_list
    best_size = len(best_family)

    # Convert to result
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = [i + 1 for i in range(n) if mask & (1 << i)]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
