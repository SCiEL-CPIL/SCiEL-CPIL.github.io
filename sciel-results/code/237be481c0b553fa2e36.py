# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 16
    k = 4
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
    
    num_masks = len(all_masks)  # 1820

    # Precompute pairwise intersections: inter[i][j] = all_masks[i] & all_masks[j]
    # This is 1820*1820/2 ~ 1.6M entries. Use a list of lists.
    # Actually, let's use a 2D list but only store upper triangle or use dict.
    # Better: precompute inter[i][j] for all i,j as a list of lists.
    # 1820^2 = 3.3M integers, each 4-8 bytes. ~16-26 MB. Acceptable.
    inter = [[0] * num_masks for _ in range(num_masks)]
    for i in range(num_masks):
        mi = all_masks[i]
        for j in range(i + 1, num_masks):
            v = mi & all_masks[j]
            inter[i][j] = v
            inter[j][i] = v

    def check_conflict(candidate_idx, family_indices):
        """Check if adding candidate_idx to family_indices creates a sunflower."""
        cm = all_masks[candidate_idx]
        m = len(family_indices)
        for ai in range(m):
            a_idx = family_indices[ai]
            a_inter_c = inter[a_idx][candidate_idx]
            for bi in range(ai + 1, m):
                b_idx = family_indices[bi]
                # Sunflower: a&b == a&c == b&c
                if inter[a_idx][b_idx] == a_inter_c and inter[b_idx][candidate_idx] == a_inter_c:
                    return True
        return False

    def greedy(order):
        """Run greedy in given order, return list of indices in family."""
        family = []
        for idx in order:
            if not check_conflict(idx, family):
                family.append(idx)
        return family

    # 1. Deterministic greedy (lexicographic order)
    best_family = greedy(list(range(num_masks)))
    best_size = len(best_family)

    if time.time() - start_time > time_limit:
        return [[all_masks[i] and (i+1) for i in range(n) if all_masks[best_family[j]] & (1 << i)] 
                for j in range(len(best_family))]

    # 2. Randomized greedy restarts
    random.seed(42)
    for restart in range(50):
        if time.time() - start_time > time_limit - 10:
            break
        order = list(range(num_masks))
        random.shuffle(order)
        family = greedy(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family

    # 3. Local search: try to improve by swapping
    # Build set of indices in best_family for O(1) lookup
    in_family = set(best_family)
    
    # For each set in family, try removing it and adding a better candidate
    # Or: for each candidate not in family, try to find a swap that increases size
    
    # Strategy: Try to add a candidate by removing one family member
    # For each candidate c not in family:
    #   For each family member f:
    #     Check if (family - {f}) + {c} is valid
    #     If yes and it allows further additions, do it
    
    # Simpler: Hill climbing with swaps
    improved = True
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        family_set = set(best_family)
        # Try each candidate not in family
        for c in range(num_masks):
            if c in family_set:
                continue
            # Check if c can be added by removing one family member
            for f in list(best_family):
                # Try removing f and adding c
                new_family = [x for x in best_family if x != f]
                # Check if c is valid with new_family
                if not check_conflict(c, new_family):
                    new_family.append(c)
                    # Now try to greedily add more from remaining candidates
                    # Quick greedy pass on remaining
                    added = 0
                    for c2 in range(num_masks):
                        if c2 in set(new_family):
                            continue
                        if not check_conflict(c2, new_family):
                            new_family.append(c2)
                            added += 1
                    if len(new_family) > best_size:
                        best_family = new_family
                        best_size = len(new_family)
                        family_set = set(best_family)
                        improved = True
                        break
            if improved:
                break

    # Convert to result
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = [i + 1 for i in range(n) if mask & (1 << i)]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
