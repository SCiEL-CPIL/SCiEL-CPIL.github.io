import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    n_total = len(all_masks)  # 1820

    # Precompute pairwise intersections
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            ij = mi & all_masks[j]
            inter[i][j] = ij
            inter[j][i] = ij

    def can_add(family, m_idx, family_set):
        """Check if mask m_idx can be added to family without forming a sunflower."""
        m = all_masks[m_idx]
        # Group family members by their intersection with m
        groups = {}
        for a_idx in family_set:
            a = all_masks[a_idx]
            inter_a_m = a & m
            if inter_a_m in groups:
                # Check all existing b in this group
                for b_idx in groups[inter_a_m]:
                    if inter[a_idx][b_idx] == inter_a_m:
                        return False
                groups[inter_a_m].append(a_idx)
            else:
                groups[inter_a_m] = [a_idx]
        return True

    def can_add_fast(family_list, m_idx, family_set):
        """Same as can_add but takes a list for potential index-based access."""
        m = all_masks[m_idx]
        groups = {}
        for a_idx in family_set:
            a = all_masks[a_idx]
            inter_a_m = a & m
            if inter_a_m in groups:
                for b_idx in groups[inter_a_m]:
                    if inter[a_idx][b_idx] == inter_a_m:
                        return False
                groups[inter_a_m].append(a_idx)
            else:
                groups[inter_a_m] = [a_idx]
        return True

    rng = random.Random(12345)
    best_family = []
    best_size = 0

    # Multiple random greedy passes
    for iteration in range(500):
        if time.time() > deadline - 5:
            break

        shuffled = list(range(n_total))
        rng.shuffle(shuffled)

        family = []
        family_set = set()
        for idx in shuffled:
            if can_add(family, idx, family_set):
                family.append(idx)
                family_set.add(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

        if best_size >= 250:
            break

    # Local search: try to add more sets to the best family
    if time.time() < deadline - 3 and best_family:
        family = list(best_family)
        family_set = set(family)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(n_total):
                if time.time() > deadline - 2:
                    break
                if idx in family_set:
                    continue
                if can_add(family, idx, family_set):
                    family.append(idx)
                    family_set.add(idx)
                    improved = True

        if len(family) > best_size:
            best_family = family
            best_size = len(family)

    # NEW: Swap-based local search to escape local optima
    # Try removing one set and adding a different one to potentially unlock more additions
    if time.time() < deadline - 1 and best_family:
        family = list(best_family)
        family_set = set(family)
        
        # Try multiple rounds of swaps
        for swap_round in range(3):
            if time.time() > deadline - 1:
                break
            
            improved_swap = True
            while improved_swap and time.time() < deadline - 1:
                improved_swap = False
                for i in range(len(family)):
                    if time.time() > deadline - 1:
                        break
                    
                    removed_idx = family[i]
                    # Create a temporary family without this element
                    temp_set = set(family_set) - {removed_idx}
                    
                    # Try to find a candidate that can be added to the reduced family
                    # and is not already in the family
                    found_candidate = False
                    for cand_idx in range(n_total):
                        if time.time() > deadline - 1:
                            break
                        if cand_idx in temp_set:
                            continue
                        
                        # Check if this candidate can be added to the reduced family
                        # We need to check against all pairs in the reduced family
                        m = all_masks[cand_idx]
                        groups = {}
                        valid = True
                        for a_idx in temp_set:
                            a = all_masks[a_idx]
                            inter_a_m = a & m
                            if inter_a_m in groups:
                                for b_idx in groups[inter_a_m]:
                                    if inter[a_idx][b_idx] == inter_a_m:
                                        valid = False
                                        break
                                if not valid:
                                    break
                                groups[inter_a_m].append(a_idx)
                            else:
                                groups[inter_a_m] = [a_idx]
                        
                        if valid:
                            # Perform the swap
                            family[i] = cand_idx
                            family_set.remove(removed_idx)
                            family_set.add(cand_idx)
                            improved_swap = True
                            found_candidate = True
                            break
                    
                    if found_candidate:
                        # After a swap, try to add more sets
                        improved_add = True
                        while improved_add and time.time() < deadline - 1:
                            improved_add = False
                            for idx in range(n_total):
                                if time.time() > deadline - 1:
                                    break
                                if idx in family_set:
                                    continue
                                if can_add(family, idx, family_set):
                                    family.append(idx)
                                    family_set.add(idx)
                                    improved_add = True
                        
                        if len(family) > best_size:
                            best_size = len(family)
                            best_family = list(family)
                        
                        # If we found a good swap, break out of the inner loop to restart
                        break

    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
