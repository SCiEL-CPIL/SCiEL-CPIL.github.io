"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 16
    k = 4

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)

    m = len(all_masks)  # C(16,4) = 1820

    # Precompute pairwise intersections as numpy array
    inter_np = np.zeros((m, m), dtype=np.int32)
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            v = mi & all_masks[j]
            inter_np[i, j] = v
            inter_np[j, i] = v

    def mask_to_list(mask):
        """Convert bitmask to sorted list of 1..16 elements."""
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result

    def check_violation_fast(family_arr, new_idx):
        """
        Check if adding new_idx to family creates a sunflower violation.
        family_arr is a numpy array of indices in the family.
        """
        nf = len(family_arr)
        if nf < 2:
            return False
        
        # Get intersections of new_idx with all family members
        new_inter = inter_np[new_idx, family_arr]  # shape (nf,)
        
        # Get pairwise intersections among family members
        fam_inter = inter_np[np.ix_(family_arr, family_arr)]  # (nf, nf)
        
        # Check: for any pair (i,j) in family, inter[f[i]][f[j]] == new_inter[i] == new_inter[j]
        ni_i = new_inter[:, np.newaxis]  # (nf, 1)
        ni_j = new_inter[np.newaxis, :]  # (1, nf)
        
        # Upper triangle mask
        mask_upper = np.triu(np.ones((nf, nf), dtype=bool), k=1)
        
        violation = (ni_i == ni_j) & (fam_inter == ni_i) & mask_upper
        return bool(np.any(violation))

    def full_validate(family_indices):
        """Full validation of a family - O(n^3) but n is small."""
        nf = len(family_indices)
        if nf < 3:
            return True
        farr = np.array(family_indices, dtype=np.int32)
        fam_inter = inter_np[np.ix_(farr, farr)]
        for a in range(nf):
            for b in range(a + 1, nf):
                v = fam_inter[a, b]
                for c in range(b + 1, nf):
                    if fam_inter[a, c] == v and fam_inter[b, c] == v:
                        return False
        return True

    best_size = 0
    best_family_idx = []

    # Strategy 1: Greedy in natural order
    family = []
    family_arr = np.array(family, dtype=np.int32)
    for idx in range(m):
        if time.time() > deadline - 5:
            break
        if not check_violation_fast(family_arr, idx):
            family.append(idx)
            family_arr = np.array(family, dtype=np.int32)

    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles with different seeds
    seeds = [42, 12345, 7, 99, 2024, 555, 777, 13579, 2468, 1357, 9182, 7530, 8888, 4321, 1111, 2222, 3333, 4444, 5555, 6666]
    
    for seed_val in seeds:
        if time.time() > deadline - 10:
            break
        
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)

        family = []
        family_arr = np.array(family, dtype=np.int32)
        for idx in order:
            if time.time() > deadline - 5:
                break
            if not check_violation_fast(family_arr, idx):
                family.append(idx)
                family_arr = np.array(family, dtype=np.int32)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Strategy 3: Local improvement - try to add sets that were skipped
    def local_add(family_indices, time_budget):
        """Try to add more sets to the family."""
        improved = family_indices[:]
        in_family = set(improved)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)
        
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            improved_arr = np.array(improved, dtype=np.int32)
            for idx in candidates[:]:
                if time.time() > time_budget:
                    break
                if not check_violation_fast(improved_arr, idx):
                    improved.append(idx)
                    in_family.add(idx)
                    candidates.remove(idx)
                    changed = True
                    improved_arr = np.array(improved, dtype=np.int32)
        
        return improved

    # Apply local addition to best so far
    if time.time() < deadline - 8:
        improved = local_add(best_family_idx, deadline - 8)
        if full_validate(improved) and len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Strategy 4: Local improvement with swaps (remove one, add one)
    def local_swap(family_indices, time_budget):
        """Try to improve family by swaps."""
        improved = family_indices[:]
        in_family = set(improved)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)
        
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            nf = len(improved)
            
            for i in range(nf):
                if time.time() > time_budget:
                    break
                removed = improved[i]
                
                # Temporarily remove
                temp_family = improved[:i] + improved[i+1:]
                temp_arr = np.array(temp_family, dtype=np.int32)
                
                # Try to find a replacement
                for idx in candidates[:]:
                    if time.time() > time_budget:
                        break
                    if not check_violation_fast(temp_arr, idx):
                        # Valid swap
                        improved[i] = idx
                        in_family.remove(removed)
                        in_family.add(idx)
                        candidates.remove(idx)
                        changed = True
                        break
                else:
                    # No valid replacement found, continue
                    continue
                
                if changed:
                    break  # Restart after a successful swap
        
        return improved

    # Apply local swap improvement
    if time.time() < deadline - 5:
        improved = local_swap(best_family_idx, deadline - 5)
        if full_validate(improved) and len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Strategy 5: More aggressive random restarts with local improvement
    for seed_val in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        if time.time() > deadline - 15:
            break
        
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)

        family = []
        family_arr = np.array(family, dtype=np.int32)
        for idx in order:
            if time.time() > deadline - 10:
                break
            if not check_violation_fast(family_arr, idx):
                family.append(idx)
                family_arr = np.array(family, dtype=np.int32)

        # Quick local improvement
        if len(family) > best_size - 5 and time.time() < deadline - 10:
            improved = local_add(family, deadline - 10)
            if full_validate(improved) and len(improved) > best_size:
                best_size = len(improved)
                best_family_idx = improved[:]

    # Final local improvement on the best
    if time.time() < deadline - 3:
        improved = local_add(best_family_idx, deadline - 3)
        if full_validate(improved) and len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    if time.time() < deadline - 2:
        improved = local_swap(best_family_idx, deadline - 2)
        if full_validate(improved) and len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    # Convert back to sorted lists of 1..16
    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]

    return result
# EVOLVE-BLOCK-END
