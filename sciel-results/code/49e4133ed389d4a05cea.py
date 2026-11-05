"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

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

    # Precompute pairwise intersections
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            inter[i][j] = mi & all_masks[j]
            inter[j][i] = inter[i][j]

    def check_violation_idx(family_indices, idx):
        """
        Check if adding index idx to family creates a sunflower violation.
        """
        nf = len(family_indices)
        fidx = family_indices
        for ii in range(nf):
            a = fidx[ii]
            an = inter[a][idx]
            for jj in range(ii + 1, nf):
                b = fidx[jj]
                ab = inter[a][b]
                bn = inter[b][idx]
                if ab == an == bn:
                    return True
        return False

    def check_violation_after_removal(family_indices, remove_idx, add_idx):
        """
        Check if replacing family_indices[remove_idx] with add_idx creates a violation.
        We need to check all pairs in the new family.
        """
        nf = len(family_indices)
        # Build the new family list without remove_idx
        new_family = []
        for i in range(nf):
            if i != remove_idx:
                new_family.append(family_indices[i])
        new_family.append(add_idx)
        
        # Check all pairs in new_family
        nnf = len(new_family)
        for ii in range(nnf):
            a = new_family[ii]
            for jj in range(ii + 1, nnf):
                b = new_family[jj]
                ab = inter[a][b]
                # We need a third set c such that inter[a][c] == inter[b][c] == ab
                # But we only have the new_family. So we check all triples.
                # Actually, the violation is: there exist three distinct sets a,b,c in new_family
                # such that inter[a][b] == inter[a][c] == inter[b][c]
                # This is O(n^3) which is too slow. Let's use a different approach.
                pass
        
        # Better: for each pair (a,b) in new_family, check if there's a third set c
        # such that inter[a][b] == inter[a][c] == inter[b][c]
        # But we can precompute for each pair (a,b) the intersection value,
        # and then for each c, check if inter[a][c] == inter[b][c] == inter[a][b]
        
        # Actually, let's just do the O(n^2 * n) check but it's the same as before.
        # The key insight: after removal, the family size is nf (same as before).
        # We need to check if any triple in new_family is a sunflower.
        
        # Efficient approach: for each pair (a,b) in new_family, compute v = inter[a][b].
        # Then for each other c in new_family, check if inter[a][c] == v and inter[b][c] == v.
        # This is O(n^2 * n) = O(n^3) in the worst case, but n is small (~100-200).
        
        # Let's just use the same check_violation_idx logic but on the modified family.
        # We'll create a temporary list and check.
        
        # Actually, the simplest correct approach:
        # For each pair (i,j) in new_family, let v = inter[new_family[i]][new_family[j]]
        # For each k != i,j in new_family, check if inter[new_family[i]][new_family[k]] == v 
        # and inter[new_family[j]][new_family[k]] == v
        # If so, violation.
        
        for ii in range(nnf):
            a = new_family[ii]
            for jj in range(ii + 1, nnf):
                b = new_family[jj]
                v = inter[a][b]
                for kk in range(nnf):
                    if kk == ii or kk == jj:
                        continue
                    c = new_family[kk]
                    if inter[a][c] == v and inter[b][c] == v:
                        return True
        return False

    def mask_to_list(mask):
        """Convert bitmask to sorted list of 1..16 elements."""
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result

    best_size = 0
    best_family_idx = []

    # Strategy 1: Greedy in natural order
    family = []
    for idx in range(m):
        if time.time() > deadline:
            break
        if not check_violation_idx(family, idx):
            family.append(idx)

    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles
    random.seed(12345)
    max_attempts = 30

    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break

        order = list(range(m))
        random.shuffle(order)

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_idx(family, idx):
                family.append(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Strategy 3: Local improvement with swaps
    def local_improve(family_indices, time_budget):
        """Try to improve family by additions and swaps."""
        improved = family_indices[:]
        in_family = set(improved)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)
        
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            
            # Try additions
            for idx in candidates:
                if time.time() > time_budget:
                    break
                if not check_violation_idx(improved, idx):
                    improved.append(idx)
                    in_family.add(idx)
                    candidates.remove(idx)
                    changed = True
            
            # Try swaps: remove one, add one
            if time.time() > time_budget - 1:
                break
            for i in range(len(improved)):
                if time.time() > time_budget:
                    break
                removed = improved[i]
                for idx in candidates:
                    if time.time() > time_budget:
                        break
                    # Check if removing improved[i] and adding idx is valid
                    if not check_violation_after_removal(improved, i, idx):
                        improved[i] = idx
                        in_family.remove(removed)
                        in_family.add(idx)
                        candidates.remove(idx)
                        changed = True
                        break  # Restart the swap loop after a successful swap
        
        return improved

    # Apply local improvement to best so far
    time_budget = deadline - 5
    improved = local_improve(best_family_idx, time_budget)
    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Strategy 4: More random restarts with different seeds
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579]:
        if time.time() > deadline - 8:
            break
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_idx(family, idx):
                family.append(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Final local improvement on the best
    if time.time() < deadline - 2:
        improved = local_improve(best_family_idx, deadline - 1)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    # Convert back to sorted lists of 1..16
    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]

    return result
# EVOLVE-BLOCK-END
