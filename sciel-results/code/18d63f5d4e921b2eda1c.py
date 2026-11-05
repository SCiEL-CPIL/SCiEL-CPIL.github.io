"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
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

    m = len(all_masks)
    # m = C(16,4) = 1820

    # Precompute pairwise intersections
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            inter[i][j] = mi & all_masks[j]
            inter[j][i] = inter[i][j]

    # Precompute for each pair (i,j) the set of "bad" indices k such that
    # inter[i][k] == inter[j][k] == inter[i][j]
    # i.e., adding k would create a sunflower with i and j.
    # We'll store this as a bitmask or set for fast lookup.
    # Since m=1820, we can use a list of sets.
    # bad_with[i][j] = set of k (k != i, k != j) such that inter[i][k] == inter[j][k] == inter[i][j]
    # Actually, we need: for a candidate idx, check if there exist i,j in family with
    # inter[i][j] == inter[i][idx] == inter[j][idx]
    # Equivalent: idx is in bad_with[i][j] for some i,j in family.
    # But building bad_with for all 1820*1820 pairs is too expensive.
    # Instead, let's use a different approach:
    # For each idx, precompute the "signature" with respect to the family.
    # Actually, let's just optimize the check function.

    # Alternative: For each pair (i,j), precompute the intersection value v = inter[i][j].
    # Then for a candidate idx, we need inter[i][idx] == v AND inter[j][idx] == v.
    # Let's precompute for each idx and each possible intersection value,
    # which family members have that intersection with idx.
    # This is getting complex. Let's stick with the O(nf^2) check but optimize it.

    # Better approach: For each idx, precompute a dictionary:
    # inter_map[idx] = dict mapping intersection_value -> list of indices j with inter[idx][j] == that value
    # But that's also heavy.

    # Let's try a different optimization:
    # For a candidate idx, group family members by inter[family_member][idx].
    # For each group with value v, check if any pair in the group has inter == v.
    # This is O(nf + sum over groups of group_size^2) which could be worse.

    # Let's just keep the O(nf^2) check but make it faster with local variables.

    def check_violation_idx(family_indices, idx):
        nf = len(family_indices)
        fidx = family_indices
        inter_local = inter
        for ii in range(nf):
            a = fidx[ii]
            an = inter_local[a][idx]
            for jj in range(ii + 1, nf):
                b = fidx[jj]
                ab = inter_local[a][b]
                bn = inter_local[b][idx]
                if ab == an == bn:
                    return True
        return False

    def mask_to_list(mask):
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

    # Strategy 2: Multiple random shuffles with more attempts
    random.seed(12345)
    
    for attempt in range(200):
        if time.time() > deadline - 15:
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

    # Strategy 3: Local improvement - try to add sets that were skipped
    in_family = set(best_family_idx)
    candidates = [idx for idx in range(m) if idx not in in_family]
    random.shuffle(candidates)

    improved = best_family_idx[:]
    for idx in candidates:
        if time.time() > deadline - 5:
            break
        if not check_violation_idx(improved, idx):
            improved.append(idx)

    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Strategy 4: More random restarts with different seeds
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579,
                     101, 202, 303, 404, 505, 606, 707, 808, 909, 1000,
                     1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000]:
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

        # Quick local improvement on this family
        in_f = set(family)
        cands = [idx for idx in range(m) if idx not in in_f]
        random.shuffle(cands)
        for idx in cands:
            if time.time() > deadline - 8:
                break
            if not check_violation_idx(family, idx):
                family.append(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Final local improvement on the best
    in_family = set(best_family_idx)
    candidates = [idx for idx in range(m) if idx not in in_family]
    random.shuffle(candidates)

    improved = best_family_idx[:]
    for idx in candidates:
        if time.time() > deadline - 2:
            break
        if not check_violation_idx(improved, idx):
            improved.append(idx)

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
