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
        A sunflower of size 3 exists if there are three distinct sets a, b, c
        in the family such that inter[a][b] == inter[a][c] == inter[b][c].
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

    # Strategy 2: Multiple random shuffles with local improvement
    random.seed(42)
    max_attempts = 30

    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break

        order = list(range(m))
        random.shuffle(order)

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            mask = all_masks[idx]
            if not check_violation_idx(family, idx):
                family.append(idx)

        # Local improvement: try to add sets that were skipped
        in_family = set(family)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)

        improved = family[:]
        for idx in candidates:
            if time.time() > deadline - 2:
                break
            if not check_violation_idx(improved, idx):
                improved.append(idx)

        if len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Strategy 3: Local improvement on the best family found so far
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

    # Strategy 4: More random restarts with different seeds
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579]:
        if time.time() > deadline - 5:
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

        # Local improvement
        in_family = set(family)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)

        improved = family[:]
        for idx in candidates:
            if time.time() > deadline - 2:
                break
            if not check_violation_idx(improved, idx):
                improved.append(idx)

        if len(improved) > best_size:
            best_size = len(improved)
            best_family_idx = improved[:]

    # Final local improvement on the best
    in_family = set(best_family_idx)
    candidates = [idx for idx in range(m) if idx not in in_family]
    random.shuffle(candidates)

    improved = best_family_idx[:]
    for idx in candidates:
        if time.time() > deadline - 1:
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
