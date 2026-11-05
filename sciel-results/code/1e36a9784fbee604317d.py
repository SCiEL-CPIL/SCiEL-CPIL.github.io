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

    m = len(all_masks)
    # m = C(16,4) = 1820

    # Precompute pairwise intersections
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            v = mi & all_masks[j]
            inter[i][j] = v
            inter[j][i] = v

    def check_violation_fast(family_indices, idx):
        """
        Check if adding index idx to family creates a sunflower violation.
        Uses bitmask logic:
        Sunflower exists if for some A, B in family:
        (A & B) subset C AND C disjoint from (A \ B) AND C disjoint from (B \ A)
        """
        nf = len(family_indices)
        fidx = family_indices
        mc = all_masks[idx]
        
        for ii in range(nf):
            a = fidx[ii]
            ma = all_masks[a]
            for jj in range(ii + 1, nf):
                b = fidx[jj]
                mb = all_masks[b]
                ab = inter[a][b]
                
                # Condition 1: A & B must be subset of C
                if (ab & ~mc) != 0:
                    continue
                
                # Condition 2: C must be disjoint from A \ B
                # A \ B = ma & ~mb
                if (mc & (ma & ~mb)) != 0:
                    continue
                
                # Condition 3: C must be disjoint from B \ A
                # B \ A = mb & ~ma
                if (mc & (mb & ~ma)) != 0:
                    continue
                
                # If all conditions met, A, B, C form a sunflower
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
        if not check_violation_fast(family, idx):
            family.append(idx)

    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles
    random.seed(12345)
    max_attempts = 200 # Increased attempts due to faster check

    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break

        order = list(range(m))
        random.shuffle(order)

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_fast(family, idx):
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
        if time.time() > deadline - 3:
            break
        if not check_violation_fast(improved, idx):
            improved.append(idx)

    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Strategy 4: More random restarts with different seeds
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579, 2468, 1357, 9999, 1111, 2222, 3333, 4444, 5555, 6666, 7777]:
        if time.time() > deadline - 5:
            break
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)

        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not check_violation_fast(family, idx):
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
        if time.time() > deadline - 1:
            break
        if not check_violation_fast(improved, idx):
            improved.append(idx)

    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Strategy 5: Swap-based local search - try replacing one element with another
    if time.time() < deadline - 2:
        current_family = best_family_idx[:]
        in_current = set(current_family)
        not_in_current = [idx for idx in range(m) if idx not in in_current]
        random.shuffle(not_in_current)
        
        improved_swap = current_family[:]
        swap_improved = False
        for remove_idx in list(improved_swap):
            if time.time() > deadline - 1:
                break
            # Try removing this element and see if we can add a better one
            test_family = [x for x in improved_swap if x != remove_idx]
            added_any = False
            for add_idx in not_in_current[:50]:  # Try first 50 candidates
                if time.time() > deadline - 1:
                    break
                if add_idx == remove_idx:
                    continue
                if not check_violation_fast(test_family, add_idx):
                    test_family.append(add_idx)
                    added_any = True
                    break
            if added_any and len(test_family) > len(improved_swap):
                improved_swap = test_family
                swap_improved = True
        
        if swap_improved and len(improved_swap) > best_size:
            best_size = len(improved_swap)
            best_family_idx = improved_swap[:]

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    # Convert back to sorted lists of 1..16
    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]

    return result
# EVOLVE-BLOCK-END
