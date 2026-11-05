"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

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

    # Precompute pairwise intersections using numpy for speed
    masks_arr = np.array(all_masks, dtype=np.int64)
    inter = np.zeros((m, m), dtype=np.int64)
    for i in range(m):
        inter[i, i+1:] = masks_arr[i] & masks_arr[i+1:]
        inter[i+1:, i] = inter[i, i+1:]
    # Convert to list of lists for faster indexing in Python loops
    inter_list = inter.tolist()

    def check_violation_idx(family_indices, idx):
        """
        Check if adding index idx to family creates a sunflower violation.
        For each pair (a,b) in family, check if inter[a][b] == inter[a][idx] == inter[b][idx].
        """
        nf = len(family_indices)
        if nf < 2:
            return False
        fidx = family_indices
        inter_local = inter_list
        for ii in range(nf):
            a = fidx[ii]
            an = inter_local[a][idx]
            for jj in range(ii + 1, nf):
                b = fidx[jj]
                if inter_local[a][b] == an and inter_local[b][idx] == an:
                    return True
        return False

    def check_violation_after_removal(family_indices, remove_pos, add_idx):
        """
        Check if replacing family_indices[remove_pos] with add_idx creates a violation.
        Optimized: only need to check triples involving add_idx or pairs that lose a sunflower member.
        Since we're replacing one element, we check:
        1. All pairs (a,b) in remaining family where a sunflower with add_idx could form
        2. Pairs involving add_idx with two remaining family members
        """
        nf = len(family_indices)
        if nf < 3:
            return False
        
        inter_local = inter_list
        # Build set of remaining family indices
        remaining = [family_indices[i] for i in range(nf) if i != remove_pos]
        nrem = len(remaining)
        
        # Check: for each pair (a,b) in remaining, if inter[a][b] == inter[a][add_idx] == inter[b][add_idx]
        # That's a sunflower with add_idx
        for ii in range(nrem):
            a = remaining[ii]
            aa_add = inter_local[a][add_idx]
            for jj in range(ii + 1, nrem):
                b = remaining[jj]
                if inter_local[a][b] == aa_add and inter_local[b][add_idx] == aa_add:
                    return True
        
        # Also need to check: could removing family_indices[remove_pos] and adding add_idx
        # create a sunflower among the remaining members? No, because the remaining members
        # were already valid (no sunflower among them). The only new triples involve add_idx.
        # But wait - we also need to check if there's a sunflower formed by add_idx and two
        # remaining members where the intersection pattern matches. That's exactly what we checked above.
        
        # However, we also need to verify that no existing sunflower is broken in a way that
        # allows a new one... actually no, removing a set can only break sunflowers, not create them.
        # The only new sunflowers can involve add_idx. So the check above is sufficient.
        
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

    # Precompute a heuristic score for each mask: prefer sets with more "spread out" elements
    # This helps greedy find larger families
    def mask_spread(mask):
        """Score based on how spread out the elements are (higher = better for avoiding collisions)"""
        bits = [i for i in range(n) if mask & (1 << i)]
        if len(bits) < 2:
            return 0
        # Sum of pairwise distances
        dist = 0
        for i in range(len(bits)):
            for j in range(i+1, len(bits)):
                dist += abs(bits[i] - bits[j])
        return dist

    spread_scores = [mask_spread(all_masks[i]) for i in range(m)]

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

    # Strategy 2: Greedy by spread score (descending)
    if time.time() < deadline - 5:
        order_by_spread = sorted(range(m), key=lambda x: spread_scores[x], reverse=True)
        family = []
        for idx in order_by_spread:
            if time.time() > deadline:
                break
            if not check_violation_idx(family, idx):
                family.append(idx)
        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Strategy 3: Multiple random shuffles
    random.seed(12345)
    max_attempts = 20

    for attempt in range(max_attempts):
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

    # Strategy 4: Local improvement with swaps
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
            if time.time() > time_budget - 2:
                break
            for i in range(len(improved)):
                if time.time() > time_budget:
                    break
                removed = improved[i]
                for idx in candidates:
                    if time.time() > time_budget:
                        break
                    if not check_violation_after_removal(improved, i, idx):
                        improved[i] = idx
                        in_family.remove(removed)
                        in_family.add(idx)
                        candidates.remove(idx)
                        changed = True
                        break  # Restart the swap loop after a successful swap
        
        return improved

    # Apply local improvement to best so far
    time_budget = deadline - 8
    improved = local_improve(best_family_idx, time_budget)
    if len(improved) > best_size:
        best_size = len(improved)
        best_family_idx = improved[:]

    # Strategy 5: More random restarts with different seeds
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579, 2468, 1357]:
        if time.time() > deadline - 10:
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
            # Quick local improvement on new best
            if time.time() < deadline - 3:
                improved = local_improve(best_family_idx, deadline - 2)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_family_idx = improved[:]

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
