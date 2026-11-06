import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    # Generate all 4-element subsets of [1,16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    n_total = len(all_masks)  # 1820

    # Precompute pairwise intersections
    # inter[i][j] = all_masks[i] & all_masks[j]
    # We'll use a dict keyed by (i,j) with i<j for memory efficiency
    # Actually, let's use a 2D list for speed
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        inter[i][i] = mi
        for j in range(i + 1, n_total):
            v = mi & all_masks[j]
            inter[i][j] = v
            inter[j][i] = v

    def is_valid_addition(family, family_idx_set, new_mask, new_idx):
        """Check if adding new_mask (at index new_idx) to family creates a sunflower."""
        for i in family_idx_set:
            if i == new_idx:
                continue
            a_inter_new = inter[i][new_idx]
            for j in family_idx_set:
                if j == i or j == new_idx:
                    continue
                if inter[i][j] == a_inter_new and inter[j][new_idx] == a_inter_new:
                    return False
        return True

    best_size = 0
    best_family_indices = []
    rng = random.Random(12345)

    for iteration in range(500):
        if time.time() > deadline:
            break

        shuffled = list(range(n_total))
        rng.shuffle(shuffled)

        family = []
        family_set = set()

        for idx in shuffled:
            if time.time() > deadline:
                break
            if is_valid_addition(family, family_set, all_masks[idx], idx):
                family.append(idx)
                family_set.add(idx)

        if len(family) > best_size:
            best_size = len(family)
            best_family_indices = list(family)
            best_family_set = set(family)

        if best_size >= 250:
            break

    # Local search / refinement
    if best_size > 0 and time.time() < deadline - 3:
        family = list(best_family_indices)
        family_set = set(family)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Try adding any mask not in family
            for idx in range(n_total):
                if time.time() > deadline - 1:
                    break
                if idx in family_set:
                    continue
                if is_valid_addition(family, family_set, all_masks[idx], idx):
                    family.append(idx)
                    family_set.add(idx)
                    improved = True
                    if len(family) >= 250:
                        break
            if len(family) >= 250:
                break
            # Try swaps
            for i in range(len(family)):
                if time.time() > deadline - 1:
                    break
                removed_idx = family[i]
                family_no_i = set(family) - {removed_idx}
                for idx in range(n_total):
                    if time.time() > deadline - 1:
                        break
                    if idx in family_set:
                        continue
                    if is_valid_addition(list(family_no_i), family_no_i, all_masks[idx], idx):
                        family[i] = idx
                        family_set.discard(removed_idx)
                        family_set.add(idx)
                        improved = True
                        break
                if improved:
                    break

        if len(family) > best_size:
            best_size = len(family)
            best_family_indices = list(family)

    # Convert to 1-indexed sorted lists
    result = []
    for idx in best_family_indices:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
