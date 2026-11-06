import time
import random
from itertools import combinations
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s limit

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)

    n_total = len(all_masks)  # 1820
    masks_arr = np.array(all_masks, dtype=np.int32)

    # Precompute pairwise intersection matrix
    # inter[i, j] = all_masks[i] & all_masks[j]
    inter = np.bitwise_and(masks_arr[:, np.newaxis], masks_arr[np.newaxis, :])

    # For the sunflower condition: three distinct sets A, B, C form a sunflower
    # if (A&B) == (A&C) == (B&C)
    # This means: for a candidate C and existing pair (A, B) in family,
    # C conflicts if (C&A) == (A&B) and (C&B) == (A&B)
    # i.e., (C&A) == (C&B) == (A&B)

    # Strategy: greedy with conflict tracking
    # For each candidate C, we need to check against all pairs in the family.
    # Precompute for each C, the set of pairs (i,j) that would conflict with C.

    # A pair (i, j) conflicts with C iff:
    #   (C & mask_i) == (mask_i & mask_j) AND (C & mask_j) == (mask_i & mask_j)
    # i.e., (C & mask_i) == inter[i,j] AND (C & mask_j) == inter[i,j]

    # For efficiency, we can precompute for each C and each i:
    #   C & mask_i
    # Then for a pair (i,j) in the family, C conflicts if:
    #   (C & mask_i) == inter[i,j] AND (C & mask_j) == inter[i,j]

    # Let's precompute: for each candidate c_idx and each family member i_idx,
    #   inter_c_i[c_idx, i_idx] = all_masks[c_idx] & all_masks[i_idx]
    # This is just the inter matrix itself (symmetric).

    # So inter[c_idx, i_idx] = all_masks[c_idx] & all_masks[i_idx]

    # For a greedy algorithm:
    # Maintain a set of selected indices.
    # For candidate C, check if there exist two selected indices i, j such that:
    #   inter[C, i] == inter[i, j] AND inter[C, j] == inter[i, j]
    # which simplifies to: inter[C, i] == inter[i, j] == inter[C, j]

    # Efficient check: for each selected i, compute inter[C, i].
    # Then for each pair of selected indices (i, j) where inter[C,i] == inter[C,j],
    # check if inter[i,j] == inter[C,i].

    # Even better: for candidate C, compute the vector v[i] = inter[C, i] for all selected i.
    # Group selected indices by their v[i] value.
    # For each group G with value w, check if for any pair (i,j) in G, inter[i,j] == w.
    # If yes, conflict.

    # Let's implement this:

    def can_add_fast(C_idx, selected_set, selected_list):
        """Check if C_idx can be added given selected_set (set of indices) and selected_list (list)."""
        if len(selected_list) < 2:
            return True

        # Compute intersections of C with each selected element
        C_mask = all_masks[C_idx]
        # For each selected index i, compute C_mask & all_masks[i]
        # We can do this vectorized if selected_list is not too large

        sel_arr = np.array(selected_list, dtype=np.int32)
        C_inter = np.bitwise_and(C_mask, masks_arr[sel_arr])  # shape (len(selected_list),)

        # Group by intersection value
        # For each unique value w in C_inter, get the indices in selected_list that have that value
        # Then check if any pair within that group has inter[i,j] == w

        # Use a dictionary to group
        groups = {}
        for k, val in enumerate(C_inter):
            val_int = int(val)
            if val_int not in groups:
                groups[val_int] = []
            groups[val_int].append(k)  # k is index into sel_arr

        for w, idxs in groups.items():
            if len(idxs) < 2:
                continue
            # Check all pairs in idxs
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i_global = sel_arr[idxs[a]]
                    j_global = sel_arr[idxs[b]]
                    if inter[i_global, j_global] == w:
                        return False
        return True

    # Greedy with multiple random restarts
    best_family = []
    best_size = 0

    for trial in range(30):
        if time.time() > deadline - 10:
            break

        order = list(range(n_total))
        random.shuffle(order)

        selected_list = []
        selected_set = set()

        for idx in order:
            if time.time() > deadline - 10:
                break
            if idx in selected_set:
                continue
            if can_add_fast(idx, selected_set, selected_list):
                selected_list.append(idx)
                selected_set.add(idx)

        if len(selected_list) > best_size:
            best_size = len(selected_list)
            best_family = selected_list[:]

    # Local search: try to add more sets
    family = best_family[:]
    fam_set = set(family)

    # Try to add any remaining set
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(n_total):
            if idx in fam_set:
                continue
            if can_add_fast(idx, fam_set, family):
                family.append(idx)
                fam_set.add(idx)
                improved = True

    # Swap-based local search
    T = 1.0
    cooling = 0.999
    while time.time() < deadline - 2 and T > 0.005:
        if not family:
            break

        idx_out = random.randint(0, len(family) - 1)
        mask_out = family[idx_out]

        # Find a random mask not in family
        idx_in = random.randint(0, n_total - 1)
        attempts = 0
        while idx_in in fam_set and attempts < 20:
            idx_in = random.randint(0, n_total - 1)
            attempts += 1
        if idx_in in fam_set:
            T *= cooling
            continue

        # Check if idx_in can be added to family without mask_out
        fam_temp = [x for k, x in enumerate(family) if k != idx_out]
        if can_add_fast(idx_in, set(fam_temp), fam_temp):
            family = fam_temp + [idx_in]
            fam_set.remove(mask_out)
            fam_set.add(idx_in)

        T *= cooling

    # Final attempt to add
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for idx in range(n_total):
            if idx in fam_set:
                continue
            if can_add_fast(idx, fam_set, family):
                family.append(idx)
                fam_set.add(idx)
                improved = True

    # Convert back to sorted lists
    result = []
    for idx in family:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elements)

    return result
