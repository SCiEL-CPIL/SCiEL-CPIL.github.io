"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    n = 16
    k = 4

    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    mask_to_set = {}
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_masks.append(mask)
        mask_to_set[mask] = sorted(c)

    num_sets = len(all_masks)

    # Precompute submasks for each set (for intersection checks)
    # Since k=4, each set has 16 submasks (2^4)
    # We only care about intersections of size 0, 1, 2, 3 (not 4, since sets are distinct)
    submasks = [None] * (1 << n)
    for mask in all_masks:
        subs = []
        # Get indices of bits set in mask
        indices = []
        temp = mask
        idx = 0
        while temp:
            if temp & 1:
                indices.append(idx)
            temp >>= 1
            idx += 1

        for r in range(0, 4):  # 0 to 3
            for sub_idx in combinations(indices, r):
                sub_mask = 0
                for x in sub_idx:
                    sub_mask |= (1 << x)
                subs.append(sub_mask)
        submasks[mask] = subs

    def can_add(c_mask, family):
        """
        Check if c_mask can be added to family without creating a sunflower.
        Uses the insight from inspiration 1: group family members by their
        intersection with c_mask. If two sets A, B in family have the same
        intersection K with c_mask, then we must check if A & B == K.
        """
        # Group family members by their intersection with c_mask
        inter_map = {}
        for a in family:
            inter = a & c_mask
            if inter not in inter_map:
                inter_map[inter] = [a]
            else:
                lst = inter_map[inter]
                # Check new set 'a' against all existing in lst
                for b in lst:
                    if (a & b) == inter:
                        return False
                lst.append(a)
        return True

    best_family = []
    best_size = 0

    rng = random.Random(42)

    # Run multiple greedy passes with different orderings
    while time.time() < deadline:
        order = list(all_masks)
        rng.shuffle(order)

        current_family = []
        for c_mask in order:
            if time.time() > deadline - 2.0:
                break
            if can_add(c_mask, current_family):
                current_family.append(c_mask)

        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family

        # If we reached the maximum, break early
        if best_size >= 250:
            break

    # Convert best family masks back to sorted lists
    result = [mask_to_set[m] for m in best_family]

    return result

# EVOLVE-BLOCK-END
