"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 512
    deadline = time.time() + 55.0  # Safety margin for 60s limit

    # Precompute all 4-term arithmetic progressions
    # For each position x (1-indexed), precompute a bitmask of "forbidden triples"
    # that would complete a 4-AP with x.
    # If x is added, a 4-AP is formed iff the other 3 positions are all selected.
    # We store for each x a list of masks (each mask has exactly 3 bits set).
    # Adding x is valid iff no mask is a subset of the current selected set.

    # For position x, find all 4-APs containing x and store the other 3 as a bitmask
    forbidden_masks = [[] for _ in range(n + 1)]  # 1-indexed

    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                # 4-AP: (a, b, c, e)
                # If adding a: forbidden if b,c,e all selected
                # If adding b: forbidden if a,c,e all selected
                # If adding c: forbidden if a,b,e all selected
                # If adding e: forbidden if a,b,c all selected
                mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
                mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
                mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
                mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                forbidden_masks[a].append(mask_a)
                forbidden_masks[b].append(mask_b)
                forbidden_masks[c].append(mask_c)
                forbidden_masks[e].append(mask_e)

    def can_add(x, selected_mask):
        """Check if x can be added without forming a 4-AP."""
        for mask in forbidden_masks[x]:
            if (mask & selected_mask) == mask:
                return False
        return True

    def greedy_order(order, selected_mask=0):
        """Greedy add elements in given order, return final mask."""
        for x in order:
            if (selected_mask >> (x - 1)) & 1:
                continue
            if can_add(x, selected_mask):
                selected_mask |= (1 << (x - 1))
        return selected_mask

    def mask_to_list(mask):
        """Convert bitmask to sorted list of 1-indexed positions."""
        result = []
        for i in range(n):
            if (mask >> i) & 1:
                result.append(i + 1)
        return result

    def popcount(mask):
        """Count bits set in mask."""
        return bin(mask).count('1')

    # Deterministic baseline
    best_mask = greedy_order(range(1, n + 1))
    best_size = popcount(best_mask)

    # Randomized greedy with restarts using bitsets
    candidates = list(range(1, n + 1))

    while time.time() < deadline:
        random.shuffle(candidates)
        cur_mask = greedy_order(candidates)
        cur_size = popcount(cur_mask)

        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask

        # Local search: try to swap elements to increase size
        if time.time() > deadline - 5.0:
            break

        # Try several local search steps
        for _ in range(50):
            if time.time() > deadline - 3.0:
                break

            # Try to add a random unselected element
            # First, find unselected elements
            unselected = []
            for i in range(1, n + 1):
                if not (cur_mask >> (i - 1)) & 1:
                    unselected.append(i)

            if not unselected:
                break

            # Try adding a random unselected element
            y = random.choice(unselected)
            if can_add(y, cur_mask):
                cur_mask |= (1 << (y - 1))
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
                continue

            # Try a swap: remove a random selected element, then try to add y
            selected_list = []
            for i in range(1, n + 1):
                if (cur_mask >> (i - 1)) & 1:
                    selected_list.append(i)

            if not selected_list:
                break

            x = random.choice(selected_list)
            new_mask = cur_mask & ~(1 << (x - 1))
            if can_add(y, new_mask):
                cur_mask = new_mask | (1 << (y - 1))
                # cur_size stays the same
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask

            # Try to add a different random element after removal
            if len(unselected) > 1:
                for _ in range(5):
                    y2 = random.choice(unselected)
                    if y2 == y:
                        continue
                    if can_add(y2, new_mask):
                        cur_mask = new_mask | (1 << (y2 - 1))
                        if cur_size > best_size:
                            best_size = cur_size
                            best_mask = cur_mask
                        break

    return mask_to_list(best_mask)

# EVOLVE-BLOCK-END
