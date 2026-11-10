import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill

    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability: for each mask, list of masks it's comparable with
    # comparable[a][b] = 1 if a is strict subset of b or b is strict subset of a
    # We'll use a bitset approach: for each mask, store which other masks it's comparable with

    # Precompute popcount for efficiency
    popcount = [0] * total_masks
    for i in range(1, total_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Precompute for each mask, the list of masks that are supersets (strict)
    # and subsets (strict) - actually we just need comparability
    # For speed, precompute a boolean matrix as a list of bytearrays or just use bit operations on the fly

    # Actually, for 4096 masks, we can precompute a 4096x4096 comparability matrix
    # but that's 16M entries. Let's use a more compact approach.

    # Precompute: for each mask x, the set of masks y such that x is a strict subset of y
    # This is: y = x | z where z is a non-empty subset of ~x
    # And the set of masks y such that y is a strict subset of x
    # This is: y is a proper subset of x

    # For local search, we need to quickly compute:
    # delta = (comparables of new_mask with others) - (comparables of old_mask with others)
    # where "others" = current family minus old_mask

    # Let's precompute for each mask, a sorted list of all masks it's comparable with
    # Then we can quickly count how many of those are in the current family

    # Actually, the most efficient approach for 40 sets:
    # For a swap (remove old, add new):
    #   gain = |{x in family\{old} : comparable(new, x)}| - |{x in family\{old} : comparable(old, x)}|
    # We can precompute for each mask a bitset of comparable masks (4096 bits = 512 bytes)
    # Then intersection with family bitset gives the count

    # Let's use Python integers as bitsets (4096 bits fits in a 512-byte int)
    # comparable_bitset[x] = integer where bit y is 1 iff x and y are comparable (x != y)

    comparable_bitset = [0] * total_masks
    for x in range(total_masks):
        mask = 0
        for y in range(total_masks):
            if x == y:
                continue
            if (x & y) == x or (x & y) == y:
                mask |= (1 << y)
        comparable_bitset[x] = mask

    def is_comparable(a, b):
        if a == b:
            return 0
        return 1 if ((a & b) == a or (a & b) == b) else 0

    def calc_score(masks):
        count = 0
        N = len(masks)
        for i in range(N):
            for j in range(i + 1, N):
                if (masks[i] & masks[j]) == masks[i]:
                    if masks[i] != masks[j]:
                        count += 1
                elif (masks[i] & masks[j]) == masks[j]:
                    if masks[i] != masks[j]:
                        count += 1
        return count

    def greedy_construct(seed):
        random.seed(seed)
        current = []
        current_set = set()

        # Start with a chain: empty, {0}, {0,1}, ..., {0,...,11}
        chain_mask = 0
        for i in range(n + 1):
            current.append(chain_mask)
            current_set.add(chain_mask)
            if i < n:
                chain_mask |= (1 << i)

        # Greedily add remaining
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)

        while len(current) < m:
            if time.time() > deadline - 15:
                break
            best_x = -1
            best_score = -1
            for x in all_masks:
                if x in current_set:
                    continue
                score = 0
                for y in current:
                    score += is_comparable(x, y)
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x != -1:
                current.append(best_x)
                current_set.add(best_x)
            else:
                break
        return current

    def local_search(masks, time_limit):
        current = masks[:]
        current_set = set(current)
        current_score = calc_score(current)
        N = len(current)

        # Precompute family bitset for quick comparability counting
        # family_bits: integer with bit i set if mask i is in family
        def make_family_bits(fam):
            bits = 0
            for m in fam:
                bits |= (1 << m)
            return bits

        family_bits = make_family_bits(current)

        def count_comparables_in_family(mask, fam_bits):
            # Count how many masks in fam_bits are comparable with mask
            comp = comparable_bitset[mask]
            # Intersection count
            inter = comp & fam_bits
            return bin(inter).count('1')

        # For a swap: remove old, add new
        # delta = count_comparables(new, family - old) - count_comparables(old, family - old)
        # count_comparables(old, family - old) = count_comparables(old, family) - 1 (since old is not comparable with itself)
        # Actually, comparable_bitset[old] doesn't include old itself, so count_comparables(old, family) already excludes old
        # So: delta = count_comparables(new, family - old) - count_comparables(old, family)
        # count_comparables(new, family - old) = count_comparables(new, family) - is_comparable(new, old)

        improved = True
        stall_count = 0

        while time.time() < time_limit:
            if not improved:
                stall_count += 1
                if stall_count > 100:
                    break
            else:
                stall_count = 0
            improved = False

            # Find best swap
            best_delta = 0
            best_i = -1
            best_x = -1

            # Try all positions and a sample of candidates
            # For efficiency, try all i and a random subset of x
            random.shuffle(range(N))

            for i in range(N):
                old = current[i]
                # Count comparables of old in family (excluding old itself)
                old_comp = count_comparables_in_family(old, family_bits)

                # Try random candidates
                for _ in range(50):
                    x = random.randint(0, total_masks - 1)
                    if x in current_set:
                        continue
                    # Count comparables of x in family (excluding old)
                    # First count in full family, then subtract if comparable with old
                    x_comp_full = count_comparables_in_family(x, family_bits)
                    if is_comparable(x, old):
                        x_comp_no_old = x_comp_full - 1
                    else:
                        x_comp_no_old = x_comp_full

                    delta = x_comp_no_old - old_comp
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_x = x

            if best_i != -1 and best_delta > 0:
                old = current[best_i]
                current[best_i] = best_x
                current_set.remove(old)
                current_set.add(best_x)
                family_bits = make_family_bits(current)
                current_score += best_delta
                improved = True

        return current, current_score

    # Run multiple restarts with different seeds
    best_result = None
    best_score = -1

    # First, try a few greedy constructions
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        masks = greedy_construct(seed)
        if len(masks) < m:
            # Pad with random masks
            current_set = set(masks)
            while len(masks) < m:
                r = random.randint(0, total_masks - 1)
                if r not in current_set:
                    masks.append(r)
                    current_set.add(r)

        # Local search
        refined, score = local_search(masks, deadline - 5)
        if score > best_score:
            best_score = score
            best_result = refined

    # If we have time, do more local search on the best result
    if time.time() < deadline - 3 and best_result is not None:
        refined, score = local_search(best_result, deadline - 1)
        if score > best_score:
            best_score = score
            best_result = refined

    # Convert to list of lists
    result = []
    for mask in best_result:
        lst = []
        for b in range(n):
            if (mask >> b) & 1:
                lst.append(b)
        result.append(lst)

    return result
