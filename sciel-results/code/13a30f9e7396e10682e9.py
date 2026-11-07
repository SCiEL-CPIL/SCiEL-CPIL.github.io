import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit

    # Precompute conflict structure:
    # Adding x to S creates a forbidden triple iff there exist distinct b,c in S
    # such that b+x in S, c+x in S, and (b+c in S or b+c == x).
    #
    # We precompute for each x the set of "witness pairs" (b,c) with b < c,
    # both in 1..128-x, such that b+c <= 128 or b+c == x.
    # Then can_add(S, x) is: no witness pair (b,c) for x has both b and c in S.
    #
    # Represent each witness pair as a 2-element bitmask: (1<<b) | (1<<c).
    # can_add checks: for each pair_mask in witness_pairs[x], if (S & pair_mask) == pair_mask, reject.

    # Precompute witness pairs for each x
    witness_pairs = [[] for _ in range(129)]  # index 1..128
    for x in range(1, 129):
        # b, c must be distinct, in 1..128-x (so b+x, c+x <= 128)
        # and b+c in 1..128 or b+c == x
        # b < c to avoid duplicates
        max_val = 128 - x
        for b in range(1, max_val + 1):
            for c in range(b + 1, max_val + 1):
                s = b + c
                if s == x or s <= 128:
                    witness_pairs[x].append((1 << b) | (1 << c))

    def can_add(S_mask, x):
        """Check if adding x to S_mask results in a valid set."""
        for pair_mask in witness_pairs[x]:
            if (S_mask & pair_mask) == pair_mask:
                return False
        return True

    def greedy_construct(order, S_mask):
        """Greedy construction given a fixed order of candidates."""
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask, max_passes=50):
        """Improve S via additions and systematic swaps."""
        S_set = set()
        temp = S_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S_set.add(idx)
            temp -= lsb

        not_in_S = [x for x in range(1, 129) if x not in S_set]

        for _ in range(max_passes):
            if time.time() > deadline:
                break

            improved = False
            random.shuffle(not_in_S)

            for x in not_in_S[:]:
                if time.time() > deadline:
                    break

                # Try to add x directly
                if can_add(S_mask, x):
                    S_mask |= (1 << x)
                    S_set.add(x)
                    not_in_S.remove(x)
                    improved = True
                    continue

                # Try swaps: remove y from S, add x
                S_list = list(S_set)
                random.shuffle(S_list)

                found_swap = False
                for y in S_list[:20]:
                    if time.time() > deadline:
                        break
                    S_temp_mask = S_mask ^ (1 << y)
                    if can_add(S_temp_mask, x):
                        S_mask = S_temp_mask | (1 << x)
                        S_set.remove(y)
                        S_set.add(x)
                        not_in_S.append(y)
                        if x in not_in_S:
                            not_in_S.remove(x)
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    break

            if not improved:
                break

        return S_mask

    # Phase 1: Multiple greedy constructions with diverse orderings
    random.seed(42)

    best_mask = 0

    # Ascending
    S_mask = greedy_construct(list(range(1, 129)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    # Descending
    S_mask = greedy_construct(list(range(128, 0, -1)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    # Several random orders
    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask

    # Phase 2: Local search from best set
    if time.time() < deadline:
        best_mask = local_search(best_mask, max_passes=30)

    # Phase 3: More random restarts with local search
    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        S_mask = local_search(S_mask, max_passes=20)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask

    # Final: try to add any remaining elements
    if time.time() < deadline:
        for x in range(1, 129):
            if (best_mask >> x) & 1:
                continue
            if can_add(best_mask, x):
                best_mask |= (1 << x)

    # One more local search pass on the final best set
    if time.time() < deadline and best_mask.bit_count() < 128:
        best_mask = local_search(best_mask, max_passes=10)

    # Convert mask to list of ints
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(int(idx))
        temp -= lsb

    return result
