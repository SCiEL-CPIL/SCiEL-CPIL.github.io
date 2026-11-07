import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s

    N = 128

    # Use bitmask for fast membership testing
    # S_mask has bit i set if (i+1) is in the set
    def make_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask

    def in_set(mask, x):
        return (mask >> (x - 1)) & 1

    def is_valid_with_mask(mask, candidate, S_list):
        """Check if adding candidate to set represented by mask maintains validity.
        S_list is the list of elements currently in the set (not including candidate)."""

        # New mask would include candidate
        new_mask = mask | (1 << (candidate - 1))

        # 1. Check triples where candidate is a member: (candidate, x, y)
        #    Violation if (candidate+x), (candidate+y), (x+y) are all in new set
        for i in range(len(S_list)):
            x = S_list[i]
            s1 = candidate + x
            if s1 > N:
                continue
            if not in_set(new_mask, s1):
                continue
            for j in range(i + 1, len(S_list)):
                y = S_list[j]
                s2 = candidate + y
                if s2 > N:
                    continue
                if not in_set(new_mask, s2):
                    continue
                s3 = x + y
                if s3 > N:
                    continue
                if in_set(new_mask, s3):
                    return False

        # 2. Check triples where candidate is a sum: (a, b, c) with a+b == candidate
        #    Violation if a+c and b+c are in S (original set, since candidate is the sum)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if a + b == candidate:
                    # Check if there is a c in S (c != a, c != b)
                    # such that a+c in S and b+c in S
                    for k in range(len(S_list)):
                        c = S_list[k]
                        if c == a or c == b:
                            continue
                        ac = a + c
                        bc = b + c
                        if ac <= N and bc <= N:
                            if in_set(mask, ac) and in_set(mask, bc):
                                return False

        return True

    # Two-layer base: upper block [65,128] (sum-free, 64 elems) + odd lower [1,31] (16 elems)
    # Upper block: all pair sums > 128, so no sum is in S
    # Odd lower block: all pair sums are even and > 31, so no sum is in odd lower block
    # Cross sums (odd + upper) are all > 65, so they can only be in upper block
    #   but upper elements are 65..128, and odd+65 >= 66, odd+65 <= 31+128=159
    #   Actually odd x in [1,31], upper y in [65,128]: x+y in [66,159]
    #   For x+y to be in S, it must be in upper block [65,128]
    #   x+y in [66,128] requires y <= 128-x, i.e., y in [65, 128-x]
    #   We need to check: for distinct a,b,c in S, are a+b, a+c, b+c all in S?
    #   If all three are from upper block: sums > 128, OK
    #   If two from upper, one from odd: say a,b upper, c odd
    #     a+b > 128 (not in S), so OK
    #   If one from upper, two from odd: say a upper, b,c odd
    #     a+b in [66,159], a+c in [66,159], b+c even in [4,62]
    #     b+c is even and <= 62, not in upper block [65,128], not in odd block [1,31] (even)
    #     So b+c not in S, OK
    #   If all three from odd: sums are even, not in S, OK
    # So this base is valid!
    base = set(range(65, N + 1)) | set(range(1, 32, 2))
    base_list = sorted(base)
    base_mask = make_mask(base)

    # Greedy: try to add remaining elements from 1 to 64
    S = set(base)
    S_list = list(base)
    S_mask = base_mask
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if x in S:
            continue
        if is_valid_with_mask(S_mask, x, S_list):
            S.add(x)
            S_list.append(x)
            S_mask |= (1 << (x - 1))

    # Local search: try to add more elements by removing some
    # Try random restarts with different orderings
    best_S = set(S)
    best_list = list(S)
    best_mask = S_mask

    # Try multiple random orderings
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline:
            break

        # Start with base and try random ordering of small elements
        S_try = set(base)
        S_try_list = list(base)
        S_try_mask = base_mask
        small_elements = [x for x in range(1, 65) if x not in base]
        rng.shuffle(small_elements)

        for x in small_elements:
            if time.time() > deadline:
                break
            if is_valid_with_mask(S_try_mask, x, S_try_list):
                S_try.add(x)
                S_try_list.append(x)
                S_try_mask |= (1 << (x - 1))

        if len(S_try) > len(best_S):
            best_S = set(S_try)
            best_list = list(S_try)
            best_mask = S_try_mask

    # Further local search: try to add any missing element
    for trial in range(1000):
        if time.time() > deadline:
            break

        S_try = set(best_S)
        S_try_list = list(best_S)
        S_try_mask = best_mask
        improved = False

        # Try to add any element not in S_try
        missing = [x for x in range(1, N + 1) if x not in S_try]
        rng.shuffle(missing)

        for x in missing:
            if time.time() > deadline:
                break
            if is_valid_with_mask(S_try_mask, x, S_try_list):
                S_try.add(x)
                S_try_list.append(x)
                S_try_mask |= (1 << (x - 1))
                improved = True

        if improved and len(S_try) > len(best_S):
            best_S = set(S_try)
            best_list = list(S_try)
            best_mask = S_try_mask

    # Swap-based local search: remove one element, try to add two
    for trial in range(200):
        if time.time() > deadline:
            break

        S_try = set(best_S)
        S_try_list = list(best_S)
        S_try_mask = best_mask

        # Try removing each element and see if we can add two others
        elems_to_try_remove = list(best_S)
        rng.shuffle(elems_to_try_remove)

        for rem in elems_to_try_remove:
            if time.time() > deadline:
                break

            # Remove rem
            S_try.discard(rem)
            S_try_list = [x for x in S_try_list if x != rem]
            S_try_mask &= ~(1 << (rem - 1))

            # Try to add missing elements
            missing = [x for x in range(1, N + 1) if x not in S_try]
            rng.shuffle(missing)

            added_count = 0
            for x in missing:
                if time.time() > deadline:
                    break
                if is_valid_with_mask(S_try_mask, x, S_try_list):
                    S_try.add(x)
                    S_try_list.append(x)
                    S_try_mask |= (1 << (x - 1))
                    added_count += 1
                    if added_count >= 2:
                        break

            # Only keep if we added at least 2 (net gain of 1)
            if added_count >= 2 and len(S_try) > len(best_S):
                best_S = set(S_try)
                best_list = list(S_try)
                best_mask = S_try_mask
            else:
                # Restore
                S_try.add(rem)
                S_try_list.append(rem)
                S_try_mask |= (1 << (rem - 1))

    # 2-opt swap: try swapping one element out for another (net 0) to enable further additions
    for trial in range(50):
        if time.time() > deadline:
            break

        S_try = set(best_S)
        S_try_list = list(best_S)
        S_try_mask = best_mask
        improved = False

        elems_in = list(best_S)
        missing = [x for x in range(1, N + 1) if x not in best_S]
        rng.shuffle(elems_in)
        rng.shuffle(missing)

        for rem in elems_in:
            if time.time() > deadline:
                break
            if improved:
                break
            for add in missing:
                if time.time() > deadline:
                    break
                if improved:
                    break
                # Try swap: remove rem, add add
                S_swap = set(S_try)
                S_swap_list = list(S_try)
                S_swap_mask = S_try_mask
                S_swap.discard(rem)
                S_swap_list = [x for x in S_swap_list if x != rem]
                S_swap_mask &= ~(1 << (rem - 1))
                S_swap.add(add)
                S_swap_list.append(add)
                S_swap_mask |= (1 << (add - 1))

                # Now try to add more
                missing2 = [x for x in range(1, N + 1) if x not in S_swap]
                rng.shuffle(missing2)
                added = 0
                for x in missing2:
                    if time.time() > deadline:
                        break
                    if is_valid_with_mask(S_swap_mask, x, S_swap_list):
                        S_swap.add(x)
                        S_swap_list.append(x)
                        S_swap_mask |= (1 << (x - 1))
                        added += 1

                if len(S_swap) > len(best_S):
                    best_S = S_swap
                    best_list = S_swap_list
                    best_mask = S_swap_mask
                    improved = True

    # Convert to list of Python ints
    result = sorted(best_S)
    return [int(x) for x in result]

