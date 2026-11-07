import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s

    N = 128

    # Start with the safe base: {65, ..., 128}
    base = set(range(65, N + 1))

    # Bitmask for O(1) membership tests
    def make_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m

    def can_add(mask, S, candidate):
        """Check if adding candidate to S (with membership mask) maintains validity."""
        # Case 1: candidate is a sum (a+b == candidate)
        # Violation if exists c in S, c!=a,c!=b, with a+c in S and b+c in S
        for a in range(1, candidate):
            b = candidate - a
            if b <= a:
                break
            if not (mask >> b) & 1:
                continue
            # Need c in S, c != a, c != b, with (a+c) in S and (b+c) in S
            # a+c and b+c must be in 1..128
            lo_c = 1
            hi_c = N - max(a, b)
            if hi_c < lo_c:
                continue
            # Iterate over possible c values in S
            for c in range(lo_c, hi_c + 1):
                if c == a or c == b:
                    continue
                if not (mask >> c) & 1:
                    continue
                ac = a + c
                bc = b + c
                if (mask >> ac) & 1 and (mask >> bc) & 1:
                    return False

        # Case 2: candidate is a member (candidate, x, y) with x,y in S
        # Violation if (candidate+x), (candidate+y), (x+y) are all in S∪{candidate}
        elems = list(S)
        n = len(elems)
        for i in range(n):
            x = elems[i]
            cx = candidate + x
            if cx > N:
                continue
            cx_in = (mask >> cx) & 1 or cx == candidate
            if not cx_in:
                continue
            for j in range(i + 1, n):
                y = elems[j]
                cy = candidate + y
                if cy > N:
                    continue
                cy_in = (mask >> cy) & 1 or cy == candidate
                if not cy_in:
                    continue
                xy = x + y
                if xy > N:
                    continue
                if (mask >> xy) & 1 or xy == candidate:
                    return False

        return True

    # Greedy: try to add elements from 1 to 64
    S = set(base)
    mask = make_mask(S)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if can_add(mask, S, x):
            S.add(x)
            mask |= (1 << x)

    best_S = set(S)
    best_mask = make_mask(best_S)

    # Try multiple random orderings for greedy construction
    rng = random.Random(42)
    small_elements = list(range(1, 65))
    for trial in range(5000):
        if time.time() > deadline:
            break

        S_try = set(base)
        mask_try = make_mask(S_try)
        rng.shuffle(small_elements)

        for x in small_elements:
            if time.time() > deadline:
                break
            if can_add(mask_try, S_try, x):
                S_try.add(x)
                mask_try |= (1 << x)

        if len(S_try) > len(best_S):
            best_S = set(S_try)
            best_mask = mask_try

    # Local search: try to add any missing element
    for trial in range(10000):
        if time.time() > deadline:
            break

        S_try = set(best_S)
        mask_try = best_mask
        improved = False

        missing = [x for x in range(1, N + 1) if x not in S_try]
        rng.shuffle(missing)

        for x in missing:
            if time.time() > deadline:
                break
            if can_add(mask_try, S_try, x):
                S_try.add(x)
                mask_try |= (1 << x)
                improved = True

        if improved and len(S_try) > len(best_S):
            best_S = set(S_try)
            best_mask = mask_try

    # Swap-based local search: remove one element, try to add one or two others
    for trial in range(5000):
        if time.time() > deadline:
            break

        S_try = set(best_S)
        mask_try = best_mask
        improved = False

        # Pick a random element to remove
        elems = list(S_try)
        rng.shuffle(elems)

        for rem in elems:
            if time.time() > deadline:
                break

            S_tmp = set(S_try)
            S_tmp.discard(rem)
            mask_tmp = mask_try & ~(1 << rem)

            # Try to add elements not in S_tmp
            candidates = [x for x in range(1, N + 1) if x not in S_tmp and x != rem]
            rng.shuffle(candidates)

            added = 0
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(mask_tmp, S_tmp, x):
                    S_tmp.add(x)
                    mask_tmp |= (1 << x)
                    added += 1
                    if added >= 2:
                        break

            # Net gain: added elements - 1 removed
            if added > 1:  # net positive
                if len(S_tmp) > len(best_S):
                    best_S = set(S_tmp)
                    best_mask = mask_tmp
                    improved = True
                    break
            elif added == 1 and len(S_tmp) > len(best_S):
                pass

        if improved:
            break

    # Final polish: try to add any remaining missing elements
    S_try = set(best_S)
    mask_try = best_mask
    missing = [x for x in range(1, N + 1) if x not in S_try]
    rng.shuffle(missing)
    for x in missing:
        if time.time() > deadline:
            break
        if can_add(mask_try, S_try, x):
            S_try.add(x)
            mask_try |= (1 << x)

    if len(S_try) > len(best_S):
        best_S = set(S_try)

    # Convert to list of Python ints
    result = sorted(best_S)
    return [int(x) for x in result]
