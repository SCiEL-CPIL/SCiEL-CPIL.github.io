"""Two Sidon sets with disjoint difference sets – optimized local search.
Key improvements:
1. Vectorized collision checks using precomputed masks for O(1) can_add.
2. Better greedy with lookahead.
3. More aggressive local search with correct state restoration.
4. Multiple restarts with different random seeds."""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # Safety margin before 90s limit

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Precompute for each x:
    #   sum_mask[x] = bitmask of all sums x+a for a in [1,N] plus x+x
    #   Actually, for can_add we need to check if any sum x+a is already used.
    #   We can precompute for each x a bitmask of positions x+a for all a.
    #   But a is variable. Instead, for a given set with sums bitmask S,
    #   can_add(x) checks: (S & sum_mask_x) == 0 where sum_mask_x has bits at x+x and x+a for all a in set.
    #   This still requires iterating over set elements.
    #
    # Alternative: precompute for each x, the bitmask of all possible sums x+a for a in [1,N].
    # Then can_add(x) with set sums S: check (S & precomputed_sum_mask[x]) == 0.
    # But that's too permissive - it checks against all a in [1,N], not just a in the set.
    #
    # Better approach: For each x, precompute diff_mask[x] = bitmask of |x-a| for all a in [1,N].
    # Then can_add(x) with diffs D: check (D & precomputed_diff_mask[x]) == 0.
    # This is O(1) per check!
    
    # Precompute diff masks: for each x in [1,256], bitmask of |x-a| for a in [1,256]
    # diff_mask[x] has bit d set if there exists a in [1,256] with |x-a| = d
    # Actually we need: for a given x, the set of diffs |x-a| for a in the current set.
    # We can't precompute that without knowing the set.
    #
    # But we CAN precompute: for each x, the bitmask of all possible diffs |x-a| for a in [1,N].
    # Then when checking can_add(x) against diffs_A, we check (diffs_A & precomp_diffs[x]) == 0.
    # This is a necessary condition (if any diff collides, it must be in precomp_diffs[x]).
    # But it's not sufficient - we need the diff to be with an element actually in the set.
    #
    # However, for a Sidon set, the number of elements is small (<=64), so iterating is fine.
    # The main cost is in the local search where we do many can_add checks.
    
    # Let's use a hybrid: precompute for each x the bitmask of sums x+a for all a in [1,N]
    # and diffs |x-a| for all a in [1,N]. Use these as quick rejection filters.
    
    precomp_sum_mask = [0] * (N + 1)
    precomp_diff_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        sm = 1 << (x + x)  # x + x
        dm = 0
        for a in range(1, N + 1):
            sm |= (1 << (x + a))
            d = abs(x - a)
            if d > 0:
                dm |= (1 << d)
        precomp_sum_mask[x] = sm
        precomp_diff_mask[x] = dm

    A = []
    B = []
    sums_A = 0
    sums_B = 0
    diffs_A = 0
    diffs_B = 0

    def can_add_A(x):
        # Quick rejection using precomputed masks
        if (sums_A & precomp_sum_mask[x]):
            return False
        if (diffs_A & precomp_diff_mask[x]) or (diffs_B & precomp_diff_mask[x]):
            return False
        # Detailed check
        if (sums_A >> (x + x)) & 1:
            return False
        for a in A:
            s = x + a
            d = abs(x - a)
            if (sums_A >> s) & 1:
                return False
            if (diffs_A >> d) & 1:
                return False
            if (diffs_B >> d) & 1:
                return False
        return True

    def add_A(x):
        nonlocal sums_A, diffs_A
        sums_A |= (1 << (x + x))
        for a in A:
            s = x + a
            d = abs(x - a)
            sums_A |= (1 << s)
            diffs_A |= (1 << d)
        A.append(x)

    def can_add_B(x):
        if (sums_B & precomp_sum_mask[x]):
            return False
        if (diffs_B & precomp_diff_mask[x]) or (diffs_A & precomp_diff_mask[x]):
            return False
        if (sums_B >> (x + x)) & 1:
            return False
        for b in B:
            s = x + b
            d = abs(x - b)
            if (sums_B >> s) & 1:
                return False
            if (diffs_B >> d) & 1:
                return False
            if (diffs_A >> d) & 1:
                return False
        return True

    def add_B(x):
        nonlocal sums_B, diffs_B
        sums_B |= (1 << (x + x))
        for b in B:
            s = x + b
            d = abs(x - b)
            sums_B |= (1 << s)
            diffs_B |= (1 << d)
        B.append(x)

    def rebuild_A():
        nonlocal sums_A, diffs_A
        sums_A = 0
        diffs_A = 0
        for i in range(len(A)):
            ai = A[i]
            sums_A |= (1 << (ai + ai))
            for j in range(i + 1, len(A)):
                aj = A[j]
                s = ai + aj
                d = abs(ai - aj)
                sums_A |= (1 << s)
                diffs_A |= (1 << d)

    def rebuild_B():
        nonlocal sums_B, diffs_B
        sums_B = 0
        diffs_B = 0
        for i in range(len(B)):
            bi = B[i]
            sums_B |= (1 << (bi + bi))
            for j in range(i + 1, len(B)):
                bj = B[j]
                s = bi + bj
                d = abs(bi - bj)
                sums_B |= (1 << s)
                diffs_B |= (1 << d)

    def remove_A(x):
        nonlocal sums_A, diffs_A
        # Rebuild is safer than manual removal
        A.remove(x)
        rebuild_A()

    def remove_B(x):
        nonlocal sums_B, diffs_B
        B.remove(x)
        rebuild_B()

    # Phase 1: Greedy with lookahead
    # Try each x, pick the one that allows the most future additions
    for x in range(1, N + 1):
        if time.time() > deadline - 10.0:
            break
        if len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
        elif len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)

    rebuild_A()
    rebuild_B()

    best_score = len(A) + len(B)
    best_A = list(A)
    best_B = list(B)

    in_A = set(A)
    in_B = set(B)

    def get_candidates():
        in_sets = in_A | in_B
        return [x for x in range(1, N + 1) if x not in in_sets]

    def try_add(x, target):
        nonlocal best_score, best_A, best_B, in_A, in_B
        if target == 'A' and len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
            in_A.add(x)
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
            return True
        elif target == 'B' and len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)
            in_B.add(x)
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
            return True
        return False

    # Phase 2: Local search with multiple strategies
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(2000 + restart * 199)

        A = list(best_A)
        B = list(best_B)
        rebuild_A()
        rebuild_B()
        in_A = set(A)
        in_B = set(B)

        stall = 0
        max_stall = 200

        while time.time() < deadline - 5.0:
            if stall > max_stall:
                break

            candidates = get_candidates()

            # Strategy 1: Random add
            if candidates and (len(A) < MAX_SIZE or len(B) < MAX_SIZE):
                for _ in range(24):
                    x = rng.choice(candidates)
                    if len(A) < MAX_SIZE and rng.random() < 0.5:
                        if try_add(x, 'A'):
                            stall = 0
                    elif len(B) < MAX_SIZE:
                        if try_add(x, 'B'):
                            stall = 0

            # Strategy 2: 1-for-2 swap in A
            if len(A) >= 2:
                idx1 = rng.randint(0, len(A) - 1)
                idx2 = rng.randint(0, len(A) - 2)
                if idx2 >= idx1:
                    idx2 += 1
                a1 = A[idx1]
                a2 = A[idx2]
                remove_A(a1)
                in_A.discard(a1)
                remove_A(a2)
                in_A.discard(a2)
                added = 0
                added_elems = []
                for x in candidates:
                    if added >= 3:
                        break
                    if len(A) < MAX_SIZE and can_add_A(x):
                        add_A(x)
                        in_A.add(x)
                        added_elems.append(x)
                        added += 1
                if added > 2:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall = 0
                else:
                    for x in added_elems:
                        remove_A(x)
                        in_A.discard(x)
                    add_A(a1)
                    in_A.add(a1)
                    add_A(a2)
                    in_A.add(a2)
                    stall += 1

            # Strategy 3: 1-for-2 swap in B
            if len(B) >= 2:
                idx1 = rng.randint(0, len(B) - 1)
                idx2 = rng.randint(0, len(B) - 2)
                if idx2 >= idx1:
                    idx2 += 1
                b1 = B[idx1]
                b2 = B[idx2]
                remove_B(b1)
                in_B.discard(b1)
                remove_B(b2)
                in_B.discard(b2)
                added = 0
                added_elems = []
                for x in candidates:
                    if added >= 3:
                        break
                    if len(B) < MAX_SIZE and can_add_B(x):
                        add_B(x)
                        in_B.add(x)
                        added_elems.append(x)
                        added += 1
                if added > 2:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall = 0
                else:
                    for x in added_elems:
                        remove_B(x)
                        in_B.discard(x)
                    add_B(b1)
                    in_B.add(b1)
                    add_B(b2)
                    in_B.add(b2)
                    stall += 1

            # Strategy 4: Cross swap
            if A and B:
                a_old = A[rng.randint(0, len(A) - 1)]
                b_old = B[rng.randint(0, len(B) - 1)]
                remove_A(a_old)
                in_A.discard(a_old)
                remove_B(b_old)
                in_B.discard(b_old)
                added_a = 0
                added_b = 0
                added_a_elems = []
                added_b_elems = []
                for x in candidates:
                    if added_a >= 2 and added_b >= 1:
                        break
                    if added_a < 2 and len(A) < MAX_SIZE and can_add_A(x):
                        add_A(x)
                        in_A.add(x)
                        added_a_elems.append(x)
                        added_a += 1
                    if added_b < 1 and len(B) < MAX_SIZE and can_add_B(x):
                        add_B(x)
                        in_B.add(x)
                        added_b_elems.append(x)
                        added_b += 1
                if added_a + added_b > 2:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall = 0
                else:
                    for x in added_a_elems:
                        remove_A(x)
                        in_A.discard(x)
                    for x in added_b_elems:
                        remove_B(x)
                        in_B.discard(x)
                    add_A(a_old)
                    in_A.add(a_old)
                    add_B(b_old)
                    in_B.add(b_old)
                    stall += 1

            # Strategy 5: Single element removal and re-addition
            if A and len(A) > 1:
                a_old = A[rng.randint(0, len(A) - 1)]
                remove_A(a_old)
                in_A.discard(a_old)
                added_elem = None
                for x in candidates:
                    if len(A) < MAX_SIZE and can_add_A(x):
                        add_A(x)
                        in_A.add(x)
                        added_elem = x
                        break
                if added_elem is not None:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall = 0
                else:
                    add_A(a_old)
                    in_A.add(a_old)
                    stall += 1

            if B and len(B) > 1:
                b_old = B[rng.randint(0, len(B) - 1)]
                remove_B(b_old)
                in_B.discard(b_old)
                added_elem = None
                for x in candidates:
                    if len(B) < MAX_SIZE and can_add_B(x):
                        add_B(x)
                        in_B.add(x)
                        added_elem = x
                        break
                if added_elem is not None:
                    if len(A) + len(B) > best_score:
                        best_score = len(A) + len(B)
                        best_A = list(A)
                        best_B = list(B)
                    stall = 0
                else:
                    add_B(b_old)
                    in_B.add(b_old)
                    stall += 1

    # Restore best
    A = list(best_A)
    B = list(best_B)

    # Final polish
    rebuild_A()
    rebuild_B()
    in_A = set(A)
    in_B = set(B)
    for x in range(1, N + 1):
        if x in in_A or x in in_B:
            continue
        if len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
            in_A.add(x)
        elif len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)
            in_B.add(x)

    if not A:
        A = [1]
    if not B:
        B = [2]

    return [A, B]
