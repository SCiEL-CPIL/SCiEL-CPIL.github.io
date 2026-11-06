import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 87.0  # Safety margin for 90s kill limit

    n = 256
    best_A = []
    best_B = []
    best_size = 0

    # Precompute: for each x, the "sum conflicts" with a set S are
    # x+s in sums_S for any s in S, or 2x in sums_S.
    # For diffs: x-s or s-x in D_other.

    def can_add(A, sums_A, diffs_A, diffs_B, x):
        # Sidon check for A
        for s in A:
            if x + s in sums_A:
                return False
        if 2 * x in sums_A:
            return False
        # Diff disjointness check
        for s in A:
            d1 = x - s
            d2 = s - x
            if d1 in diffs_B or d2 in diffs_B:
                return False
        return True

    def add_to_set(S, sums_S, diffs_S, x):
        S.add(x)
        for s in S:
            if s == x:
                continue
            sums_S.add(x + s)
            diffs_S.add(x - s)
            diffs_S.add(s - x)
        sums_S.add(2 * x)

    def recompute(S):
        sums = set()
        diffs = set()
        S_list = sorted(S)
        k = len(S_list)
        for i in range(k):
            for j in range(i, k):
                sums.add(S_list[i] + S_list[j])
            for j in range(k):
                if i != j:
                    diffs.add(S_list[i] - S_list[j])
        return sums, diffs

    def is_valid_state(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        sums_A, diffs_A = recompute(A)
        sums_B, diffs_B = recompute(B)
        # Check Sidon
        expected_sums_A = len(A) * (len(A) + 1) // 2
        expected_sums_B = len(B) * (len(B) + 1) // 2
        if len(sums_A) != expected_sums_A:
            return False
        if len(sums_B) != expected_sums_B:
            return False
        if diffs_A & diffs_B:
            return False
        return True

    def local_search(A, B, time_limit):
        A = set(A)
        B = set(B)

        sums_A, diffs_A = recompute(A)
        sums_B, diffs_B = recompute(B)

        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break

            # Try adds first
            candidates = [x for x in range(1, n + 1) if x not in A and x not in B]
            random.shuffle(candidates)

            for x in candidates[:30]:
                if len(A) < 64 and can_add(A, sums_A, diffs_A, diffs_B, x):
                    add_to_set(A, sums_A, diffs_A, x)
                    improved = True
                    break
                if len(B) < 64 and can_add(B, sums_B, diffs_B, diffs_A, x):
                    add_to_set(B, sums_B, diffs_B, x)
                    improved = True
                    break

            if improved:
                continue

            # Try moves A->B
            for x in list(A):
                if len(B) >= 64:
                    continue
                A_temp = A - {x}
                _, diffs_A_temp = recompute(A_temp)
                if can_add(B, sums_B, diffs_B, diffs_A_temp, x):
                    A.remove(x)
                    sums_A, diffs_A = recompute(A)
                    add_to_set(B, sums_B, diffs_B, x)
                    improved = True
                    break

            if improved:
                continue

            # Try moves B->A
            for x in list(B):
                if len(A) >= 64:
                    continue
                B_temp = B - {x}
                _, diffs_B_temp = recompute(B_temp)
                if can_add(A, sums_A, diffs_A, diffs_B_temp, x):
                    B.remove(x)
                    sums_B, diffs_B = recompute(B)
                    add_to_set(A, sums_A, diffs_A, x)
                    improved = True
                    break

            if improved:
                continue

            # Swaps in A
            for _ in range(15):
                if not A:
                    break
                x = random.choice(list(A))
                y_candidates = [y for y in range(1, n + 1) if y not in A and y not in B]
                if not y_candidates:
                    break
                y = random.choice(y_candidates)

                A_temp = A - {x}
                sums_A_temp, diffs_A_temp = recompute(A_temp)
                if can_add(A_temp, sums_A_temp, diffs_A_temp, diffs_B, y):
                    A.remove(x)
                    sums_A, diffs_A = recompute(A)
                    add_to_set(A, sums_A, diffs_A, y)
                    improved = True
                    break

            if improved:
                continue

            # Swaps in B
            for _ in range(15):
                if not B:
                    break
                x = random.choice(list(B))
                y_candidates = [y for y in range(1, n + 1) if y not in A and y not in B]
                if not y_candidates:
                    break
                y = random.choice(y_candidates)

                B_temp = B - {x}
                sums_B_temp, diffs_B_temp = recompute(B_temp)
                if can_add(B_temp, sums_B_temp, diffs_B_temp, diffs_A, y):
                    B.remove(x)
                    sums_B, diffs_B = recompute(B)
                    add_to_set(B, sums_B, diffs_B, y)
                    improved = True
                    break

        return list(A), list(B)

    num_restarts = 80

    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        A = set()
        B = set()
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()

        # Randomized Greedy Construction
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        for x in candidates:
            if time.time() - start_time > time_limit:
                break

            if len(A) < 64 and can_add(A, sums_A, diffs_A, diffs_B, x):
                add_to_set(A, sums_A, diffs_A, x)
                continue

            if len(B) < 64 and can_add(B, sums_B, diffs_B, diffs_A, x):
                add_to_set(B, sums_B, diffs_B, x)
                continue

        # Local Search
        A, B = local_search(A, B, time_limit)

        # Validate
        if is_valid_state(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = list(A)
                best_B = list(B)

    # Final check
    if not best_A or not best_B:
        return [[1], [2]]

    return [best_A, best_B]
