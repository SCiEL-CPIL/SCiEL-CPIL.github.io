import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    N = 1000

    def build_counts(S):
        """Build sum-count array from set S."""
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            a = S_sorted[i]
            counts[2 * a] += 1
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                counts[a + b] += 1
        return counts

    def can_add_fast(S, counts, x):
        """Check if x can be added to S without violating B2[2]."""
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if counts[x + y] >= 2:
                return False
        return True

    def do_add(S, counts, x):
        """Add x to S and update counts."""
        S.add(x)
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1

    def do_remove(S, counts, x):
        """Remove x from S and update counts."""
        S.discard(x)
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    def greedy_fill(S, counts, order):
        """Fill S greedily in given order."""
        for x in order:
            if x not in S and can_add_fast(S, counts, x):
                do_add(S, counts, x)

    def local_search(S, counts, time_limit):
        """Local search: try swaps and adds to improve size."""
        while time.time() < time_limit:
            S_list = sorted(S)
            n = len(S_list)

            # Try to add a random element
            added = False
            for _ in range(200):
                x = random.randint(1, N)
                if x not in S and can_add_fast(S, counts, x):
                    do_add(S, counts, x)
                    added = True
                    break

            if added:
                continue

            # Try a swap: remove one element, try to add a different one
            if n > 0:
                # Pick a random element to remove (prefer those causing many collisions)
                # Compute "pressure" for each element
                pressures = {}
                for x in S_list:
                    p = 0
                    if counts[2 * x] >= 2:
                        p += 10
                    for y in S:
                        if y != x and counts[x + y] >= 2:
                            p += 1
                    pressures[x] = p

                # Remove the highest-pressure element (with some randomness)
                max_p = max(pressures.values()) if pressures else 0
                candidates_to_remove = [x for x in S_list if pressures[x] >= max_p * 0.7]
                if not candidates_to_remove:
                    candidates_to_remove = S_list
                x_remove = random.choice(candidates_to_remove)

                do_remove(S, counts, x_remove)

                # Now try to add a new element
                added_swap = False
                for _ in range(300):
                    x_new = random.randint(1, N)
                    if x_new not in S and can_add_fast(S, counts, x_new):
                        do_add(S, counts, x_new)
                        added_swap = True
                        break

                if not added_swap:
                    # Re-add the removed element to restore
                    do_add(S, counts, x_remove)

    best_set = set()
    best_size = 0

    # Strategy 1: Ascending greedy
    S1 = set()
    c1 = np.zeros(2 * N + 1, dtype=np.int32)
    greedy_fill(S1, c1, range(1, N + 1))
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = set(S1)

    # Strategy 2: Descending greedy
    S2 = set()
    c2 = np.zeros(2 * N + 1, dtype=np.int32)
    greedy_fill(S2, c2, range(N, 0, -1))
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = set(S2)

    # Strategy 3: Random order greedy (multiple trials)
    for trial in range(3):
        S3 = set()
        c3 = np.zeros(2 * N + 1, dtype=np.int32)
        order = list(range(1, N + 1))
        random.shuffle(order)
        greedy_fill(S3, c3, order)
        if len(S3) > best_size:
            best_size = len(S3)
            best_set = set(S3)

    # Strategy 4: Dense random init then local search
    S4 = set()
    c4 = np.zeros(2 * N + 1, dtype=np.int32)
    # Start with a random subset of ~700 elements, then prune
    init_candidates = list(range(1, N + 1))
    random.shuffle(init_candidates)
    for x in init_candidates:
        if len(S4) >= 700:
            break
        if can_add_fast(S4, c4, x):
            do_add(S4, c4, x)
    # Now try to fill remaining capacity
    for x in init_candidates:
        if x not in S4 and can_add_fast(S4, c4, x):
            do_add(S4, c4, x)
    if len(S4) > best_size:
        best_size = len(S4)
        best_set = set(S4)

    # Local search on the best solution found
    if time.time() < deadline - 10:
        S_ls = set(best_set)
        c_ls = build_counts(S_ls)
        local_search(S_ls, c_ls, deadline - 5)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = set(S_ls)

    # If time permits, try local search on other good solutions too
    remaining = deadline - time.time()
    if remaining > 15 and len(S1) >= best_size - 5:
        S_ls2 = set(S1)
        c_ls2 = build_counts(S_ls2)
        local_search(S_ls2, c_ls2, min(deadline - 5, time.time() + remaining / 3))
        if len(S_ls2) > best_size:
            best_size = len(S_ls2)
            best_set = set(S_ls2)

    if time.time() < deadline - 5:
        S_ls3 = set(S2)
        c_ls3 = build_counts(S_ls3)
        local_search(S_ls3, c_ls3, min(deadline - 5, time.time() + 10))
        if len(S_ls3) > best_size:
            best_size = len(S_ls3)
            best_set = set(S_ls3)

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
