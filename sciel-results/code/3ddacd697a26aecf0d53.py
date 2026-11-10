import random
import time
import numpy as np


def _greedy_build(N, order, deadline):
    """Build a B2[2] set using greedy with given order."""
    S = set()
    counts = np.zeros(2 * N + 1, dtype=np.int32)
    S_arr = np.array([], dtype=np.int32)

    for x in order:
        if time.time() > deadline:
            break
        # Check if x can be added
        valid = True
        if counts[2 * x] >= 2:
            valid = False
        elif len(S_arr) > 0:
            sums = x + S_arr
            if np.any(counts[sums] >= 2):
                valid = False
        if valid:
            S.add(x)
            counts[2 * x] += 1
            if len(S_arr) > 0:
                sums = x + S_arr
                counts[sums] += 1
            S_arr = np.append(S_arr, x)

    return S, counts, S_arr


def _local_search(S, counts, S_arr, N, deadline, seed=42):
    """Perform local search to improve the B2[2] set using vectorized checks."""
    random.seed(seed)
    current_S = set(S)
    S_arr = np.array(sorted(S_arr), dtype=np.int32)
    current_counts = counts.copy()
    best_S = frozenset(current_S)
    best_size = len(current_S)

    # Track which elements are in S for O(1) membership
    in_S = np.zeros(N + 1, dtype=bool)
    for x in current_S:
        in_S[x] = True

    it = 0
    while time.time() < deadline:
        it += 1

        # Try to add a random element from candidates
        available_mask = ~in_S
        available_mask[0] = False
        available_indices = np.where(available_mask)[0]
        if len(available_indices) == 0:
            break

        y = int(random.choice(available_indices))

        # Check if y can be added: 2*y and y+z for z in S
        valid = True
        if current_counts[2 * y] >= 2:
            valid = False
        elif len(S_arr) > 0:
            sums = y + S_arr
            if np.any(current_counts[sums] >= 2):
                valid = False

        if valid:
            # Accept add
            current_counts[2 * y] += 1
            if len(S_arr) > 0:
                sums = y + S_arr
                current_counts[sums] += 1
            current_S.add(y)
            in_S[y] = True
            S_arr = np.append(S_arr, y)

            if len(current_S) > best_size:
                best_S = frozenset(current_S)
                best_size = len(current_S)
        else:
            # Try swap: remove a random element x, then add y
            if len(current_S) > 0:
                x = int(random.choice(S_arr))

                # Remove x from current state
                current_counts[2 * x] -= 1
                if len(S_arr) > 0:
                    mask = S_arr != x
                    if np.any(mask):
                        sums = x + S_arr[mask]
                        current_counts[sums] -= 1
                current_S.remove(x)
                in_S[x] = False
                S_arr = S_arr[S_arr != x]

                # Now try to add y
                valid2 = True
                if current_counts[2 * y] >= 2:
                    valid2 = False
                elif len(S_arr) > 0:
                    sums = y + S_arr
                    if np.any(current_counts[sums] >= 2):
                        valid2 = False

                if valid2:
                    # Accept swap
                    current_counts[2 * y] += 1
                    if len(S_arr) > 0:
                        sums = y + S_arr
                        current_counts[sums] += 1
                    current_S.add(y)
                    in_S[y] = True
                    S_arr = np.append(S_arr, y)

                    if len(current_S) > best_size:
                        best_S = frozenset(current_S)
                        best_size = len(current_S)
                else:
                    # Rollback swap: put x back
                    current_counts[2 * x] += 1
                    if len(S_arr) > 0:
                        sums = x + S_arr
                        current_counts[sums] += 1
                    current_S.add(x)
                    in_S[x] = True
                    S_arr = np.append(S_arr, x)

    return set(best_S)


def _two_opt_search(S, counts, S_arr, N, deadline, seed=42):
    """Try 2-opt moves: remove two elements, add two different ones."""
    random.seed(seed)
    current_S = set(S)
    S_arr = np.array(sorted(S_arr), dtype=np.int32)
    current_counts = counts.copy()
    best_S = frozenset(current_S)
    best_size = len(current_S)

    in_S = np.zeros(N + 1, dtype=bool)
    for x in current_S:
        in_S[x] = True

    it = 0
    while time.time() < deadline:
        it += 1
        if len(S_arr) < 2:
            break

        # Pick two elements to remove
        idx1, idx2 = random.sample(range(len(S_arr)), 2)
        x1 = int(S_arr[idx1])
        x2 = int(S_arr[idx2])

        # Remove both
        current_counts[2 * x1] -= 1
        current_counts[2 * x2] -= 1
        if x1 != x2:
            current_counts[x1 + x2] -= 1
        # Remove sums with other elements
        if len(S_arr) > 2:
            mask = (S_arr != x1) & (S_arr != x2)
            if np.any(mask):
                other = S_arr[mask]
                current_counts[x1 + other] -= 1
                current_counts[x2 + other] -= 1

        current_S.remove(x1)
        current_S.remove(x2)
        in_S[x1] = False
        in_S[x2] = False
        S_arr = S_arr[(S_arr != x1) & (S_arr != x2)]

        # Try to add two new elements
        available_mask = ~in_S
        available_mask[0] = False
        available_indices = np.where(available_mask)[0]

        if len(available_indices) < 2:
            # Rollback
            current_S.add(x1)
            current_S.add(x2)
            in_S[x1] = True
            in_S[x2] = True
            S_arr = np.append(S_arr, np.array([x1, x2]))
            current_counts[2 * x1] += 1
            current_counts[2 * x2] += 1
            if x1 != x2:
                current_counts[x1 + x2] += 1
            if len(S_arr) > 2:
                mask = (S_arr != x1) & (S_arr != x2)
                if np.any(mask):
                    other = S_arr[mask]
                    current_counts[x1 + other] += 1
                    current_counts[x2 + other] += 1
            break

        # Pick two candidates to add
        y1, y2 = random.sample(available_indices.tolist(), 2)
        y1 = int(y1)
        y2 = int(y2)

        # Check if both can be added
        valid = True
        if current_counts[2 * y1] >= 2 or current_counts[2 * y2] >= 2:
            valid = False
        elif y1 != y2 and current_counts[y1 + y2] >= 2:
            valid = False
        elif len(S_arr) > 0:
            sums1 = y1 + S_arr
            sums2 = y2 + S_arr
            if np.any(current_counts[sums1] >= 2) or np.any(current_counts[sums2] >= 2):
                valid = False

        if valid:
            # Accept
            current_counts[2 * y1] += 1
            current_counts[2 * y2] += 1
            if y1 != y2:
                current_counts[y1 + y2] += 1
            if len(S_arr) > 0:
                current_counts[y1 + S_arr] += 1
                current_counts[y2 + S_arr] += 1
            current_S.add(y1)
            current_S.add(y2)
            in_S[y1] = True
            in_S[y2] = True
            S_arr = np.append(S_arr, np.array([y1, y2]))

            if len(current_S) > best_size:
                best_S = frozenset(current_S)
                best_size = len(current_S)
        else:
            # Rollback
            current_S.add(x1)
            current_S.add(x2)
            in_S[x1] = True
            in_S[x2] = True
            S_arr = np.append(S_arr, np.array([x1, x2]))
            current_counts[2 * x1] += 1
            current_counts[2 * x2] += 1
            if x1 != x2:
                current_counts[x1 + x2] += 1
            if len(S_arr) > 2:
                mask = (S_arr != x1) & (S_arr != x2)
                if np.any(mask):
                    other = S_arr[mask]
                    current_counts[x1 + other] += 1
                    current_counts[x2 + other] += 1

    return set(best_S)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    N = 1000

    best_S = set()
    best_size = 0
    best_counts = None
    best_S_arr = None

    # Try multiple greedy orderings
    orderings = [
        range(1, N + 1),  # ascending
        range(N, 0, -1),  # descending
        list(range(1, N + 1, 2)),  # odd first
        list(range(2, N + 1, 2)),  # even first
        list(range(1, N + 1, 3)),  # every 3rd
        list(range(1, N + 1, 4)),  # every 4th
        list(range(1, N + 1, 5)),  # every 5th
        list(range(2, N + 1, 3)),  # 2 mod 3
    ]

    for i, order in enumerate(orderings):
        if time.time() > deadline:
            break

        S, counts, S_arr = _greedy_build(N, order, deadline)

        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            best_counts = counts
            best_S_arr = S_arr

        # Local search on this solution
        if time.time() < deadline - 20:
            improved_S = _local_search(S, counts, S_arr, N, deadline, seed=42 + i)
            if len(improved_S) > best_size:
                best_S = improved_S
                best_size = len(improved_S)
                # Rebuild counts and S_arr
                S_list = sorted(best_S)
                S_arr_new = np.array(S_list, dtype=np.int32)
                counts_new = np.zeros(2 * N + 1, dtype=np.int32)
                for x in best_S:
                    counts_new[2 * x] += 1
                for idx in range(len(S_list)):
                    for jdx in range(idx + 1, len(S_list)):
                        counts_new[S_list[idx] + S_list[jdx]] += 1
                best_counts = counts_new
                best_S_arr = S_arr_new

    # Final 2-opt search on best solution found
    if time.time() < deadline - 10 and best_S:
        two_opt_S = _two_opt_search(best_S, best_counts, best_S_arr, N, deadline, seed=100)
        if len(two_opt_S) > best_size:
            best_S = two_opt_S
            best_size = len(best_S)

    # Final local search
    if time.time() < deadline - 5 and best_S:
        # Rebuild counts for final search
        S_list = sorted(best_S)
        S_arr_final = np.array(S_list, dtype=np.int32)
        counts_final = np.zeros(2 * N + 1, dtype=np.int32)
        for x in best_S:
            counts_final[2 * x] += 1
        for idx in range(len(S_list)):
            for jdx in range(idx + 1, len(S_list)):
                counts_final[S_list[idx] + S_list[jdx]] += 1

        final_S = _local_search(best_S, counts_final, S_arr_final, N, deadline, seed=200)
        if len(final_S) > best_size:
            best_S = final_S
            best_size = len(best_S)

    return list(best_S)
