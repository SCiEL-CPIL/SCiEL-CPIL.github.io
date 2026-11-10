"""B2[2] set construction for {1..1000}: vectorized greedy + conflict-driven local search."""
import time
import random
import numpy as np

N = 1000
MAX_REP = 2


def _greedy_build(order, deadline):
    """Greedy build using vectorized NumPy validity checks."""
    S = []
    in_S = np.zeros(N + 1, dtype=bool)
    counts = np.zeros(2 * N + 1, dtype=np.int32)

    for x in order:
        if time.time() > deadline:
            break
        if in_S[x]:
            continue

        # Check: 2*x
        if counts[2 * x] >= MAX_REP:
            continue

        # Check: x + y for all y in S (vectorized)
        if S:
            S_arr = np.array(S, dtype=np.int32)
            sums = x + S_arr
            if np.any(counts[sums] >= MAX_REP):
                continue

        # Accept x
        in_S[x] = True
        S.append(x)
        counts[2 * x] += 1
        if len(S) > 1:
            S_arr = np.array(S[:-1], dtype=np.int32)
            counts[x + S_arr] += 1

    return S, counts


def _rebuild_counts(S):
    """Rebuild the sum-count array from scratch."""
    counts = np.zeros(2 * N + 1, dtype=np.int32)
    for i, x in enumerate(S):
        counts[2 * x] += 1
        if i > 0:
            counts[x + np.array(S[:i], dtype=np.int32)] += 1
    return counts


def _can_add(x, S_arr, counts):
    """Vectorized check: can x be added given current S and counts?"""
    if counts[2 * x] >= MAX_REP:
        return False
    if len(S_arr) == 0:
        return True
    sums = x + S_arr
    if np.any(counts[sums] >= MAX_REP):
        return False
    return True


def _add_element(x, S, S_arr, in_S, counts):
    """Add x to the set, updating all structures."""
    S.append(x)
    in_S[x] = True
    counts[2 * x] += 1
    if len(S) > 1:
        old = np.array(S[:-1], dtype=np.int32)
        counts[x + old] += 1
    S_arr = np.append(S_arr, x)
    return S_arr


def _remove_element(x, S, S_arr, in_S, counts):
    """Remove x from the set, updating all structures."""
    S.remove(x)
    in_S[x] = False
    counts[2 * x] -= 1
    if len(S) > 0:
        remaining = np.array(S, dtype=np.int32)
        counts[x + remaining] -= 1
    S_arr = S_arr[S_arr != x]
    return S_arr


def _local_search(S, counts, deadline, seed=42):
    """Conflict-driven local search: insert when possible, swap when blocked."""
    rng = random.Random(seed)
    S = list(S)
    in_S = np.zeros(N + 1, dtype=bool)
    for x in S:
        in_S[x] = True
    S_arr = np.array(S, dtype=np.int32)
    counts = counts.copy()

    best_size = len(S)
    best_S = list(S)

    # Precompute all candidates not in S
    all_nums = np.arange(1, N + 1, dtype=np.int32)

    step = 0
    while time.time() < deadline:
        step += 1
        if step % 2000 == 0 and time.time() > deadline:
            break

        # --- Phase 1: try insertions ---
        # Find elements not in S that can be added
        not_in_S = ~in_S
        not_in_S[0] = False
        candidates = np.where(not_in_S)[0]

        if len(candidates) > 0:
            # Try a random sample of candidates for insertion
            sample_size = min(len(candidates), 50)
            sample_idx = rng.sample(range(len(candidates)), sample_size)
            found_insert = False
            for idx in sample_idx:
                if time.time() > deadline:
                    break
                y = int(candidates[idx])
                if _can_add(y, S_arr, counts):
                    S_arr = _add_element(y, S, S_arr, in_S, counts)
                    found_insert = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
            if found_insert:
                continue  # keep trying insertions

        # --- Phase 2: conflict-driven swaps ---
        # Find elements in S that block the most insertions, remove one, try to add
        if len(S) == 0:
            break

        # Score each element in S by how many candidates it blocks
        # (approximate: count of candidates y where y+x or 2x conflicts)
        # For speed, just pick a random element to remove
        x = int(rng.choice(S_arr))

        # Remove x
        S_arr = _remove_element(x, S, S_arr, in_S, counts)

        # Try to add a random candidate
        not_in_S = ~in_S
        not_in_S[0] = False
        candidates = np.where(not_in_S)[0]

        if len(candidates) == 0:
            # Put x back
            S_arr = _add_element(x, S, S_arr, in_S, counts)
            continue

        # Try several candidates
        sample_size = min(len(candidates), 30)
        sample_idx = rng.sample(range(len(candidates)), sample_size)
        added = False
        for idx in sample_idx:
            if time.time() > deadline:
                break
            y = int(candidates[idx])
            if _can_add(y, S_arr, counts):
                S_arr = _add_element(y, S, S_arr, in_S, counts)
                added = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                break

        if not added:
            # Rollback: put x back
            S_arr = _add_element(x, S, S_arr, in_S, counts)

    return best_S


def _targeted_local_search(S, counts, deadline, seed=0):
    """More targeted: repeatedly find the 'most blocking' element and swap it out."""
    rng = random.Random(seed)
    S = list(S)
    in_S = np.zeros(N + 1, dtype=bool)
    for x in S:
        in_S[x] = True
    S_arr = np.array(S, dtype=np.int32)
    counts = counts.copy()

    best_size = len(S)
    best_S = list(S)

    while time.time() < deadline:
        if len(S) == 0:
            break

        # Compute for each x in S: how many y not in S would be addable if x were removed?
        # Approximate by checking a sample of candidates
        not_in_S = ~in_S
        not_in_S[0] = False
        candidates = np.where(not_in_S)[0]
        if len(candidates) == 0:
            break

        # Sample candidates to estimate conflict scores
        sample_size = min(len(candidates), 80)
        sample = np.array(rng.sample(list(candidates), sample_size), dtype=np.int32)

        # For each x in S, count how many sample candidates are blocked by x
        # x blocks y if: (2y count >= MAX_REP after removing x) OR (y+z count >= MAX_REP for some z)
        # Simplification: x blocks y if (x+y) sum is at capacity, or 2x conflicts
        # Even simpler: just try removing each x and see if any candidate becomes addable
        # This is O(|S| * sample) which is fine for |S| ~ 700, sample ~ 80

        best_x = None
        best_y = None
        best_gain = 0

        # Try a subset of S elements for removal
        if len(S) > 100:
            s_sample = rng.sample(S, min(len(S), 50))
        else:
            s_sample = S

        for x in s_sample:
            if time.time() > deadline:
                break
            # Temporarily remove x
            S.remove(x)
            in_S[x] = False
            counts[2 * x] -= 1
            if len(S) > 0:
                rem = np.array(S, dtype=np.int32)
                counts[x + rem] -= 1
            S_arr_tmp = S_arr[S_arr != x]

            # Try to find a candidate y that can now be added
            for y in sample:
                y = int(y)
                if in_S[y]:
                    continue
                if counts[2 * y] >= MAX_REP:
                    continue
                if len(S_arr_tmp) > 0:
                    sums = y + S_arr_tmp
                    if np.any(counts[sums] >= MAX_REP):
                        continue

                # Found a valid (x, y) swap
                best_x = x
                best_y = y
                best_gain = 1
                break

            if best_x is not None:
                # Perform the swap
                if len(S_arr_tmp) > 0:
                    counts[best_y + S_arr_tmp] += 1
                counts[2 * best_y] += 1
                S.append(best_y)
                in_S[best_y] = True
                S_arr = np.append(S_arr_tmp, best_y)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                break
            else:
                # Rollback: put x back
                if len(S) > 0:
                    rem = np.array(S, dtype=np.int32)
                    counts[x + rem] += 1
                counts[2 * x] += 1
                S.append(x)
                in_S[x] = True
                S_arr = np.append(S_arr_tmp, x)

    return best_S


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    best_S = []
    best_size = 0

    # --- Phase 1: Multiple greedy builds with different orderings ---
    rng = random.Random(42)

    orderings = []
    # Ascending
    orderings.append(list(range(1, N + 1)))
    # Descending
    orderings.append(list(range(N, 0, -1)))
    # Odds first, then evens
    orderings.append(list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)))
    # Evens first, then odds
    orderings.append(list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)))
    # Random shuffles
    for _ in range(8):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        orderings.append(order)

    for i, order in enumerate(orderings):
        if time.time() > deadline - 20:
            break
        S, counts = _greedy_build(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

        # Quick local search on this solution
        if time.time() < deadline - 15:
            improved = _local_search(S, counts, deadline - 10, seed=42 + i)
            if len(improved) > best_size:
                best_size = len(improved)
                best_S = list(improved)

    # --- Phase 2: Targeted local search on the best solution ---
    if best_S and time.time() < deadline - 10:
        counts = _rebuild_counts(best_S)
        improved = _targeted_local_search(best_S, counts, deadline - 5, seed=100)
        if len(improved) > best_size:
            best_size = len(improved)
            best_S = list(improved)

    # --- Phase 3: Final general local search ---
    if best_S and time.time() < deadline - 3:
        counts = _rebuild_counts(best_S)
        improved = _local_search(best_S, counts, deadline - 1, seed=999)
        if len(improved) > best_size:
            best_size = len(improved)
            best_S = list(improved)

    return sorted(best_S)
