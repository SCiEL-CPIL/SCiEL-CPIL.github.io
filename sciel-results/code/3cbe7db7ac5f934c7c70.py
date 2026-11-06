"""Small maximal Sidon set constructor for [1,256].

Strategy:
- Use a bitmask-based representation for the set of used pair sums.
- Greedy construction with a blocking heuristic: at each step, pick the addable
  element that blocks the most remaining candidates.
- Removal-based local search to shrink the set while preserving maximality.
- Multiple random restarts with different orderings.
- Deadline guard: return best found before 55 s.
"""
import time
import random
import sys

N = 256
MAX_SUM = 2 * N          # 512
# sums range from 2 to 512 inclusive -> 511 distinct values; index = s - 2
NUM_SUMS = MAX_SUM - 1   # 511


def _compute_sums(arr):
    """Return set of all unordered pair sums a+b with a<=b for arr."""
    sums = set()
    n = len(arr)
    for i in range(n):
        a = arr[i]
        for j in range(i, n):
            sums.add(a + arr[j])
    return sums


def _is_maximal(arr, sums_set):
    """Check inclusion-maximality: no element in [1,N]\arr can be added while preserving Sidon."""
    arr_set = set(arr)
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        # Check if x can be added: 2*x not in sums, and x+a not in sums for all a in arr
        if 2 * x in sums_set:
            continue
        ok = True
        for a in arr:
            if x + a in sums_set:
                ok = False
                break
        if ok:
            return False  # x can be added -> not maximal
    return True


def _greedy_blocking(deadline, rng):
    """
    Greedy construction: at each step, among all addable elements,
    pick the one that blocks the most remaining candidates.
    Returns (selected_list, sum_mask).
    """
    selected = []
    sum_mask = 0  # bit i set iff sum (i+2) is used

    remaining = set(range(1, N + 1))

    while remaining:
        if time.time() > deadline:
            break

        # Find all addable elements
        addable = []
        for x in remaining:
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue
            # Check x+a for all a in selected
            can = True
            for a in selected:
                idx = x + a - 2
                if sum_mask & (1 << idx):
                    can = False
                    break
            if can:
                addable.append(x)

        if not addable:
            break

        # Score each addable element by how many remaining elements it would block
        # y is blocked by x if after adding x, y can no longer be added.
        # y can be added iff 2*y not in sums AND y+a not in sums for all a in set.
        # After adding x, new sums are 2*x and x+a for all a in selected.
        # So y is blocked if:
        #   2*y in (sum_mask | {2*x} | {x+a : a in selected})
        #   OR there exists a in (selected + [x]) such that y+a in (sum_mask | {2*x} | {x+a' : a' in selected})
        
        # For each candidate x in addable, count how many y in remaining\{x} are blocked.
        best_x = None
        best_score = -1
        
        for x in addable:
            # Compute the set of new sums that would be added
            new_sums = set()
            new_sums.add(2 * x)
            for a in selected:
                new_sums.add(x + a)
            
            # Count how many y in remaining\{x} are blocked
            score = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y is currently addable (it is, since y is in addable or not)
                # Actually, y might not be in addable. We want to count y that are
                # currently addable but become non-addable after adding x.
                # y is currently addable iff 2*y not in sum_mask and y+a not in sum_mask for all a in selected.
                # Let's check if y is currently addable.
                currently_addable = True
                idx_2y = 2 * y - 2
                if sum_mask & (1 << idx_2y):
                    currently_addable = False
                else:
                    for a in selected:
                        idx = y + a - 2
                        if sum_mask & (1 << idx):
                            currently_addable = False
                            break
                
                if not currently_addable:
                    continue
                
                # Check if y would be blocked after adding x
                blocked = False
                # Check 2*y in new sums
                if 2 * y in new_sums:
                    blocked = True
                else:
                    # Check y+a in new sums for some a in selected + [x]
                    for a in selected:
                        if y + a in new_sums:
                            blocked = True
                            break
                    if not blocked:
                        if y + x in new_sums:
                            blocked = True
                
                if blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            best_x = rng.choice(addable)
        
        x = best_x
        remaining.discard(x)
        selected.append(x)
        sum_mask |= (1 << (2 * x - 2))
        for a in selected[:-1]:
            sum_mask |= (1 << (x + a - 2))

    return selected


def _greedy_sequential(order, deadline):
    """
    Greedy construction following a given order.
    Returns (selected_list, sum_mask).
    """
    selected = []
    sum_mask = 0

    for x in order:
        if time.time() > deadline:
            break
        idx_2x = 2 * x - 2
        if sum_mask & (1 << idx_2x):
            continue
        can = True
        for a in selected:
            idx = x + a - 2
            if sum_mask & (1 << idx):
                can = False
                break
        if can:
            selected.append(x)
            sum_mask |= (1 << (2 * x - 2))
            for a in selected[:-1]:
                sum_mask |= (1 << (x + a - 2))

    return selected


def _try_remove(arr, sums_set, deadline):
    """
    Try to remove elements from arr to get a smaller maximal Sidon set.
    Returns the (possibly smaller) list.
    """
    current = arr[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            candidate = current[:i] + current[i + 1:]
            if len(candidate) == 0:
                continue
            # Check Sidon
            cand_sums = _compute_sums(candidate)
            # Verify Sidon: no duplicate sums
            n_c = len(candidate)
            expected_sums = n_c * (n_c + 1) // 2
            if len(cand_sums) != expected_sums:
                continue  # not Sidon
            # Check maximality
            if _is_maximal(candidate, cand_sums):
                current = candidate
                sums_set = cand_sums
                improved = True
                break
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin under 60s limit
    rng = random.Random(42)

    best_size = N + 1
    best_set = None

    # Phase 1: Blocking greedy with multiple restarts
    for attempt in range(100):
        if time.time() > deadline:
            break
        # Shuffle the order of remaining to introduce randomness
        # The blocking greedy itself is deterministic given the initial set,
        # but we can randomize by shuffling the order in which we consider
        # ties in the blocking score. For simplicity, let's just run it as is,
        # but maybe we can randomize the initial "remaining" set order.
        # Actually, the blocking greedy picks the best x each time, so it's deterministic.
        # To get different results, we can randomize the tie-breaking.
        # Let's modify the blocking greedy to accept a random tie-breaker.
        # For now, let's just run it and see if it gives a good result.
        selected = _greedy_blocking(deadline, rng)
        if not selected:
            continue
        sums_set = _compute_sums(selected)
        n = len(selected)
        expected = n * (n + 1) // 2
        if len(sums_set) != expected:
            continue  # not Sidon
        if _is_maximal(selected, sums_set):
            sz = len(selected)
            if sz < best_size:
                best_size = sz
                best_set = selected[:]
                if best_size <= 12:
                    break

    # Phase 2: Random sequential greedy with multiple restarts
    for attempt in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        selected = _greedy_sequential(order, deadline)
        if not selected:
            continue
        sums_set = _compute_sums(selected)
        n = len(selected)
        expected = n * (n + 1) // 2
        if len(sums_set) != expected:
            continue
        if _is_maximal(selected, sums_set):
            sz = len(selected)
            if sz < best_size:
                best_size = sz
                best_set = selected[:]
                if best_size <= 12:
                    break

    # Phase 3: Local search (removal) on best found
    if best_set is not None:
        current = best_set[:]
        sums_set = _compute_sums(current)
        current = _try_remove(current, sums_set, deadline)
        if len(current) < best_size:
            best_size = len(current)
            best_set = current

    # Fallback
    if best_set is None:
        selected = []
        sum_mask = 0
        for x in range(1, N + 1):
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue
            can = True
            for a in selected:
                idx = x + a - 2
                if sum_mask & (1 << idx):
                    can = False
                    break
            if can:
                selected.append(x)
                sum_mask |= (1 << (2 * x - 2))
                for a in selected[:-1]:
                    sum_mask |= (1 << (x + a - 2))
        return selected

    return best_set
