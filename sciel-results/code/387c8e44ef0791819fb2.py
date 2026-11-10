"""Combined approach: greedy collision-maximizing construction + fast exact B&B verification."""
import time
import random

_START_TIME = time.time()
_DEADLINE = 85.0  # seconds, safety margin before 90s kill


def _is_sidon_with_diagonals(subset):
    """Check if subset has all unordered pair sums unique including diagonals (2a[i])."""
    n = len(subset)
    if n <= 1:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _max_sidon_subset_size_exact(arr):
    """
    Exact branch-and-bound to find the largest Sidon subset size.
    Uses sorted arr and incremental sum-set tracking for speed.
    """
    n = len(arr)
    if n <= 1:
        return n
    if _is_sidon_with_diagonals(arr):
        return n

    arr_sorted = sorted(arr)
    best = 0

    def dfs(idx, current_size, current_sums, current_elements):
        nonlocal best

        # Pruning: if we can't beat best
        if current_size + (n - idx) <= best:
            return

        # Update best
        if current_size > best:
            best = current_size
            if best == n:
                return

        if idx == n:
            return

        # Option 1: Exclude arr_sorted[idx]
        dfs(idx + 1, current_size, current_sums, current_elements)

        # Option 2: Include arr_sorted[idx]
        x = arr_sorted[idx]
        can_add = True
        new_sums = set()

        for y in current_elements:
            s = x + y
            if s in current_sums:
                can_add = False
                break
            new_sums.add(s)

        if can_add:
            s_diag = 2 * x
            if s_diag in current_sums:
                can_add = False
            else:
                new_sums.add(s_diag)

        if can_add:
            dfs(idx + 1, current_size + 1, current_sums | new_sums, current_elements + [x])

    dfs(0, 0, set(), [])
    return best


def _count_collision_sums(candidate, existing_set):
    """
    Count how many pair sums involving 'candidate' and elements in 'existing_set'
    already appear as pair sums within 'existing_set' itself.
    This measures how much adding 'candidate' violates Sidon property.
    """
    # Get all pair sums (including diagonals) of existing_set
    existing_sums = set()
    n = len(existing_set)
    for i in range(n):
        for j in range(i, n):
            existing_sums.add(existing_set[i] + existing_set[j])

    # Count collisions: sums involving candidate that are already in existing_sums
    collisions = 0
    for x in existing_set:
        s = candidate + x
        if s in existing_sums:
            collisions += 1
    s_diag = 2 * candidate
    if s_diag in existing_sums:
        collisions += 1

    return collisions


def _greedy_collision_build(rng, base_offset=0):
    """
    Greedily build a set of 12 integers in [0,128] by always choosing the next
    integer that maximizes immediate pair-sum collisions with previously chosen elements.
    This creates a dense collision structure, making it hard to extract large Sidon subsets.
    """
    chosen = []
    used = set()

    for step in range(12):
        best_val = None
        best_collisions = -1

        # Candidate range: focus on a window to create dense collisions
        # Use a sliding window approach with some randomness
        window_start = max(0, base_offset + step * 2 - 10)
        window_end = min(128, base_offset + step * 2 + 15)

        # Generate candidates in the window
        candidates = list(range(window_start, window_end + 1))
        # Add some random candidates from the full range for diversity
        for _ in range(20):
            candidates.append(rng.randint(0, 128))

        # Remove already used
        candidates = [c for c in candidates if c not in used]

        if not candidates:
            # Fallback: pick any unused value
            for v in range(0, 129):
                if v not in used:
                    candidates.append(v)
            candidates = [c for c in candidates if c not in used]

        if not candidates:
            # Should not happen, but safety
            for v in range(0, 129):
                if v not in used:
                    used.add(v)
                    chosen.append(v)
                    break
            continue

        # For each candidate, compute collision count with existing set
        for cand in candidates:
            if not chosen:
                # First element: no collisions possible, pick smallest
                coll = 0
            else:
                coll = _count_collision_sums(cand, chosen)

            if coll > best_collisions:
                best_collisions = coll
                best_val = cand
            elif coll == best_collisions:
                # Tie-break: prefer smaller values to keep density
                if best_val is None or cand < best_val:
                    best_val = cand

        if best_val is not None:
            chosen.append(best_val)
            used.add(best_val)
        else:
            # Safety fallback
            for v in range(0, 129):
                if v not in used:
                    chosen.append(v)
                    used.add(v)
                    break

    return chosen


def _local_refine(arr, rng, time_limit=5.0):
    """
    Local refinement: try swapping individual elements to reduce the max Sidon subset size.
    Uses the exact evaluator for verification when time permits.
    """
    start_time = time.time()
    current = arr[:]
    current_score = _max_sidon_subset_size_exact(current)

    if current_score <= 3:
        return current  # Already very good

    # Try single-element replacements
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(12):
            old_val = current[i]
            # Try a few random replacements
            for _ in range(5):
                new_val = rng.randint(0, 128)
                if new_val == old_val:
                    continue
                if new_val in current[:i] or new_val in current[i+1:]:
                    continue

                new_arr = current[:]
                new_arr[i] = new_val

                # Quick check: is it still valid?
                if len(set(new_arr)) != 12:
                    continue

                # Evaluate with exact solver
                new_score = _max_sidon_subset_size_exact(new_arr)

                if new_score < current_score:
                    current = new_arr
                    current_score = new_score
                    improved = True
                    break

            if improved:
                break

        if not improved:
            # Try a random restart with local search
            if time.time() - start_time > time_limit * 0.5:
                break

    return current


def construct():
    global _START_TIME, _DEADLINE

    rng = random.Random(42)

    # Phase 1: Generate multiple candidate sets using greedy collision build
    candidates = []

    # Different base offsets and strategies
    for base in [0, 10, 20, 30, 50, 64]:
        for _ in range(3):
            arr = _greedy_collision_build(rng, base_offset=base)
            if len(set(arr)) == 12 and all(0 <= x <= 128 for x in arr):
                candidates.append(arr)

    # Also add some structured candidates
    structured = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121],
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 128],
    ]
    for s in structured:
        if len(set(s)) == 12 and all(0 <= x <= 128 for x in s):
            candidates.append(s)

    # Phase 2: Evaluate all candidates and pick the best
    best_arr = candidates[0]
    best_score = _max_sidon_subset_size_exact(best_arr)

    for arr in candidates[1:]:
        if time.time() > _DEADLINE - 10.0:
            break
        score = _max_sidon_subset_size_exact(arr)
        if score < best_score:
            best_score = score
            best_arr = arr[:]

    # Phase 3: Local refinement on the best candidate
    time_remaining = _DEADLINE - time.time() - 2.0
    if time_remaining > 3.0:
        best_arr = _local_refine(best_arr, rng, time_limit=min(time_remaining, 15.0))

    # Final validation
    assert len(best_arr) == 12
    assert len(set(best_arr)) == 12
    assert all(0 <= x <= 128 for x in best_arr)

    return best_arr


# EVOLVE-BLOCK-START
# (construct is defined above; this block is for SCiEL evolution markers)
# EVOLVE-BLOCK-END
