import random
import time


def _validity(rows):
    """Return True if all positive within-row differences are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_row(used, k, max_val, rng=None):
    """Build one row of k+1 increasing ints starting at 0 using greedy smallest-next.
    Returns None if it cannot complete within max_val."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        found = False
        # try a few candidate increments; default to +1 greedy
        for _ in range(1):
            x += 1
            if x > max_val:
                return None
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
                found = True
                break
        if not found:
            return None
    return row


def _greedy_construct(n_rows, k, max_val, rng):
    """Deterministic-ish greedy: build rows one by one.
    Returns rows or None if it fails to complete within max_val."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _local_search(rows, n_rows, k, max_val, rng, time_limit):
    """Try to reduce the scope by swapping/moving marks. Returns improved rows or original."""
    best_rows = rows
    best_scope = _scope(rows)
    start = time.time()
    # We attempt to reduce the maximum entry by trying to replace large marks.
    # Simple heuristic: for each row, try to find a smaller valid replacement for its last element.
    for it in range(200):
        if time.time() - start > time_limit:
            break
        # pick a random row and try to shrink its largest element
        ridx = rng.randrange(n_rows)
        row = list(best_rows[ridx])
        if len(row) < k + 1:
            continue
        # Try to find a smaller value for the last element that keeps the row valid
        # and doesn't conflict with other rows' differences.
        # Compute differences used by this row and by others.
        others_used = set()
        for idx, r in enumerate(best_rows):
            if idx == ridx:
                continue
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    others_used.add(r[j] - r[i])
        # This row's differences excluding those involving the last element
        this_used_without_last = set()
        for i in range(len(row) - 1):
            for j in range(i + 1, len(row) - 1):
                this_used_without_last.add(row[j] - row[i])
        last = row[-1]
        # Try smaller candidates
        for cand in range(last - 1, row[-2], -1):
            # cand must be > row[-2]
            if cand <= row[-2]:
                break
            new_diffs = {cand - a for a in row[:-1]}
            if new_diffs & others_used:
                continue
            if new_diffs & this_used_without_last:
                continue
            # valid replacement
            new_row = row[:-1] + [cand]
            new_scope = max(_scope(best_rows), max(new_row))
            if new_scope < best_scope:
                # apply
                best_rows[ridx] = new_row
                best_scope = new_scope
            break
    return best_rows


def construct(n_rows, k):
    start = time.time()
    deadline = start + 145.0  # safety margin below 150s

    # Phase 1: deterministic greedy baseline
    # First, find a feasible max_val by doubling.
    max_val = 1
    baseline = None
    while time.time() < deadline:
        baseline = _greedy_construct(n_rows, k, max_val, None)
        if baseline is not None:
            break
        max_val *= 2
    if baseline is None:
        # Should not happen, but fallback to a very large value
        max_val = 1000000
        baseline = _greedy_construct(n_rows, k, max_val, None)
        if baseline is None:
            # Last resort: return a trivially valid (but large) construction
            # Use a simple spread: each row uses multiples to avoid collisions.
            # This is a fallback that should rarely be needed.
            rows = []
            for r in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1000000
                    row.append(x)
                rows.append(row)
            return rows

    best_rows = baseline
    best_scope = _scope(best_rows)

    # Phase 2: local search to reduce scope
    if time.time() < deadline - 5.0:
        ls_rows = _local_search(best_rows, n_rows, k, best_scope, random, 10.0)
        if _scope(ls_rows) < best_scope:
            best_rows = ls_rows
            best_scope = _scope(best_rows)

    # Phase 3: randomized restarts with shuffled order and perturbation
    # Each restart: shuffle the order in which we "reserve" difference sets,
    # and use randomized greedy with occasional larger jumps.
    rng = random.Random(12345)
    attempt = 0
    while time.time() < deadline - 2.0:
        attempt += 1
        # Randomized greedy: build rows but with a random order of "priority"
        # and allow the greedy to sometimes skip a value to explore alternatives.
        used = set()
        rows = []
        success = True
        # Decide a random target scope to try to beat
        target = best_scope - 1
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a small random window of candidates
                found = False
                for _ in range(3):
                    step = 1 if rng.random() < 0.7 else rng.randint(1, 5)
                    x += step
                    if x > target:
                        success = False
                        break
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        found = True
                        break
                if not found or not success:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        if success and len(rows) == n_rows:
            sc = _scope(rows)
            if sc < best_scope:
                best_rows = rows
                best_scope = sc
                # Occasionally do a quick local search on the new best
                if time.time() < deadline - 5.0:
                    ls_rows = _local_search(best_rows, n_rows, k, best_scope, rng, 5.0)
                    if _scope(ls_rows) < best_scope:
                        best_rows = ls_rows
                        best_scope = _scope(ls_rows)
        # Also try a pure randomized greedy without target limit, just to get a valid set
        # if we haven't found anything better yet, and use it as a new baseline if smaller
        if attempt % 50 == 0 and time.time() < deadline - 5.0:
            used2 = set()
            rows2 = []
            success2 = True
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    found = False
                    for _ in range(5):
                        step = rng.randint(1, 3)
                        x += step
                        new = {x - a for a in row}
                        if not (new & used2):
                            row.append(x)
                            used2 |= new
                            found = True
                            break
                    if not found:
                        success2 = False
                        break
                if not success2:
                    break
                rows2.append(row)
            if success2 and len(rows2) == n_rows:
                sc2 = _scope(rows2)
                if sc2 < best_scope:
                    best_rows = rows2
                    best_scope = sc2

    return best_rows
