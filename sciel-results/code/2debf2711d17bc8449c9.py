"""Difference triangle set (4,7): ILS with kick perturbations and feasibility/optimization phases."""
import time
import random
from collections import Counter

_DEADLINE = 145.0  # safety margin under 150s wall-clock


def _row_diffs(row):
    """All positive differences a[j]-a[i] for i<j in one row."""
    d = []
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.append(row[j] - ai)
    return d


def _row_scope(row):
    return row[-1]


def _build_counter(rows):
    """Build a Counter of all differences across all rows."""
    c = Counter()
    for row in rows:
        for d in _row_diffs(row):
            c[d] += 1
    return c


def _conflicts_from_counter(c):
    """Count conflicts from a difference counter."""
    return sum(v - 1 for v in c.values() if v > 1)


def _total_conflicts(rows):
    return _conflicts_from_counter(_build_counter(rows))


def _greedy_row(used, k):
    """Build one row greedily given already-used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row


def _greedy_all(n_rows, k):
    """Initial greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
        used |= set(_row_diffs(rows[-1]))
    return rows


def _valid(rows):
    """True if all positive differences across rows are distinct."""
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _compute_delta_conflicts(row, idx, old_val, new_val, counter):
    """Compute conflict delta when changing row[idx] from old_val to new_val."""
    old_d = []
    for j in range(len(row)):
        if j != idx:
            if j < idx:
                old_d.append(old_val - row[j])
            else:
                old_d.append(row[j] - old_val)
    new_d = []
    for j in range(len(row)):
        if j != idx:
            if j < idx:
                new_d.append(new_val - row[j])
            else:
                new_d.append(row[j] - new_val)

    delta_conf = 0
    # Remove old diffs
    for d in old_d:
        c = counter[d]
        if c >= 2:
            delta_conf -= 1
        counter[d] = c - 1
    # Add new diffs
    for d in new_d:
        c = counter[d]
        if c >= 1:
            delta_conf += 1
        counter[d] = c + 1

    return delta_conf, old_d, new_d


def _revert_counter(counter, old_d, new_d):
    """Revert counter changes."""
    for d in new_d:
        counter[d] -= 1
    for d in old_d:
        counter[d] += 1


def _local_search_feasibility(rows, counter, cur_conf, max_steps, rng):
    """Phase 1: Reduce conflicts to zero (feasibility)."""
    cur_rows = [list(r) for r in rows]
    for step in range(max_steps):
        if cur_conf == 0:
            break
        # Pick a row involved in a conflict
        # Find diffs with count > 1
        conflict_diffs = [d for d, c in counter.items() if c > 1]
        if not conflict_diffs:
            break
        # Pick a random conflict diff
        target_d = rng.choice(conflict_diffs)
        # Find rows containing this diff
        affected_rows = []
        for ri, row in enumerate(cur_rows):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] - row[i] == target_d:
                        affected_rows.append((ri, i, j))
                        break
        if not affected_rows:
            continue
        # Pick a random affected row/position
        ri, i, j = rng.choice(affected_rows)
        row = cur_rows[ri]
        # Try to move one of the endpoints to eliminate the conflict
        for idx in [i, j]:
            old_val = row[idx]
            # Try small perturbations
            for delta in range(1, 8):
                for sign in [1, -1]:
                    new_val = old_val + sign * delta
                    # Check bounds
                    lo = row[idx - 1] + 1
                    hi = (row[idx + 1] - 1) if idx + 1 < len(row) else 10**9
                    if new_val < lo or new_val > hi:
                        continue
                    if new_val == old_val:
                        continue
                    # Compute delta conflicts
                    delta_conf, old_d, new_d = _compute_delta_conflicts(row, idx, old_val, new_val, counter)
                    if delta_conf < 0:
                        # Accept
                        row[idx] = new_val
                        cur_conf += delta_conf
                        break
                    else:
                        _revert_counter(counter, old_d, new_d)
            if cur_conf == 0:
                break
    return cur_rows, cur_conf


def _local_search_optimization(rows, counter, cur_conf, max_steps, rng):
    """Phase 2: Reduce scope while maintaining validity (conflicts == 0)."""
    cur_rows = [list(r) for r in rows]
    best_scope = max(_row_scope(r) for r in cur_rows)
    for step in range(max_steps):
        if cur_conf > 0:
            break
        # Try to reduce the scope of the largest row
        scopes = [_row_scope(r) for r in cur_rows]
        bi = scopes.index(max(scopes))
        row = cur_rows[bi]
        improved = False
        for idx in range(1, len(row)):
            # Try to decrease row[idx]
            for dec in range(1, 10):
                new_val = row[idx] - dec
                if new_val <= row[idx - 1]:
                    break
                if idx + 1 < len(row) and new_val >= row[idx + 1]:
                    continue
                # Check if this maintains validity
                delta_conf, old_d, new_d = _compute_delta_conflicts(row, idx, row[idx], new_val, counter)
                if delta_conf == 0:
                    # Valid move, apply it
                    row[idx] = new_val
                    improved = True
                    break
                else:
                    _revert_counter(counter, old_d, new_d)
            if improved:
                break
        if not improved:
            # Try a random small perturbation to escape local optimum
            ri = rng.randrange(len(cur_rows))
            row = cur_rows[ri]
            idx = 1 + rng.randrange(len(row) - 1)
            old_val = row[idx]
            window = 3
            delta = rng.randint(-window, window)
            new_val = old_val + delta
            lo = row[idx - 1] + 1
            hi = (row[idx + 1] - 1) if idx + 1 < len(row) else 10**9
            if new_val < lo:
                new_val = lo
            if new_val > hi:
                new_val = hi
            if new_val != old_val:
                delta_conf, old_d, new_d = _compute_delta_conflicts(row, idx, old_val, new_val, counter)
                if delta_conf <= 0:
                    row[idx] = new_val
                    cur_conf += delta_conf
                else:
                    _revert_counter(counter, old_d, new_d)
    return cur_rows, cur_conf


def _kick(rows, counter, cur_conf, intensity, rng):
    """Apply a kick perturbation to escape local optima."""
    cur_rows = [list(r) for r in rows]
    n_kicks = max(1, intensity)
    for _ in range(n_kicks):
        ri = rng.randrange(len(cur_rows))
        row = cur_rows[ri]
        idx = 1 + rng.randrange(len(row) - 1)
        old_val = row[idx]
        window = rng.randint(3, 10)
        delta = rng.randint(-window, window)
        new_val = old_val + delta
        lo = row[idx - 1] + 1
        hi = (row[idx + 1] - 1) if idx + 1 < len(row) else 10**9
        if new_val < lo:
            new_val = lo
        if new_val > hi:
            new_val = hi
        if new_val != old_val:
            delta_conf, old_d, new_d = _compute_delta_conflicts(row, idx, old_val, new_val, counter)
            # Always accept kick (even if worse)
            row[idx] = new_val
            cur_conf += delta_conf
    return cur_rows, cur_conf


def _ils_optimize(rows, k, deadline, rng):
    """Iterated Local Search with kick perturbations."""
    t0 = time.time()
    cur_rows = [list(r) for r in rows]
    cur_counter = _build_counter(cur_rows)
    cur_conf = _conflicts_from_counter(cur_counter)
    best_rows = [list(r) for r in cur_rows]
    best_conf = cur_conf
    best_scope = max(_row_scope(r) for r in best_rows)

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Phase 1: Feasibility - reduce conflicts to zero
        cur_rows, cur_conf = _local_search_feasibility(cur_rows, cur_counter, cur_conf, 500, rng)
        # Rebuild counter if needed
        if cur_conf == 0:
            # Phase 2: Optimization - reduce scope
            cur_rows, cur_conf = _local_search_optimization(cur_rows, cur_counter, cur_conf, 300, rng)
            # Update best
            cur_scope = max(_row_scope(r) for r in cur_rows)
            if cur_conf < best_conf or (cur_conf == best_conf and cur_scope < best_scope):
                best_conf = cur_conf
                best_scope = cur_scope
                best_rows = [list(r) for r in cur_rows]
                cur_counter = _build_counter(cur_rows)
            else:
                # Apply kick to escape local optimum
                kick_intensity = 2 + (iteration // 5)
                cur_rows, cur_conf = _kick(cur_rows, cur_counter, cur_conf, kick_intensity, rng)
                cur_counter = _build_counter(cur_rows)
        else:
            # Still infeasible, apply stronger kick
            kick_intensity = 3 + (iteration // 3)
            cur_rows, cur_conf = _kick(cur_rows, cur_counter, cur_conf, kick_intensity, rng)
            cur_counter = _build_counter(cur_rows)

        # Check time
        if time.time() > deadline - 2:
            break

    # Final polish on best
    if best_conf == 0:
        _local_polish(best_rows, k, rng)
        # Verify validity
        if not _valid(best_rows):
            best_rows = [list(r) for r in rows]
            best_conf = _total_conflicts(best_rows)

    return best_rows


def _local_polish(rows, k, rng):
    """Try to reduce the scope of the largest row by small local moves, keeping validity."""
    scopes = [_row_scope(r) for r in rows]
    bi = scopes.index(max(scopes))
    row = rows[bi]
    counter = _build_counter(rows)
    improved = True
    while improved:
        improved = False
        for idx in range(1, len(row)):
            for dec in range(1, 8):
                new_val = row[idx] - dec
                if new_val <= row[idx - 1]:
                    break
                if idx + 1 < len(row) and new_val >= row[idx + 1]:
                    continue
                delta_conf, old_d, new_d = _compute_delta_conflicts(row, idx, row[idx], new_val, counter)
                if delta_conf == 0:
                    row[idx] = new_val
                    improved = True
                    break
                else:
                    _revert_counter(counter, old_d, new_d)
    row.sort()


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct across rows."""
    rng = random.Random(12345)

    # Greedy initial
    rows = _greedy_all(n_rows, k)

    t0 = time.time()
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 5:
        return rows

    # ILS optimization
    rows = _ils_optimize(rows, k, _DEADLINE, rng)

    # Final local polish on the result
    _local_polish(rows, k, rng)

    # Ensure validity; if ILS somehow broke it, fall back to greedy
    if not _valid(rows):
        rows = _greedy_all(n_rows, k)

    return rows
