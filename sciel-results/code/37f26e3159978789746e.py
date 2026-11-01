"""Difference triangle set (4,7): conflict-directed repair targeting scope minimization."""
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


def _build_counter(rows):
    """Build a Counter of all differences across all rows."""
    c = Counter()
    for row in rows:
        for d in _row_diffs(row):
            c[d] += 1
    return c


def _conflicts_from_counter(c):
    """Count total conflicts from a difference counter."""
    return sum(v - 1 for v in c.values() if v > 1)


def _row_scope(row):
    return row[-1]


def _global_scope(rows):
    return max(_row_scope(r) for r in rows)


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


def _try_reduce_element(row, idx, rows, counter, max_search=200):
    """
    Try to reduce row[idx] to the smallest valid value > row[idx-1].
    Returns (new_val, delta_conf) if successful, else None.
    """
    current_val = row[idx]
    lo = row[idx - 1] + 1
    hi = (row[idx + 1] - 1) if idx + 1 < len(row) else current_val

    # Try values from lo upward, but only up to current_val (we want to reduce)
    for candidate in range(lo, min(hi + 1, current_val)):
        if candidate == current_val:
            break

        # Compute old and new diffs involving idx
        old_d = []
        new_d = []
        for j in range(len(row)):
            if j != idx:
                if j < idx:
                    old_d.append(current_val - row[j])
                    new_d.append(candidate - row[j])
                else:
                    old_d.append(row[j] - current_val)
                    new_d.append(row[j] - candidate)

        # Check if this change reduces conflicts
        delta_conf = 0
        valid = True

        # Simulate: remove old diffs, add new diffs
        for d in old_d:
            c = counter[d]
            if c >= 2:
                delta_conf -= 1
        for d in new_d:
            c = counter[d]
            if c >= 1:
                delta_conf += 1
            # If after adding, count would exceed 1 AND this diff is new (not in old_d),
            # we need to check more carefully
            if c >= 2:
                # This diff already has >=2 occurrences; adding makes it worse
                # But we need to check if it was in old_d (then net effect is different)
                pass

        # More careful check: compute actual new conflict count for affected diffs
        # For simplicity, only accept if delta_conf < 0 (strictly reduces conflicts)
        if delta_conf < 0:
            return candidate, delta_conf

    return None


def _conflict_directed_repair(rows, k, deadline, rng):
    """
    Iteratively try to reduce the largest elements in rows, using conflict-directed
    repair. Focus on the binding row (largest scope) and try to shrink its elements.
    """
    t0 = time.time()
    best_rows = [list(r) for r in rows]
    best_scope = _global_scope(best_rows)
    best_counter = _build_counter(best_rows)
    best_conf = _conflicts_from_counter(best_counter)

    cur_rows = [list(r) for r in rows]
    cur_counter = _build_counter(cur_rows)
    cur_conf = _conflicts_from_counter(cur_counter)

    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        # Find the row with maximum scope
        scopes = [_row_scope(r) for r in cur_rows]
        binding_idx = scopes.index(max(scopes))
        binding_row = cur_rows[binding_idx]

        # Try to reduce elements in the binding row, starting from the end
        improved = False
        for idx in range(len(binding_row) - 1, 0, -1):
            result = _try_reduce_element(binding_row, idx, cur_rows, cur_counter)
            if result is not None:
                new_val, delta_conf = result
                # Apply the change
                old_val = binding_row[idx]

                # Update counter
                old_d = []
                new_d = []
                for j in range(len(binding_row)):
                    if j != idx:
                        if j < idx:
                            old_d.append(old_val - binding_row[j])
                            new_d.append(new_val - binding_row[j])
                        else:
                            old_d.append(binding_row[j] - old_val)
                            new_d.append(binding_row[j] - new_val)

                for d in old_d:
                    cur_counter[d] -= 1
                    if cur_counter[d] == 0:
                        del cur_counter[d]
                for d in new_d:
                    cur_counter[d] += 1

                binding_row[idx] = new_val
                cur_conf += delta_conf
                improved = True

                # Update best if scope decreased
                new_scope = _global_scope(cur_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in cur_rows]
                    best_counter = _build_counter(best_rows)
                    best_conf = _conflicts_from_counter(best_counter)

                no_improve_count = 0
                break  # Restart from the end after one improvement

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a different strategy: random perturbation + repair
                # Pick a random row and element, try to increase it slightly
                # then repair downward
                ri = rng.randrange(len(cur_rows))
                row = cur_rows[ri]
                if len(row) > 1:
                    idx = 1 + rng.randrange(len(row) - 1)
                    old_val = row[idx]
                    # Try increasing by a small amount
                    inc = rng.randint(1, 5)
                    new_val = old_val + inc
                    # Check bounds
                    if idx + 1 < len(row) and new_val >= row[idx + 1]:
                        new_val = row[idx + 1] - 1
                    if new_val <= old_val:
                        continue

                    # Check if this is feasible (no new conflicts that can't be resolved)
                    old_d = []
                    new_d = []
                    for j in range(len(row)):
                        if j != idx:
                            if j < idx:
                                old_d.append(old_val - row[j])
                                new_d.append(new_val - row[j])
                            else:
                                old_d.append(row[j] - old_val)
                                new_d.append(row[j] - new_val)

                    # Tentatively apply and see if we can repair
                    for d in old_d:
                        cur_counter[d] -= 1
                        if cur_counter[d] == 0:
                            del cur_counter[d]
                    for d in new_d:
                        cur_counter[d] += 1

                    row[idx] = new_val
                    new_conf = _conflicts_from_counter(cur_counter)

                    # Try to repair: reduce other elements to fix conflicts
                    repaired = False
                    for _ in range(50):
                        # Find a conflicting diff
                        conflict_diffs = [d for d, c in cur_counter.items() if c > 1]
                        if not conflict_diffs:
                            repaired = True
                            break

                        # Pick a random conflicting diff
                        cd = rng.choice(conflict_diffs)

                        # Find rows that contain this diff
                        for ridx, r in enumerate(cur_rows):
                            # Check if this row has the diff cd
                            for i in range(len(r)):
                                for j in range(i + 1, len(r)):
                                    if r[j] - r[i] == cd:
                                        # Try to adjust r[j] or r[i] to remove this diff
                                        # Try reducing r[j]
                                        for dec in range(1, 10):
                                            nv = r[j] - dec
                                            if nv <= r[j-1]:
                                                break
                                            if j + 1 < len(r) and nv >= r[j+1]:
                                                continue
                                            # Check if this removes the conflict
                                            # Simulate
                                            test_old = []
                                            test_new = []
                                            for jj in range(len(r)):
                                                if jj != j:
                                                    if jj < j:
                                                        test_old.append(r[j] - r[jj])
                                                        test_new.append(nv - r[jj])
                                                    else:
                                                        test_old.append(r[jj] - r[j])
                                                        test_new.append(r[jj] - nv)

                                            # Quick check: does new set avoid cd?
                                            if cd not in test_new:
                                                # Apply
                                                for d in test_old:
                                                    cur_counter[d] -= 1
                                                    if cur_counter[d] == 0:
                                                        del cur_counter[d]
                                                for d in test_new:
                                                    cur_counter[d] += 1
                                                r[j] = nv
                                                repaired = True
                                                break
                                            if repaired:
                                                break
                                        if repaired:
                                            break
                            if repaired:
                                break

                        if not repaired:
                            # Revert the perturbation
                            for d in new_d:
                                cur_counter[d] -= 1
                                if cur_counter[d] == 0:
                                    del cur_counter[d]
                            for d in old_d:
                                cur_counter[d] += 1
                            row[idx] = old_val
                            break

                    if repaired:
                        new_scope = _global_scope(cur_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in cur_rows]
                            best_counter = _build_counter(best_rows)
                            best_conf = _conflicts_from_counter(best_counter)
                        no_improve_count = 0
                    else:
                        # Already reverted
                        pass

            if no_improve_count > max_no_improve * 2:
                break

    return best_rows


def _final_polish(rows, k):
    """Final pass: try to reduce each element of each row by 1 repeatedly."""
    improved = True
    while improved:
        improved = False
        counter = _build_counter(rows)
        for ri, row in enumerate(rows):
            for idx in range(len(row) - 1, 0, -1):
                current = row[idx]
                lo = row[idx - 1] + 1
                hi = (row[idx + 1] - 1) if idx + 1 < len(row) else current

                for candidate in range(lo, min(hi + 1, current)):
                    if candidate == current:
                        break

                    old_d = []
                    new_d = []
                    for j in range(len(row)):
                        if j != idx:
                            if j < idx:
                                old_d.append(current - row[j])
                                new_d.append(candidate - row[j])
                            else:
                                old_d.append(row[j] - current)
                                new_d.append(row[j] - candidate)

                    # Check validity
                    delta = 0
                    for d in old_d:
                        c = counter[d]
                        if c >= 2:
                            delta -= 1
                    for d in new_d:
                        c = counter[d]
                        if c >= 1:
                            delta += 1

                    if delta <= 0:
                        # Apply
                        for d in old_d:
                            counter[d] -= 1
                            if counter[d] == 0:
                                del counter[d]
                        for d in new_d:
                            counter[d] += 1
                        row[idx] = candidate
                        improved = True
                        break

    return rows


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct across rows."""
    rng = random.Random(12345)

    # Greedy initial
    rows = _greedy_all(n_rows, k)

    t0 = time.time()
    if time.time() > _DEADLINE - 5:
        return rows

    # Conflict-directed repair
    rows = _conflict_directed_repair(rows, k, _DEADLINE, rng)

    # Final polish
    if time.time() < _DEADLINE - 2:
        rows = _final_polish(rows, k)

    # Ensure validity; if broken, fall back to greedy
    if not _valid(rows):
        rows = _greedy_all(n_rows, k)

    return rows
