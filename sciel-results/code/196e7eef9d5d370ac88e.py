"""Difference triangle set (4,7): SA with Suffix Re-greedy Moves."""
import time
import random

_DEADLINE = 145.0  # safety margin under 150s wall-clock


def _row_diffs(row):
    """All positive differences a[j]-a[i] for i<j in one row."""
    d = []
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.append(row[j] - ai)
    return d


def _conflict_count(rows):
    """Count duplicate differences across all rows (0 = valid)."""
    seen = set()
    conflicts = 0
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                conflicts += 1
            else:
                seen.add(d)
    return conflicts


def _row_scope(row):
    return row[-1]


def _total_conflicts(rows):
    return _conflict_count(rows)


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


def _greedy_suffix(prefix, forbidden, k):
    """
    Greedily extend the prefix to length k+1.
    prefix: list of ints, length L.
    forbidden: set of diffs that cannot be used.
    Returns the full row.
    """
    row = list(prefix)
    # Diffs already in prefix
    internal_diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            internal_diffs.add(row[j] - row[i])
    
    # Combine with forbidden
    used = forbidden | internal_diffs
    
    # Greedy extend
    x = row[-1]
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


def _sa_optimize(rows, k, deadline, rng):
    """Simulated Annealing with Suffix Re-greedy Moves."""
    n = len(rows)
    best_rows = [list(r) for r in rows]
    best_conf = _total_conflicts(best_rows)
    best_scope = max(_row_scope(r) for r in best_rows)

    cur_rows = [list(r) for r in rows]
    cur_conf = _total_conflicts(cur_rows)

    t_start = max(1.0, cur_conf * 0.5)
    t_end = 0.01

    t0 = time.time()
    elapsed_total = deadline - t0
    if elapsed_total <= 1:
        return best_rows

    from collections import Counter

    def build_counter(rows):
        c = Counter()
        for row in rows:
            for d in _row_diffs(row):
                c[d] += 1
        return c

    def conflicts_from_counter(c):
        return sum(v - 1 for v in c.values() if v > 1)

    cur_counter = build_counter(cur_rows)
    cur_conf = conflicts_from_counter(cur_counter)

    if cur_conf < best_conf or (cur_conf == best_conf and max(_row_scope(r) for r in cur_rows) < best_scope):
        best_conf = cur_conf
        best_rows = [list(r) for r in cur_rows]
        best_scope = max(_row_scope(r) for r in best_rows)

    steps = 0
    while time.time() < deadline:
        elapsed = time.time() - t0
        frac = min(1.0, elapsed / elapsed_total)
        temp = t_end + (t_start - t_end) * (1.0 - frac) ** 1.5
        if temp < 1e-6:
            temp = 1e-6

        # Pick a random row and a random suffix start index
        ri = rng.randrange(n)
        # Suffix start can be 1 to k (index of the first element to re-greedy)
        # If idx=1, we re-greedy the whole row (except 0).
        # If idx=k, we re-greedy only the last element.
        idx = 1 + rng.randrange(k) 
        
        row = cur_rows[ri]
        prefix = row[:idx]
        
        # Compute forbidden diffs from other rows
        forbidden = set()
        for i in range(n):
            if i == ri:
                continue
            forbidden.update(_row_diffs(cur_rows[i]))
            
        # Also, diffs in the prefix are "used" internally
        # _greedy_suffix handles this by combining forbidden and internal prefix diffs
        
        new_row = _greedy_suffix(prefix, forbidden, k)
        
        # Calculate delta
        # Remove old suffix diffs, add new suffix diffs
        # This is complex to do incrementally with the counter if we change multiple elements.
        # Given k is small, we can just rebuild the counter for the affected row?
        # Or simpler: Just calculate new_conf directly for this step?
        # No, we need speed.
        
        # Let's revert to the simpler single-element move for speed, 
        # but use the "Suffix Re-greedy" as a *restart* or *large move* occasionally.
        
        # For now, let's stick to the single element move but with a better window.
        # The previous code's SA was decent. Let's try to improve the acceptance criteria or window.
        
        # Actually, let's just use the single element move as in the parent, 
        # but ensure we are maximizing the probability of finding the global min.
        
        # Let's try a different strategy: 
        # If we are stuck (no improvement for X steps), do a large perturbation (re-greedy a whole row).
        
        # For this edit, I will replace the SA loop with a hybrid:
        # 1. Single element moves (fast).
        # 2. Every 1000 steps, re-greedy a random row (slow but escapes).
        
        if steps % 1000 == 0:
            # Large move: Re-greedy a random row
            ri = rng.randrange(n)
            forbidden = set()
            for i in range(n):
                if i == ri:
                    continue
                forbidden.update(_row_diffs(cur_rows[i]))
            
            new_row = _greedy_row(forbidden, k)
            
            # Update counter
            # Remove old row diffs
            for d in _row_diffs(cur_rows[ri]):
                cur_counter[d] -= 1
            # Add new row diffs
            for d in _row_diffs(new_row):
                cur_counter[d] += 1
            
            cur_rows[ri] = new_row
            cur_conf = conflicts_from_counter(cur_counter)
            
            if cur_conf < best_conf or (cur_conf == best_conf and max(_row_scope(r) for r in cur_rows) < best_scope):
                best_conf = cur_conf
                best_rows = [list(r) for r in cur_rows]
                best_scope = max(_row_scope(r) for r in best_rows)
        else:
            # Small move: Single element
            ri = rng.randrange(n)
            row = cur_rows[ri]
            idx = 1 + rng.randrange(len(row) - 1)
            old_val = row[idx]

            window = int(1 + temp * 5)
            delta = rng.randint(-window, window)
            new_val = old_val + delta
            if new_val <= 0:
                new_val = old_val + 1
            
            lo = row[idx - 1] + 1
            hi = (row[idx + 1] - 1) if idx + 1 < len(row) else 10**9
            if new_val < lo:
                new_val = lo
            if new_val > hi:
                new_val = hi
            if new_val == old_val:
                continue

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
            for d in old_d:
                c = cur_counter[d]
                if c >= 2:
                    delta_conf -= 1
                cur_counter[d] = c - 1
            for d in new_d:
                c = cur_counter[d]
                if c >= 1:
                    delta_conf += 1
                cur_counter[d] = c + 1

            new_conf = cur_conf + delta_conf

            if new_conf < cur_conf:
                accept = True
            elif new_conf == cur_conf:
                new_row_scope = max(row[:idx] + [new_val] + row[idx+1:])
                old_row_scope = _row_scope(row)
                other_max = max(_row_scope(r) for i, r in enumerate(cur_rows) if i != ri) if n > 1 else 0
                new_total_scope = max(new_row_scope, other_max)
                old_total_scope = max(old_row_scope, other_max)
                if new_total_scope <= old_total_scope:
                    accept = True
                else:
                    accept = rng.random() < 0.3
            else:
                delta = new_conf - cur_conf
                if delta > 0:
                    if delta / temp > 50:
                        accept = False
                    else:
                        accept = rng.random() < (2.718281828 ** (-(delta / temp)))
                else:
                    accept = True

            if accept:
                row[idx] = new_val
                cur_conf = new_conf
                if cur_conf < best_conf or (cur_conf == best_conf and max(_row_scope(r) for r in cur_rows) < best_scope):
                    best_conf = cur_conf
                    best_rows = [list(r) for r in cur_rows]
                    best_scope = max(_row_scope(r) for r in best_rows)
            else:
                for d in new_d:
                    cur_counter[d] -= 1
                for d in old_d:
                    cur_counter[d] += 1

        steps += 1

    return best_rows


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

    # SA optimization
    rows = _sa_optimize(rows, k, _DEADLINE, rng)

    # Ensure validity; if SA somehow broke it, fall back to greedy
    if not _valid(rows):
        rows = _greedy_all(n_rows, k)

    return rows
