"""Difference triangle set (7,5) solver: randomized greedy + local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _max_val(rows):
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check all positive differences are distinct across all rows."""
    seen = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_row(used, k, max_val=None, rng=None):
    """Build one row of k+1 increasing integers starting at 0, greedily.
    Returns (row, used_set) or (None, used_set) if stuck."""
    if rng is None:
        rng = random.Random()
    row = [0]
    cur = 0
    # Collect all differences as we go
    while len(row) < k + 1:
        cur += 1
        if max_val is not None and cur > max_val:
            return None, used
        # Check if cur - a for all a in row are unused
        new_diffs = set()
        ok = True
        for a in row:
            d = cur - a
            if d in used:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            row.append(cur)
            used |= new_diffs
        # else: try next value (greedy just increments)
    return row, used

def _greedy_row_randomized(used, k, max_val=None, rng=None):
    """Randomized greedy: sometimes skip a value to allow different structures."""
    if rng is None:
        rng = random.Random()
    row = [0]
    cur = 0
    while len(row) < k + 1:
        cur += 1
        if max_val is not None and cur > max_val:
            return None, used
        new_diffs = set()
        ok = True
        for a in row:
            d = cur - a
            if d in used:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            # With some probability, skip this valid value to explore
            if rng.random() < 0.05:
                continue
            row.append(cur)
            used |= new_diffs
    return row, used

def _randomized_greedy(n_rows, k, max_val=None, rng=None, randomize=False):
    """Build n_rows using greedy, optionally randomized."""
    if rng is None:
        rng = random.Random()
    used = set()
    rows = []
    for _ in range(n_rows):
        if randomize:
            row, used = _greedy_row_randomized(used, k, max_val, rng)
        else:
            row, used = _greedy_row(used, k, max_val, rng)
        if row is None:
            return None
        rows.append(row)
    return rows

def _local_improve(rows, k, n_rows, rng=None, max_iters=5000):
    """Local search: try to reduce the max value by modifying rows.
    Strategy: pick the row with the largest last element, try to rebuild it
    with a smaller max, or swap elements within a row."""
    if rng is None:
        rng = random.Random()
    best_rows = [list(r) for r in rows]
    best_max = _max_val(best_rows)
    
    for it in range(max_iters):
        # Find the row with the largest max
        row_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
        row_max = max(best_rows[row_idx])
        
        # Compute used differences excluding this row
        used_other = set()
        for i, r in enumerate(best_rows):
            if i == row_idx:
                continue
            n = len(r)
            for a in range(n):
                for b in range(a + 1, n):
                    used_other.add(r[b] - r[a])
        
        # Try to rebuild this row with a smaller max
        target_max = row_max - 1
        improved = False
        
        # Try greedy rebuild with tighter bound
        for attempt in range(20):
            new_row, _ = _greedy_row(used_other, k, target_max, rng)
            if new_row is not None and max(new_row) < row_max:
                best_rows[row_idx] = new_row
                new_max = _max_val(best_rows)
                if new_max < best_max:
                    best_max = new_max
                    improved = True
                break
        
        if not improved:
            # Try swapping the last element with a smaller valid one
            # or trying different intermediate values
            r = best_rows[row_idx]
            # Try to reduce the last element
            last = r[-1]
            for new_last in range(last - 1, max(0, last - 30), -1):
                candidate = r[:-1] + [new_last]
                # Check strictly increasing
                if candidate[-2] >= candidate[-1]:
                    continue
                # Check differences
                valid = True
                for i in range(len(candidate)):
                    for j in range(i + 1, len(candidate)):
                        d = candidate[j] - candidate[i]
                        if d in used_other:
                            valid = False
                            break
                        # Also check within-row distinctness (should be fine if increasing)
                    if not valid:
                        break
                if valid:
                    best_rows[row_idx] = candidate
                    new_max = _max_val(best_rows)
                    if new_max < best_max:
                        best_max = new_max
                    break
        
        if best_max <= 105:  # counting bound, can't do better
            break
    
    return best_rows, best_max

def _multi_start(n_rows, k, time_budget=100, seed=42):
    """Multi-start randomized greedy with local search."""
    rng = random.Random(seed)
    best_rows = None
    best_max = float('inf')
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # First: deterministic greedy
    rows = _randomized_greedy(n_rows, k, rng=rng, randomize=False)
    if rows is not None:
        m = _max_val(rows)
        if m < best_max:
            best_max = m
            best_rows = [list(r) for r in rows]
    
    # Multi-start with randomization
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Vary the randomization level
        randomize = (attempt % 3 != 0)  # 2/3 randomized, 1/3 deterministic-ish
        
        rows = _randomized_greedy(n_rows, k, rng=rng, randomize=randomize)
        if rows is None:
            continue
        
        m = _max_val(rows)
        if m < best_max:
            best_max = m
            best_rows = [list(r) for r in rows]
        
        # Local search on the current best
        if time.time() < deadline - 5:
            improved_rows, improved_max = _local_improve(
                best_rows, k, n_rows, rng=rng, max_iters=2000
            )
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        
        # Also try local search on this particular solution
        if m < best_max + 5 and time.time() < deadline - 5:
            improved_rows, improved_max = _local_improve(
                rows, k, n_rows, rng=rng, max_iters=1000
            )
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
    
    return best_rows

def _hill_climb(rows, k, n_rows, rng, time_limit=30):
    """Hill climbing: try to reduce max by modifying individual elements."""
    best_rows = [list(r) for r in rows]
    best_max = _max_val(best_rows)
    start = time.time()
    
    while time.time() - start < time_limit:
        # Compute all used differences
        all_diffs = set()
        for r in best_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.add(r[j] - r[i])
        
        # Try to reduce the global max
        # Find row with max last element
        row_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
        r = best_rows[row_idx]
        current_max = max(r)
        
        # Try reducing the last element
        improved = False
        for new_val in range(current_max - 1, max(r[-2], 0), -1):
            if new_val <= r[-2]:
                break
            candidate = r[:-1] + [new_val]
            # Check all diffs
            valid = True
            for i in range(len(candidate)):
                for j in range(i + 1, len(candidate)):
                    d = candidate[j] - candidate[i]
                    if d in all_diffs and d != (r[j] - r[i] if j < len(r) else None):
                        # Need to be more careful
                        pass
            # Simpler: just rebuild used set
            other_diffs = set()
            for ii, rr in enumerate(best_rows):
                if ii == row_idx:
                    continue
                for a in range(len(rr)):
                    for b in range(a + 1, len(rr)):
                        other_diffs.add(rr[b] - rr[a])
            
            ok = True
            for i in range(len(candidate)):
                for j in range(i + 1, len(candidate)):
                    d = candidate[j] - candidate[i]
                    if d in other_diffs:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                best_rows[row_idx] = candidate
                best_max = _max_val(best_rows)
                improved = True
                break
        
        if not improved:
            # Try random perturbation: pick a random row and random position, try a different value
            if rng.random() < 0.3:
                pi = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k)  # not the first element
                old_val = best_rows[pi][pos]
                # Try a nearby value
                delta = rng.choice([-2, -1, 1, 2, -3, 3])
                new_val = old_val + delta
                if new_val > best_rows[pi][pos - 1] and (pos == k or new_val < best_rows[pi][pos + 1]):
                    candidate = best_rows[pi][:]
                    candidate[pos] = new_val
                    # Validate
                    other_diffs = set()
                    for ii, rr in enumerate(best_rows):
                        if ii == pi:
                            continue
                        for a in range(len(rr)):
                            for b in range(a + 1, len(rr)):
                                other_diffs.add(rr[b] - rr[a])
                    ok = True
                    for i in range(len(candidate)):
                        for j in range(i + 1, len(candidate)):
                            d = candidate[j] - candidate[i]
                            if d in other_diffs:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        new_max = max(best_rows[:pi] + [candidate] + best_rows[pi+1:])
                        if new_max <= best_max:
                            best_rows[pi] = candidate
                            if new_max < best_max:
                                best_max = new_max
    
    return best_rows, best_max

def construct(n_rows, k):
    """Main entry point. Returns list of n_rows rows, each a list of k+1 integers."""
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Quick deterministic solution first
    rows = _randomized_greedy(n_rows, k, randomize=False)
    if rows is None:
        # Fallback: very simple
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = [list(r) for r in rows]
    best_max = _max_val(best_rows)
    
    # Multi-start with local search
    if time.time() < deadline - 10:
        result = _multi_start(n_rows, k, time_budget=min(80, deadline - time.time() - 10), seed=42)
        if result is not None:
            m = _max_val(result)
            if m < best_max:
                best_max = m
                best_rows = result
    
    # Hill climbing on the best
    if time.time() < deadline - 5:
        rng = random.Random(123)
        hc_rows, hc_max = _hill_climb(best_rows, k, n_rows, rng, time_limit=min(20, deadline - time.time() - 5))
        if hc_max < best_max:
            best_max = hc_max
            best_rows = hc_rows
    
    return best_rows

# EVOLVE-BLOCK-END
