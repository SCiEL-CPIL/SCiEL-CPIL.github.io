"""Difference triangle set (4,7) – improved search with local repair."""
import time
import random
import sys

def _row_diffs(row):
    """Return the set of positive differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(row[j] - row[i])
    return used

def _is_valid(rows, k):
    """Check if rows form a valid difference triangle set."""
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    return max(max(r) for r in rows)

def _greedy_row(used, k, start_hint=1):
    """Build a single row greedily, avoiding used differences."""
    row = [0]
    x = max(1, start_hint)
    while len(row) < k + 1:
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
        x += 1
    return row

def _random_row(used, k, max_val):
    """Try to build a row with random-ish increments."""
    row = [0]
    x = random.randint(1, max_val // (k + 1))
    while len(row) < k + 1:
        new = {x - a for a in row}
        if not new & used and x <= max_val:
            row.append(x)
            used |= new
            x = row[-1] + random.randint(1, max(1, max_val // (k * 2)))
        else:
            x += 1
    if row[-1] <= max_val:
        return row
    return None

def _local_search(rows, k, max_iter=5000, time_limit=100):
    """Local search: try to reduce the scope by adjusting individual entries."""
    start = time.time()
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    
    # Try to improve by reducing the largest entry
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Find the row with the largest entry
        max_idx = 0
        max_val = best[0][-1]
        for i, r in enumerate(best):
            if r[-1] > max_val:
                max_val = r[-1]
                max_idx = i
        
        # Try to reduce the last element of that row
        row = best[max_idx]
        used_others = set()
        for i, r in enumerate(best):
            if i != max_idx:
                used_others |= _row_diffs(r)
        
        # Try smaller values for the last element
        for new_last in range(row[-1] - 1, row[-2], -1):
            new_row = row[:-1] + [new_last]
            new_diffs = _row_diffs(new_row)
            if not new_diffs & used_others:
                # Check internal consistency
                if len(new_diffs) == len(new_row) * (len(new_row) - 1) // 2:
                    best[max_idx] = new_row
                    new_scope = _scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    break
        
        # Also try random perturbations
        if random.random() < 0.3:
            r_idx = random.randint(0, len(best) - 1)
            pos_idx = random.randint(1, len(best[r_idx]) - 1)
            orig_val = best[r_idx][pos_idx]
            
            # Try a random new value
            new_val = orig_val + random.randint(-5, 5)
            if new_val > best[r_idx][pos_idx - 1] and (pos_idx == len(best[r_idx]) - 1 or new_val < best[r_idx][pos_idx + 1]):
                new_row = best[r_idx][:]
                new_row[pos_idx] = new_val
                # Ensure strictly increasing
                valid = True
                for i in range(1, len(new_row)):
                    if new_row[i] <= new_row[i-1]:
                        valid = False
                        break
                if valid:
                    new_diffs = _row_diffs(new_row)
                    used_others = set()
                    for i, r in enumerate(best):
                        if i != r_idx:
                            used_others |= _row_diffs(r)
                    if not new_diffs & used_others and len(new_diffs) == len(new_row) * (len(new_row) - 1) // 2:
                        best[r_idx] = new_row
                        new_scope = _scope(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
    
    return best, best_scope

def _multi_start_greedy(n_rows, k, num_starts=50, time_limit=100):
    """Run multiple greedy starts with different random seeds/orderings."""
    start = time.time()
    best_rows = None
    best_scope = float('inf')
    
    for s in range(num_starts):
        if time.time() - start > time_limit:
            break
        
        random.seed(s * 7919 + 1)
        used = set()
        rows = []
        
        # Shuffle the order of rows to break symmetry
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        row_map = {}
        for ri in row_order:
            row = _greedy_row(used, k)
            if row is None:
                break
            row_map[ri] = row
            used |= _row_diffs(row)
        
        if len(row_map) == n_rows:
            rows = [row_map[i] for i in range(n_rows)]
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    return best_rows, best_scope

def construct(n_rows, k):
    deadline = time.time() + 120
    time_budget = 100  # leave 20s margin
    
    # Phase 1: Multi-start greedy
    rows, scope = _multi_start_greedy(n_rows, k, num_starts=100, time_limit=min(30, time_budget * 0.3))
    
    if rows is None:
        # Fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k)
            rows.append(row)
            used |= _row_diffs(row)
        scope = _scope(rows)
    
    # Phase 2: Local search to improve
    remaining = time_budget - (time.time() - (deadline - time_budget))
    if remaining > 5:
        rows, scope = _local_search(rows, k, max_iter=20000, time_limit=min(remaining - 5, 60))
    
    # Final check
    if not _is_valid(rows, k):
        # Rebuild with simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k)
            rows.append(row)
            used |= _row_diffs(row)
    
    return rows
