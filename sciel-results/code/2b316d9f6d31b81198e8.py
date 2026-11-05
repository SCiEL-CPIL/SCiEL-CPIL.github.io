"""Difference triangle set (4,7) – structured search with deadline guard."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _scope(rows):
    """Return the largest entry across all rows (the objective)."""
    return max(max(r) for r in rows)

def _validate(rows, k):
    """Check that all positive intra-row differences are distinct across rows."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_row(row, used, max_val, k):
    """
    Complete a partially-built row by greedily picking the smallest next value
    whose differences to earlier marks don't collide with `used`.
    Returns (completed_row, new_used) or (None, used) on failure.
    """
    while len(row) < k + 1:
        found = False
        for x in range(row[-1] + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None, used
    return row, used

def _greedy_all(n_rows, k, max_val, order=None):
    """
    Build n_rows rows greedily. If order is given, build rows in that order.
    Returns list of rows or None on failure.
    """
    if order is None:
        order = list(range(n_rows))
    used = set()
    rows = [None] * n_rows
    for idx in order:
        row = [0]
        res, used = _greedy_row(row, used, max_val, k)
        if res is None:
            return None
        rows[idx] = res
    return rows

def _randomized_greedy(n_rows, k, max_val, seed, n_samples=50):
    """
    Try many random row orders with greedy completion.
    Returns the best valid set found (min scope) or None.
    """
    rng = random.Random(seed)
    best_rows = None
    best_scope = max_val + 1
    for _ in range(n_samples):
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_all(n_rows, k, max_val, order)
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
    return best_rows

def _backtrack_rows(n_rows, k, max_val, deadline, seed, max_nodes=200000):
    """
    Backtracking search: build rows one at a time, choosing values with
    backtracking within each row. Uses a deadline guard.
    Returns best valid set found or None.
    """
    rng = random.Random(seed)
    best_rows = None
    best_scope = max_val + 1
    used = set()
    rows = []
    nodes = [0]
    
    def build_row(row, pos):
        """Recursively build the current row starting at position pos."""
        if time.time() > deadline:
            return False
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return False
        if pos == k + 1:
            rows.append(row[:])
            return True
        start = row[-1] + 1 if row else 1
        # Try values in randomized order for diversity, but bias toward small
        candidates = list(range(start, max_val + 1))
        # Limit branching: try a subset if too many
        if len(candidates) > 30:
            # Pick a mix: some small, some random
            small = candidates[:10]
            rest = candidates[10:]
            rng.shuffle(rest)
            candidates = small + rest[:20]
        
        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            old_used = used
            used |= new_diffs
            if build_row(row, pos + 1):
                return True
            row.pop()
            used.clear()
            used.update(old_used)
        return False
    
    def build_all_row_idx(idx):
        """Build all rows starting from row index idx."""
        if time.time() > deadline:
            return False
        if idx == n_rows:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows[:] = [r[:] for r in rows]
            return True
        row = [0]
        if build_row(row, 1):
            if build_all_row_idx(idx + 1):
                return True
            rows.pop()
        return False
    
    build_all_row_idx(0)
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Upper bound on scope: simple counting bound gives ~112 for (4,7),
    # but we need a safe upper bound. Use a generous one.
    # For n_rows rows of k+1 entries, total diffs = n_rows * k*(k+1)/2
    # Diffs range from 1 to max_val, so max_val >= n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    # Add some slack
    max_val = total_diffs + 20
    
    # First: try pure greedy with natural order
    rows = _greedy_all(n_rows, k, max_val)
    if rows is not None:
        best = [r[:] for r in rows]
        best_scope = _scope(best)
    else:
        best = None
        best_scope = max_val + 1
    
    # Try randomized greedy
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        rng_seed = 42
        n_samples = 100
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            seed = rng_seed + trial * 1000
            res = _randomized_greedy(n_rows, k, max_val, seed, n_samples=n_samples)
            if res is not None:
                s = _scope(res)
                if s < best_scope:
                    best_scope = s
                    best = [r[:] for r in res]
    
    # Try backtracking for potentially better results
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            seed = 7 + trial * 137
            res = _backtrack_rows(n_rows, k, max_val, deadline - 15, seed, max_nodes=150000)
            if res is not None:
                s = _scope(res)
                if s < best_scope:
                    best_scope = s
                    best = [r[:] for r in res]
    
    if best is None:
        # Fallback: very generous greedy
        max_val2 = total_diffs + 50
        rows = _greedy_all(n_rows, k, max_val2)
        if rows is not None:
            best = rows
        else:
            # Last resort: build each row independently with huge max
            best = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                    x += 1
                best.append(row)
    
    return best

# EVOLVE-BLOCK-END
