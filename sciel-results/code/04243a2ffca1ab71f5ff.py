"""Difference triangle set construction: (4,7) -> 8 marks per row, 4 rows.
Strategy: greedy construction with randomized restarts + local search (ILS-style).
We aim to minimize the maximum entry (scope)."""
import time
import random
import sys

_DEADLINE = 150.0  # wall-clock seconds before evaluator kills us

def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    used = set()
    for row in rows:
        d = _all_diffs(row)
        if d & used:
            return False
        used |= d
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_row(used, k, randomize=False, jitter=0):
    """Build a single row of k+1 marks starting at 0, greedily choosing smallest available.
    If randomize, add some randomness to explore alternatives."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if randomize:
            # Try a few candidates starting from x
            candidates = []
            for dx in range(0, jitter + 1):
                candidates.append(x + dx)
            random.shuffle(candidates)
            placed = False
            for cand in candidates:
                new_diffs = {cand - a for a in row}
                if not (new_diffs & used):
                    row.append(cand)
                    used |= new_diffs
                    x = cand
                    placed = True
                    break
            if not placed:
                # Fallback: keep incrementing
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
        else:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
    return row

def _greedy_construct(n_rows, k, randomize=False, jitter=0):
    """Construct n_rows rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, randomize=randomize, jitter=jitter)
        rows.append(row)
    return rows

def _try_reduce_last(rows, k):
    """Try to reduce the last mark of the last row by moving it down."""
    rows = [r[:] for r in rows]
    last_row = rows[-1]
    last_val = last_row[-1]
    # Try moving last_val down
    for new_val in range(last_val - 1, 0, -1):
        if new_val <= last_row[-2]:
            break
        test_row = last_row[:-1] + [new_val]
        test_rows = rows[:-1] + [test_row]
        if _is_valid(test_rows):
            rows[-1] = test_row
            return rows, True
    return rows, False

def _kick(rows, k):
    """Randomly perturb a row to escape local minima."""
    rows = [r[:] for r in rows]
    idx = random.randrange(len(rows))
    row = rows[idx]
    # Pick a random position (not first) and shift it
    pos = random.randrange(1, len(row))
    delta = random.choice([-1, 1]) * random.randint(1, 5)
    row[pos] += delta
    # Fix ordering if broken
    if pos > 0 and row[pos] <= row[pos - 1]:
        row[pos] = row[pos - 1] + 1
    for p in range(pos + 1, len(row)):
        if row[p] <= row[p - 1]:
            row[p] = row[p - 1] + 1
    return rows

def _local_search(rows, k, max_iters=2000):
    """Local search: try to reduce scope by adjusting individual marks."""
    rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    
    for _ in range(max_iters):
        improved = False
        # Try reducing each mark in each row
        for ri in range(len(rows)):
            row = rows[ri]
            for pi in range(len(row) - 1, 0, -1):  # from last to first (skip 0)
                old_val = row[pi]
                for new_val in range(old_val - 1, row[pi - 1], -1):
                    test_row = row[:pi] + [new_val] + row[pi+1:]
                    # Ensure strictly increasing
                    if pi + 1 < len(test_row) and test_row[pi] >= test_row[pi + 1]:
                        continue
                    test_rows = rows[:ri] + [test_row] + rows[ri+1:]
                    if _is_valid(test_rows):
                        new_scope = _scope(test_rows)
                        if new_scope <= best_scope:
                            rows = test_rows
                            best_scope = new_scope
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return rows, best_scope

def construct(n_rows, k):
    """Main entry point. Returns n_rows rows of k+1 increasing integers starting at 0."""
    start_time = time.time()
    
    # Phase 1: Greedy construction (deterministic)
    rows = _greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    
    # Phase 2: Local search on the greedy result
    ls_rows, ls_scope = _local_search(rows, k, max_iters=3000)
    if ls_scope < best_scope:
        best_rows = ls_rows
        best_scope = ls_scope
    
    # Phase 3: ILS with randomized greedy restarts
    elapsed = time.time() - start_time
    time_budget = _DEADLINE - elapsed - 5  # 5s safety margin
    if time_budget > 2:
        n_restarts = max(1, int(time_budget / 2.0))
        for restart in range(n_restarts):
            if time.time() - start_time > _DEADLINE - 5:
                break
            # Randomized greedy
            jitter = random.randint(0, 3)
            cand_rows = _greedy_construct(n_rows, k, randomize=True, jitter=jitter)
            cand_scope = _scope(cand_rows)
            
            # Local search
            cand_rows, cand_scope = _local_search(cand_rows, k, max_iters=1500)
            
            if cand_scope < best_scope:
                best_rows = cand_rows
                best_scope = cand_scope
            
            # Kick and re-search
            if time.time() - start_time > _DEADLINE - 5:
                break
            kicked = _kick(cand_rows, k)
            kicked, kicked_scope = _local_search(kicked, k, max_iters=1000)
            if kicked_scope < best_scope:
                best_rows = kicked
                best_scope = kicked_scope
    
    # Final safety check
    if not _is_valid(best_rows):
        # Fallback to simple greedy
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows
