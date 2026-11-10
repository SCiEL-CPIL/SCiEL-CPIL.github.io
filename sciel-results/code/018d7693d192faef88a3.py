"""Difference triangle set (4,7): construct rows of k+1 increasing ints starting at 0,
minimizing the largest entry while keeping all within-row differences distinct globally."""
import time
import random
import math
from itertools import combinations

def _scope_of(rows):
    return max(max(r) for r in rows)

def _is_valid(rows, k):
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _greedy_row(used, k, rng, max_val_hint=None):
    """Build one row greedily, starting at 0, choosing smallest next mark with no conflict.
    Returns (row, used_new) or None if stuck."""
    row = [0]
    used_local = set()
    x = 1
    max_val = max_val_hint if max_val_hint else 200
    while len(row) < k + 1:
        if x > max_val:
            return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used) and not (new_diffs & used_local):
            row.append(x)
            used_local |= new_diffs
            x += 1
        else:
            x += 1
    return row, used_local

def _greedy_all(n_rows, k, rng, max_val_hint=None):
    """Build all rows greedily in sequence."""
    used = set()
    rows = []
    for _ in range(n_rows):
        res = _greedy_row(used, k, rng, max_val_hint)
        if res is None:
            return None
        row, used_local = res
        rows.append(row)
        used |= used_local
    return rows

def _backtrack_row(used, k, start_val=0, max_val=None, depth=0):
    """Backtracking to build one row given already-used differences.
    Returns (row, used_local) or None."""
    if max_val is None:
        max_val = 200
    row = [0]
    used_local = set()
    
    def dfs(row, x_start):
        if len(row) == k + 1:
            return row, set(used_local)
        for x in range(x_start, max_val + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & used_local):
                row.append(x)
                used_local.update(new_diffs)
                res = dfs(row, x + 1)
                if res is not None:
                    return res
                row.pop()
                used_local.difference_update(new_diffs)
        return None
    
    res = dfs(row, 1)
    return res

def _improve_rows(rows, k, used, rng, max_iter=5000, time_limit=10.0):
    """Try to reduce the scope by regenerating rows with tighter bounds."""
    current_scope = _scope_of(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    start_time = time.time()
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        # Pick a random row to regenerate
        idx = rng.randint(0, len(rows) - 1)
        # Compute used diffs excluding this row
        other_used = set()
        for i, r in enumerate(rows):
            if i == idx:
                continue
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    other_used.add(r[b] - r[a])
        
        # Try to build a new row with smaller max
        target_max = best_scope - 1
        if target_max < k:
            target_max = k + 1
        
        res = _backtrack_row(other_used, k, max_val=target_max)
        if res is not None:
            new_row, new_used = res
            new_scope = max(best_scope, max(new_row))
            if new_scope < best_scope:
                rows[idx] = new_row
                best_rows = [r[:] for r in rows]
                best_scope = new_scope
                # Recompute global used
                used = set()
                for r in rows:
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            used.add(r[b] - r[a])
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct difference triangle set rows.
    Returns list of n_rows rows, each a list of k+1 increasing ints starting at 0.
    Minimizes the largest entry."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    # Phase 1: Get an initial solution via greedy
    initial = _greedy_all(n_rows, k, rng, max_val_hint=300)
    if initial is None:
        # Fallback: very loose greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                    x += 1
                else:
                    x += 1
            rows.append(row)
        initial = rows
    
    best_rows = [r[:] for r in initial]
    best_scope = _scope_of(best_rows)
    
    # Compute used set for best
    best_used = set()
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                best_used.add(r[j] - r[i])
    
    # Phase 2: Multiple restarts with randomized greedy + local improvement
    attempt = 0
    while time.time() < deadline - 5.0:
        attempt += 1
        
        # Try different strategies
        strategy = attempt % 3
        
        if strategy == 0:
            # Standard greedy with slight randomization
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                        x += 1
                    else:
                        x += 1
                rows.append(row)
            if rows:
                scope = _scope_of(rows)
                if scope < best_scope:
                    best_rows = [r[:] for r in rows]
                    best_scope = scope
                    best_used = set()
                    for r in best_rows:
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                best_used.add(r[j] - r[i])
        
        elif strategy == 1:
            # Randomized greedy: sometimes skip a candidate to explore
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new = {x - a for a in row}
                    if not (new & used):
                        # With small probability, skip this candidate
                        if rng.random() < 0.1 and x < 50:
                            x += 1
                            continue
                        row.append(x)
                        used |= new
                        x += 1
                    else:
                        x += 1
                if len(row) < k + 1:
                    success = False
                    break
                rows.append(row)
            if success and rows:
                scope = _scope_of(rows)
                if scope < best_scope:
                    best_rows = [r[:] for r in rows]
                    best_scope = scope
                    best_used = set()
                    for r in best_rows:
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                best_used.add(r[j] - r[i])
        
        else:
            # Backtracking with limited depth for each row
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                res = _backtrack_row(used, k, max_val=best_scope if best_scope < 200 else 150)
                if res is None:
                    success = False
                    break
                row, used_local = res
                rows.append(row)
                used |= used_local
            if success and rows:
                scope = _scope_of(rows)
                if scope < best_scope:
                    best_rows = [r[:] for r in rows]
                    best_scope = scope
                    best_used = set()
                    for r in best_rows:
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                best_used.add(r[j] - r[i])
        
        # Local improvement on current best
        if time.time() < deadline - 10.0:
            improved_rows, improved_scope = _improve_rows(
                best_rows, k, best_used, rng, 
                max_iter=2000, 
                time_limit=min(5.0, max(0.5, deadline - 10.0 - time.time()))
            )
            if improved_scope < best_scope:
                best_rows = improved_rows
                best_scope = improved_scope
                # Recompute used
                best_used = set()
                for r in best_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            best_used.add(r[j] - r[i])
        
        # Change seed for next iteration
        rng.seed(rng.randint(0, 2**31 - 1))
    
    # Final validation
    if not _is_valid(best_rows, k):
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                    x += 1
                else:
                    x += 1
            rows.append(row)
        best_rows = rows
    
    return best_rows
