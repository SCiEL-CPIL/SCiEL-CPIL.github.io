import time
import random
import sys

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for d in _compute_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, used=None, skip_prob=0.0):
    """
    Greedy initialization with optional random skipping to break deterministic traps.
    """
    if used is None:
        used = set()
    else:
        used = set(used)
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if skip_prob > 0 and random.random() < skip_prob and x < 300:
                    skipped = False
                    for extra in range(random.randint(1, 4)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        row.append(x)
                        used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _find_row_candidates(k, forbidden, max_scope, limit=5):
    """
    Yield up to 'limit' valid rows of length k+1 starting at 0, max <= max_scope,
    with diffs disjoint from 'forbidden'.
    """
    row = [0]
    used = set()
    count = 0
    
    def dfs():
        nonlocal count
        if count >= limit:
            return
        if len(row) == k + 1:
            yield list(row)
            count += 1
            return
        
        start_val = row[-1] + 1
        if start_val > max_scope:
            return
            
        # Heuristic: try smaller values first
        for x in range(start_val, max_scope + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & forbidden:
                continue
            if new_diffs & used:
                continue
            
            row.append(x)
            used.update(new_diffs)
            
            yield from dfs()
            
            used -= new_diffs
            row.pop()
            
            if count >= limit:
                return

    return list(dfs())

def _backtrack_construct_set(n_rows, k, max_scope, time_limit=10.0):
    """
    Try to construct a valid set of n_rows rows within max_scope.
    """
    start_time = time.time()
    used = set()
    rows = []
    
    def dfs(row_idx):
        if time.time() - start_time > time_limit:
            return False
        if row_idx == n_rows:
            return True
        
        candidates = _find_row_candidates(k, used, max_scope, limit=5)
        
        # Randomize order to break symmetry? 
        # For now, try in order.
        for cand in candidates:
            cand_diffs = _compute_diffs(cand)
            # Safety check
            if cand_diffs & used:
                continue
                
            rows.append(cand)
            used.update(cand_diffs)
            
            if dfs(row_idx + 1):
                return True
            
            # Backtrack
            rows.pop()
            used -= cand_diffs
            
        return False

    if dfs(0):
        return rows
    return None

def construct(n_rows, k):
    deadline = time.time() + 145.0
    
    # 1. Greedy Initialization
    best_rows = None
    best_scope = float('inf')
    
    # Try a few greedy variants
    for skip in [0.0, 0.1, 0.2]:
        rows, _ = _greedy_init(n_rows, k, skip_prob=skip)
        if _is_valid(rows):
            scope = _get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
    if best_rows is None:
        # Fallback
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0)
        return rows
        
    # 2. Scope Reduction via Backtracking
    # We know best_scope is valid. Try to find a solution with best_scope - 1.
    current_scope = best_scope
    
    while current_scope > 105 and time.time() < deadline - 5:
        target_scope = current_scope - 1
        # Allow more time for the search as we go deeper? 
        # Keep it constant for now.
        new_rows = _backtrack_construct_set(n_rows, k, target_scope, time_limit=5.0)
        
        if new_rows is not None:
            if _is_valid(new_rows):
                current_scope = target_scope
                best_rows = new_rows
            else:
                break
        else:
            # No solution found for target_scope, so no solution for lower scopes
            break
            
    return best_rows
