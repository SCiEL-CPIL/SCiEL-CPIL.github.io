"""Constraint-satisfaction search for difference triangle sets."""
import time
import random
import numpy as np
from itertools import combinations

def _valid(rows):
    """Check that all positive intra-row differences are distinct across rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    return max(max(row) for row in rows) if rows else 0

def _row_diffs(row):
    """Compute all positive differences within a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_row(used, k):
    """Build one row greedily given the set of already-used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row

def _backtrack_row(used, k, deadline, rng, max_depth=5000):
    """
    Backtracking search to build a row with k+1 elements starting at 0.
    Uses a degree heuristic: prefer candidates that consume fewer 'popular' differences.
    """
    row = [0]
    current_used = set(used)
    
    # Precompute: for each candidate value, which differences it would create
    # We'll do this incrementally
    
    def backtrack(row, current_used, depth):
        if len(row) == k + 1:
            return row[:]
        
        if time.time() > deadline or depth > max_depth:
            return None
        
        last = row[-1]
        # Try candidates in order of "degree" - prefer those that create differences
        # that are less "popular" (i.e., not in current_used and not likely to conflict)
        candidates = []
        for x in range(last + 1, last + 1 + 200):  # Look ahead up to 200
            new_diffs = {x - a for a in row}
            if new_diffs & current_used:
                continue
            # Score: number of differences that are "small" (more likely to conflict later)
            # Prefer candidates that create larger differences (less conflict-prone)
            score = sum(1 for d in new_diffs if d <= 50)  # Count small diffs
            candidates.append((score, x, new_diffs))
        
        # Sort by score (prefer fewer small diffs), then by value
        candidates.sort()
        
        for score, x, new_diffs in candidates:
            new_row = row + [x]
            new_used = current_used | new_diffs
            result = backtrack(new_row, new_used, depth + 1)
            if result is not None:
                return result
        
        return None
    
    result = backtrack(row, current_used, 0)
    if result is not None:
        return result
    
    # Fallback to greedy
    return _greedy_row(used, k)

def _column_wise_construct(n_rows, k, deadline, rng):
    """
    Column-wise scheduling: build rows position by position.
    At each position, assign values to all rows simultaneously, choosing
    assignments that minimize total difference conflicts.
    """
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    for pos in range(1, k + 1):
        # For each row, find candidate values
        row_candidates = []
        for r in range(n_rows):
            last = rows[r][-1]
            cands = []
            for x in range(last + 1, last + 1 + 300):
                new_diffs = {x - a for a in rows[r]}
                if new_diffs & used:
                    continue
                cands.append((x, new_diffs))
            row_candidates.append(cands)
        
        # Try to find an assignment that minimizes conflicts
        # Use greedy: for each row, pick the candidate that creates the least "popular" diffs
        best_assignment = None
        best_score = float('inf')
        
        for attempt in range(50):
            assignment = []
            total_score = 0
            temp_used = set(used)
            valid = True
            
            # Process rows in random order for diversity
            order = list(range(n_rows))
            if attempt > 0:
                rng.shuffle(order)
            
            for r in order:
                cands = row_candidates[r]
                if not cands:
                    valid = False
                    break
                
                # Score each candidate by how many small diffs it creates
                best_cand = None
                best_cand_score = float('inf')
                for x, new_diffs in cands:
                    if new_diffs & temp_used:
                        continue
                    # Count diffs that are small (more conflict-prone)
                    score = sum(1 for d in new_diffs if d <= 40)
                    if score < best_cand_score:
                        best_cand_score = score
                        best_cand = (x, new_diffs)
                
                if best_cand is None:
                    valid = False
                    break
                
                x, new_diffs = best_cand
                assignment.append((r, x, new_diffs))
                temp_used |= new_diffs
                total_score += best_cand_score
            
            if valid and total_score < best_score:
                best_score = total_score
                best_assignment = assignment
        
        if best_assignment is None:
            # Fallback: use greedy for remaining
            for r in range(n_rows):
                row = _greedy_row(used, k)
                rows[r] = row[:pos+1]
                used |= _row_diffs(rows[r])
            break
        
        # Apply best assignment
        for r, x, new_diffs in best_assignment:
            rows[r].append(x)
            used |= new_diffs
    
    return rows

def _local_search(rows, k, deadline, rng):
    """
    Joint local search over all rows.
    Moves: rebuild a random row with backtracking, swap two entries within a row,
    or perturb an entry.
    """
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    current_used = compute_used(current_rows)
    
    t = 1.0
    t_min = 0.01
    cooling = 0.998
    
    while time.time() < deadline:
        move_type = rng.random()
        if move_type < 0.5:
            # Rebuild a random row using backtracking
            idx = rng.randrange(n_rows)
            used_without = set(current_used)
            row = current_rows[idx]
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_without.discard(row[j] - row[i])
            
            # Use backtracking with a tight deadline
            row_deadline = min(time.time() + 0.5, deadline)
            new_row = _backtrack_row(used_without, k, row_deadline, rng)
            
            # Verify no conflicts
            new_used = set(used_without)
            valid = True
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in new_used:
                        valid = False
                        break
                    new_used.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
            
            new_scope = _scope([current_rows[i] if i != idx else new_row for i in range(n_rows)])
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < np.exp(-delta / max(t, 0.01)):
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        current_used.discard(row[j] - row[i])
                current_rows[idx] = new_row
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        current_used.add(new_row[j] - new_row[i])
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
        elif move_type < 0.75:
            # Swap two entries in a random row
            idx = rng.randrange(n_rows)
            row = current_rows[idx]
            if len(row) < 2:
                continue
            i, j = rng.sample(range(len(row)), 2)
            if i > j:
                i, j = j, i
            new_row = list(row)
            new_row[i], new_row[j] = new_row[j], new_row[i]
            new_row.sort()
            if new_row == row:
                continue
            row_diffs = set()
            valid = True
            for a in range(len(new_row)):
                for b in range(a+1, len(new_row)):
                    d = new_row[b] - new_row[a]
                    if d in row_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            if not valid:
                continue
            used_without = set(current_used)
            old_row = row
            for a in range(len(old_row)):
                for b in range(a+1, len(old_row)):
                    used_without.discard(old_row[b] - old_row[a])
            conflict = False
            for d in row_diffs:
                if d in used_without:
                    conflict = True
                    break
            if conflict:
                continue
            new_scope = _scope([current_rows[x] if x != idx else new_row for x in range(n_rows)])
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < np.exp(-delta / max(t, 0.01)):
                for a in range(len(old_row)):
                    for b in range(a+1, len(old_row)):
                        current_used.discard(old_row[b] - old_row[a])
                current_rows[idx] = new_row
                for d in row_diffs:
                    current_used.add(d)
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
        else:
            # Perturb: adjust one entry
            idx = rng.randrange(n_rows)
            row = current_rows[idx]
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            new_val = old_val + rng.randint(-5, 5)
            if new_val <= row[pos-1] or (pos < len(row)-1 and new_val >= row[pos+1]):
                continue
            new_row = list(row)
            new_row[pos] = new_val
            new_row.sort()
            if new_row == row:
                continue
            row_diffs = set()
            valid = True
            for a in range(len(new_row)):
                for b in range(a+1, len(new_row)):
                    d = new_row[b] - new_row[a]
                    if d in row_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            if not valid:
                continue
            used_without = set(current_used)
            old_row = row
            for a in range(len(old_row)):
                for b in range(a+1, len(old_row)):
                    used_without.discard(old_row[b] - old_row[a])
            conflict = False
            for d in row_diffs:
                if d in used_without:
                    conflict = True
                    break
            if conflict:
                continue
            new_scope = _scope([current_rows[x] if x != idx else new_row for x in range(n_rows)])
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < np.exp(-delta / max(t, 0.01)):
                for a in range(len(old_row)):
                    for b in range(a+1, len(old_row)):
                        current_used.discard(old_row[b] - old_row[a])
                current_rows[idx] = new_row
                for d in row_diffs:
                    current_used.add(d)
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
        
        t = max(t_min, t * cooling)
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple initial construction strategies
    strategies = [
        ("greedy", lambda: _greedy_rows(n_rows, k, rng)),
        ("backtrack", lambda: _backtrack_rows(n_rows, k, deadline, rng)),
        ("column_wise", lambda: _column_wise_construct(n_rows, k, deadline, rng)),
    ]
    
    for name, func in strategies:
        if time.time() >= deadline:
            break
        try:
            rows = func()
            if _valid(rows):
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k)
            best_rows.append(row)
            used |= _row_diffs(row)
        best_scope = _scope(best_rows)
    
    # Phase 2: Local search with multiple restarts
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Run local search for a portion of remaining time
        ls_deadline = min(time.time() + max(5, remaining * 0.3), deadline)
        candidate = _local_search(best_rows, k, ls_deadline, rng)
        scope = _scope(candidate)
        if scope < best_scope:
            best_scope = scope
            best_rows = candidate
        
        # Occasionally try fresh constructions
        if rng.random() < 0.4:
            strategy = rng.choice(["greedy", "backtrack", "column_wise"])
            if strategy == "greedy":
                fresh = _greedy_rows(n_rows, k, rng)
            elif strategy == "backtrack":
                fresh = _backtrack_rows(n_rows, k, deadline, rng)
            else:
                fresh = _column_wise_construct(n_rows, k, deadline, rng)
            
            if _valid(fresh):
                fresh_scope = _scope(fresh)
                if fresh_scope < best_scope:
                    best_scope = fresh_scope
                    best_rows = [list(r) for r in fresh]
    
    # Final validation
    if _valid(best_rows):
        return best_rows
    
    # Fallback: return greedy solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _greedy_rows(n_rows, k, rng):
    """Build rows using greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
        used |= _row_diffs(row)
    return rows

def _backtrack_rows(n_rows, k, deadline, rng):
    """Build rows using backtracking construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row_deadline = min(time.time() + 2.0, deadline)
        row = _backtrack_row(used, k, row_deadline, rng)
        rows.append(row)
        used |= _row_diffs(row)
    return rows
