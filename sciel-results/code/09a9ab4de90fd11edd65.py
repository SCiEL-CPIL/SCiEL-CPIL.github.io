"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check all positive differences are distinct across all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _greedy_row(used, k, rng):
    """Build one row greedily with random tie-breaking."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row

def _try_build_row(used, k, max_val, rng, depth_limit=2000):
    """Try to build a row with max entry <= max_val using backtracking."""
    row = [0]
    
    def backtrack(pos, current_max):
        if pos == k + 1:
            return True
        if current_max > max_val:
            return False
        # Try values from current_max+1 to max_val
        start = row[-1] + 1
        for x in range(start, max_val + 1):
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used.update(new)
                if backtrack(pos + 1, x):
                    return True
                used.difference_update(new)
                row.pop()
        return False
    
    if backtrack(1, 0):
        return list(row)
    return None

def _greedy_rows(n_rows, k, rng):
    """Generate n_rows rows greedily."""
    used = set()
    rows = []
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

def _local_search_reduce(rows, n_rows, k, rng, deadline):
    """Try to reduce the scope by regenerating rows or adjusting marks."""
    best_rows = [list(r) for r in rows]
    best_scope = _compute_scope(best_rows)
    
    # Collect all used differences
    def get_used_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        current_scope = _compute_scope(best_rows)
        
        # Try to reduce scope: pick a row and try to rebuild it with a smaller max
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            
            # Target: try to fit this row within current_scope - 1, then - 2, etc.
            other_rows = [r for i, r in enumerate(best_rows) if i != row_idx]
            other_diffs = set()
            for row in other_rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        other_diffs.add(row[j] - row[i])
            
            # Try to rebuild row with max = current_scope - 1, - 2, ...
            for target_max in range(current_scope - 1, 0, -1):
                if time.time() > deadline:
                    break
                
                # Try multiple random attempts
                for attempt in range(10):
                    # Shuffle the order of trying values
                    row = [0]
                    used_in_row = set()
                    
                    def bt(pos, cur_max):
                        if pos == k + 1:
                            return True
                        if cur_max > target_max:
                            return False
                        start = row[-1] + 1
                        # Randomize order sometimes
                        candidates = list(range(start, target_max + 1))
                        if attempt % 3 == 0:
                            rng.shuffle(candidates)
                        for x in candidates:
                            new = {x - a for a in row}
                            if not new & other_diffs and not new & used_in_row:
                                row.append(x)
                                used_in_row.update(new)
                                if bt(pos + 1, x):
                                    return True
                                used_in_row.difference_update(new)
                                row.pop()
                        return False
                    
                    if bt(1, 0):
                        new_rows = [list(r) for r in best_rows]
                        new_rows[row_idx] = list(row)
                        new_scope = _compute_scope(new_rows)
                        if new_scope < best_scope:
                            best_rows = new_rows
                            best_scope = new_scope
                        break  # Move to next row_idx
        
        if _compute_scope(best_rows) < best_scope:
            best_scope = _compute_scope(best_rows)
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Phase 1: Greedy initialization
    best_rows = _greedy_rows(n_rows, k, rng)
    best_scope = _compute_scope(best_rows)
    
    # Phase 2: Multiple random greedy restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        rng2 = random.Random(restart * 1000 + 7)
        rows = _greedy_rows(n_rows, k, rng2)
        scope = _compute_scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
    
    # Phase 3: Local search to reduce scope
    best_rows, best_scope = _local_search_reduce(best_rows, n_rows, k, rng, deadline)
    
    # Phase 4: More aggressive search with different random seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        rng3 = random.Random(seed * 7919 + 13)
        rows = _greedy_rows(n_rows, k, rng3)
        scope = _compute_scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
            # Try local search on this better solution
            if time.time() < deadline - 20:
                best_rows, best_scope = _local_search_reduce(best_rows, n_rows, k, rng3, deadline)
    
    # Final validation
    if not _is_valid(best_rows):
        # Fallback to a known valid greedy solution
        best_rows = _greedy_rows(n_rows, k, random.Random(0))
    
    return best_rows

# EVOLVE-BLOCK-END
