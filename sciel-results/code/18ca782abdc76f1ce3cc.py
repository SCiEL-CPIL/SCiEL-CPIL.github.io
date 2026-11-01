import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # The number of differences in one row is k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Get an initial upper bound using greedy construction
    initial_rows = _greedy_construct(n_rows, k)
    best_rows = initial_rows
    best_M = _max_entry(initial_rows)
    
    # Lower bound: at least total_diffs_needed distinct differences must fit in {1, ..., M}
    M_lower = max(total_diffs_needed, k)
    
    # Try to find a solution with smaller M than the greedy bound
    for M in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
        
        rows = _solve_packing(n_rows, k, M, deadline)
        
        if rows is not None:
            best_rows = rows
            best_M = M
            break
    
    # If we still have time, try to improve further with local search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        improved = _local_search(best_rows, n_rows, k, deadline)
        if improved is not None:
            improved_M = _max_entry(improved)
            if improved_M < best_M:
                best_rows = improved
                best_M = improved_M
    
    return best_rows

def _solve_packing(n_rows, k, M, deadline):
    """
    Try to find n_rows rows of k+1 integers from {0, ..., M} starting at 0,
    such that all pairwise differences are distinct.
    Uses randomized backtracking with constraint propagation.
    """
    # Precompute all valid single rows (subsets of size k+1 from {0,...,M} containing 0
    # with all internal differences distinct)
    # For efficiency, generate them lazily
    
    used_diffs = set()
    rows = []
    
    # Generate a pool of candidate rows
    candidates = _generate_candidates(k, M, n_rows * 50)
    
    # Shuffle candidates for randomized search
    random.shuffle(candidates)
    
    # Filter candidates that are internally valid
    valid_candidates = []
    for row in candidates:
        if _row_valid(row, k):
            valid_candidates.append(row)
    
    # Try to find a packing using greedy with backtracking
    def backtrack(idx, remaining_candidates):
        if time.time() > deadline - 3:
            return False
            
        if idx == n_rows:
            return True
            
        # Try candidates in random order
        shuffled = remaining_candidates[:]
        random.shuffle(shuffled)
        
        for row in shuffled:
            row_diffs = _get_diffs(row)
            
            # Check if this row conflicts with used_diffs
            if row_diffs & used_diffs:
                continue
            
            # Accept this row
            rows.append(row)
            used_diffs.update(row_diffs)
            
            # Recurse
            new_remaining = [r for r in shuffled if r is not row]
            if backtrack(idx + 1, new_remaining):
                return True
            
            # Backtrack
            rows.pop()
            used_diffs.difference_update(row_diffs)
            
        return False
    
    # Try multiple restarts with different shuffles
    for restart in range(30):
        if time.time() > deadline - 3:
            break
        
        random.seed(restart * 7919 + M)  # Deterministic per M and restart
        used_diffs.clear()
        rows.clear()
        
        if backtrack(0, valid_candidates):
            return list(rows)
    
    return None

def _generate_candidates(k, M, count):
    """
    Generate candidate rows: subsets of size k+1 from {0, ..., M} containing 0.
    Mix of systematic and random generation.
    """
    candidates = set()
    
    # Systematic: try rows with consecutive or near-consecutive values
    for start in range(0, M - k + 1, max(1, (M - k) // 20)):
        row = [start + i for i in range(k + 1)]
        if row[0] == 0:
            candidates.add(tuple(row))
        elif 0 in row:
            # Already contains 0, good
            candidates.add(tuple(row))
    
    # Random generation
    attempts = 0
    while len(candidates) < count and attempts < count * 10:
        attempts += 1
        # Pick k values from 1..M, add 0
        vals = random.sample(range(1, M + 1), k)
        vals.sort()
        row = (0,) + tuple(vals)
        candidates.add(row)
    
    return list(candidates)

def _row_valid(row, k):
    """Check if a row has all distinct internal differences."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    if not all(row[i] < row[i+1] for i in range(k)):
        return False
    
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _get_diffs(row):
    """Get all pairwise differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _local_search(rows, n_rows, k, deadline):
    """
    Try to reduce the maximum entry using local search.
    """
    current_rows = [list(r) for r in rows]
    current_max = _max_entry(current_rows)
    
    used_diffs = set()
    for row in current_rows:
        used_diffs.update(_get_diffs(row))
    
    best_rows = [list(r) for r in current_rows]
    best_max = current_max
    
    for step in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random row to modify
        idx = random.randint(0, n_rows - 1)
        
        # Remove its differences
        row_diffs = _get_diffs(current_rows[idx])
        used_diffs.difference_update(row_diffs)
        
        # Try to rebuild this row with a smaller max value
        new_row = _rebuild_row(k, used_diffs, current_max - 1)
        
        if new_row is not None:
            new_diffs = _get_diffs(new_row)
            current_rows[idx] = new_row
            used_diffs.update(new_diffs)
            
            new_max = _max_entry(current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in current_rows]
        else:
            # Put back the old row
            used_diffs.update(row_diffs)
    
    return best_rows if best_max < current_max else None

def _rebuild_row(k, used_diffs, max_val):
    """Try to construct a valid row with max entry <= max_val avoiding used_diffs."""
    if max_val < k:
        return None
    
    # Backtrack to build a row
    def dfs(row, current_diffs, last_val):
        if len(row) == k + 1:
            return list(row)
        
        for val in range(last_val + 1, max_val + 1):
            new_diffs = {val - a for a in row}
            if new_diffs & used_diffs or new_diffs & current_diffs:
                continue
            if len(new_diffs) != len(row):
                continue
            
            row.append(val)
            current_diffs.update(new_diffs)
            
            result = dfs(row, current_diffs, val)
            if result is not None:
                return result
            
            row.pop()
            current_diffs.difference_update(new_diffs)
        
        return None
    
    return dfs([0], set(), 0)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
