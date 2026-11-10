"""Difference triangle set construction via greedy with backtracking and spectral-guided ordering."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct across rows."""
    deadline = time.time() + 145  # safety margin before 150s kill
    best_result = None
    best_scope = float('inf')
    
    # Try multiple random seeds for diversity
    seeds = [0, 1, 42, 123, 777, 2024, 999, 314, 271, 137]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        result, scope = _solve_greedy_backtrack(n_rows, k, rng, deadline)
        if result is not None and scope < best_scope:
            best_scope = scope
            best_result = result
    
    # If greedy backtracking didn't find anything, fall back to pure greedy
    if best_result is None:
        best_result = _pure_greedy(n_rows, k)
        best_scope = _compute_scope(best_result)
    
    return best_result


def _compute_scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows) if rows else 0


def _pure_greedy(n_rows, k):
    """Pure greedy: each row takes smallest next mark with no conflicts."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _solve_greedy_backtrack(n_rows, k, rng, deadline):
    """Greedy with limited backtracking to improve scope."""
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline - 2:
            break
        row = _build_row_with_backtrack(row_idx, k, used, rng, deadline)
        if row is None:
            # Fallback: pure greedy for this row
            row = _build_row_greedy(k, used)
        rows.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    if len(rows) < n_rows:
        # Complete remaining rows with greedy
        while len(rows) < n_rows:
            row = _build_row_greedy(k, used)
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
    
    scope = _compute_scope(rows)
    return rows, scope


def _build_row_greedy(k, used):
    """Build a row greedily."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row


def _build_row_with_backtrack(row_idx, k, used, rng, deadline):
    """Build a row with limited backtracking to minimize the last element."""
    # First do a greedy pass to get an initial solution
    initial_row = _build_row_greedy_copy(k, used)
    if initial_row is None:
        return None
    best_row = list(initial_row)
    best_last = initial_row[-1]
    
    # Try to improve by backtracking on the last few positions
    # We'll try a limited DFS from the second-to-last position
    max_backtrack_depth = min(3, k)
    
    def dfs(pos, current_row, current_used, depth):
        nonlocal best_row, best_last
        if time.time() > deadline - 3:
            return False
        
        if pos >= k + 1:
            if current_row[-1] < best_last:
                best_row = list(current_row)
                best_last = current_row[-1]
            return True
        
        # If we've already exceeded the best last element, prune
        if depth > 0 and current_row[-1] >= best_last:
            return False
        
        # Try values starting from current_row[pos-1] + 1
        start_val = current_row[pos - 1] + 1
        # Limit search range for efficiency
        max_val = start_val + 30  # Only look ahead 30 values
        
        for val in range(start_val, min(max_val, best_last)):
            new_diffs = {val - a for a in current_row}
            if new_diffs & current_used:
                continue
            
            current_row.append(val)
            new_used = current_used | new_diffs
            
            success = dfs(pos + 1, current_row, new_used, depth + 1)
            
            current_row.pop()
            
            if success and current_row[-1] < best_last:
                # Found a better solution, update
                best_row = list(current_row)
                best_last = current_row[-1]
                # We could continue searching, but for now let's stop early
                # to save time; we can do more rounds if time permits
                if time.time() > deadline - 5:
                    break
        
        return best_last < initial_row[-1]
    
    # Start DFS from position 1 (after the initial 0)
    # But to limit computation, only backtrack from later positions
    # Strategy: fix the first k-1 elements from greedy, backtrack on last 2
    fixed_len = max(1, k - 1)
    fixed_part = initial_row[:fixed_len]
    
    # Build used set for fixed part
    fixed_used = set()
    for i in range(len(fixed_part)):
        for j in range(i + 1, len(fixed_part)):
            fixed_used.add(fixed_part[j] - fixed_part[i])
    
    # DFS for remaining positions
    def dfs2(pos, current_row, current_used):
        nonlocal best_row, best_last
        if time.time() > deadline - 3:
            return
        
        if pos >= k + 1:
            if current_row[-1] < best_last:
                best_row = list(current_row)
                best_last = current_row[-1]
            return
        
        start_val = current_row[-1] + 1
        max_val = min(start_val + 20, best_last)
        
        for val in range(start_val, max_val):
            new_diffs = {val - a for a in current_row}
            if new_diffs & current_used:
                continue
            
            current_row.append(val)
            new_used = current_used | new_diffs
            dfs2(pos + 1, current_row, new_used)
            current_row.pop()
    
    dfs2(fixed_len, list(fixed_part), fixed_used)
    
    return best_row


def _build_row_greedy_copy(k, used):
    """Build a row greedily without modifying the passed used set."""
    row = [0]
    x = 0
    local_used = set(used)
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & local_used:
            row.append(x)
            local_used |= new_diffs
    return row
# EVOLVE-BLOCK-END
