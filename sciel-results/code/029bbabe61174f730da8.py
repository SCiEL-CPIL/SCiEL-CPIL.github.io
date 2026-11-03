"""Difference triangle set (4,7): interval-graph packing with spectral-guided greedy and local search."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound: scope >= total_diffs
    lb = total_diffs
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Spectral-guided greedy with local search
    # Strategy 2: Randomized restarts with different heuristics
    # Strategy 3: Backtracking with MRV for small scopes
    
    # Try different scopes from lower bound upward
    # But first, try a good heuristic to get a reasonable upper bound
    
    # Phase 1: Get a good initial solution using multiple randomized greedy methods
    for attempt in range(5):
        if time.time() > deadline - 30.0:
            break
        rows = _spectral_greedy(n_rows, k, attempt)
        if rows is not None:
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [row[:] for row in rows]
    
    # Phase 2: Local search to improve
    if best_rows is not None:
        if time.time() < deadline - 20.0:
            improved = _local_search(best_rows, n_rows, k, start_time, deadline - 20.0)
            if improved is not None:
                scope = max(max(row) for row in improved)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = improved
    
    # Phase 3: Try backtracking for smaller scopes if time permits
    if time.time() < deadline - 15.0:
        target_scope = max(lb, best_scope - 20)
        for scope in range(target_scope, min(lb + 50, best_scope)):
            if time.time() > deadline - 10.0:
                break
            result = _backtrack_solve(n_rows, k, scope, start_time, deadline - 10.0)
            if result is not None:
                best_rows = result
                best_scope = scope
                break
    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _simple_greedy(n_rows, k)
    
    return best_rows


def _spectral_greedy(n_rows, k, seed):
    """
    Spectral-guided greedy construction.
    Uses eigenvector-inspired ordering to guide which differences to use.
    """
    random.seed(seed)
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Create a "spectral" ordering of differences
    # Instead of true spectral methods (too slow), use a heuristic ordering
    # that spreads differences more evenly
    
    used_diffs = set()
    rows = []
    
    for r in range(n_rows):
        row = _build_row_spectral(used_diffs, k, seed + r)
        if row is None:
            return None
        rows.append(row)
    
    return rows


def _build_row_spectral(used_diffs, k, seed):
    """Build a single row using spectral-inspired difference selection."""
    random.seed(seed)
    
    row = [0]
    pos = 0
    
    while pos < k:
        # Try to find the next value that minimizes "clustering" of differences
        best_val = None
        best_score = float('inf')
        
        # Look ahead: try values from a reasonable range
        prev = row[-1]
        search_start = prev + 1
        search_end = prev + 50  # Limit search for speed
        
        for val in range(search_start, search_end + 1):
            new_diffs = set()
            valid = True
            for a in row:
                d = val - a
                if d <= 0:
                    continue
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Score: prefer values that spread differences more evenly
            # Use variance of differences as a proxy for spectral separation
            if new_diffs:
                diffs_list = list(new_diffs)
                mean_d = sum(diffs_list) / len(diffs_list)
                variance = sum((d - mean_d) ** 2 for d in diffs_list) / len(diffs_list)
                # Also consider how "spread out" the value is from existing
                spread = val - prev
                score = variance + 0.1 * spread
            else:
                score = val - prev
            
            if score < best_score:
                best_score = score
                best_val = val
        
        if best_val is None:
            # Expand search range
            for val in range(search_end + 1, search_end + 200):
                new_diffs = set()
                valid = True
                for a in row:
                    d = val - a
                    if d <= 0:
                        continue
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                if new_diffs:
                    diffs_list = list(new_diffs)
                    mean_d = sum(diffs_list) / len(diffs_list)
                    variance = sum((d - mean_d) ** 2 for d in diffs_list) / len(diffs_list)
                    spread = val - prev
                    score = variance + 0.1 * spread
                else:
                    score = val - prev
                
                if score < best_score:
                    best_score = score
                    best_val = val
        
        if best_val is None:
            return None
        
        row.append(best_val)
        pos += 1
        
        # Update used_diffs
        for i in range(len(row) - 1):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    return row


def _local_search(rows, n_rows, k, start_time, deadline):
    """
    Local search: try to reduce the scope by rebuilding individual rows.
    """
    current_rows = [row[:] for row in rows]
    current_scope = max(max(row) for row in current_rows)
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        for r in range(n_rows):
            # Compute diffs used by other rows
            other_diffs = set()
            for r2 in range(n_rows):
                if r2 == r:
                    continue
                for i in range(len(current_rows[r2])):
                    for j in range(i + 1, len(current_rows[r2])):
                        other_diffs.add(current_rows[r2][j] - current_rows[r2][i])
            
            # Try to rebuild row r with smaller scope
            new_row = _rebuild_row_minimize(current_rows[r], other_diffs, k)
            if new_row is not None:
                new_scope = max(max(row) for row in [new_row] + [current_rows[r2] for r2 in range(n_rows) if r2 != r])
                if new_scope < current_scope:
                    current_rows[r] = new_row
                    current_scope = new_scope
                    improved = True
        
        if not improved:
            break
    
    return current_rows


def _rebuild_row_minimize(original_row, other_diffs, k):
    """Rebuild a row to minimize its maximum entry, given other rows' diffs."""
    original_max = max(original_row)
    
    # Try to find a row with smaller max
    for target_max in range(original_max - 1, 0, -1):
        row = _build_row_constrained(other_diffs, k, target_max)
        if row is not None:
            return row
    
    return original_row


def _build_row_constrained(other_diffs, k, max_val):
    """Build a row with max entry <= max_val, avoiding other_diffs."""
    row = [0]
    
    def backtrack(pos, current_row):
        if pos == k + 1:
            return current_row
        
        prev = current_row[-1]
        for val in range(prev + 1, max_val + 1):
            valid = True
            new_diffs = set()
            for a in current_row:
                d = val - a
                if d <= 0:
                    continue
                if d in other_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            current_row.append(val)
            result = backtrack(pos + 1, current_row)
            if result is not None:
                return result
            current_row.pop()
        
        return None
    
    result = backtrack(1, [0])
    return result


def _backtrack_solve(n_rows, k, scope, start_time, deadline):
    """Backtracking solver with MRV heuristic."""
    sys.setrecursionlimit(10000)
    used_diffs = set()
    
    def backtrack(row_idx, pos_idx, rows):
        if time.time() > deadline:
            return None
        
        if row_idx == n_rows:
            return rows
        
        if pos_idx == 0:
            rows.append([0])
            result = backtrack(row_idx, 1, rows)
            if result is not None:
                return result
            rows.pop()
            return None
        
        prev_val = rows[row_idx][pos_idx - 1]
        min_val = prev_val + 1
        max_val = scope
        
        for val in range(min_val, max_val + 1):
            if time.time() > deadline:
                return None
            
            valid = True
            new_diffs = []
            for i in range(pos_idx):
                d = val - rows[row_idx][i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            rows[row_idx].append(val)
            for d in new_diffs:
                used_diffs.add(d)
            
            if pos_idx + 1 <= k:
                result = backtrack(row_idx, pos_idx + 1, rows)
            else:
                result = backtrack(row_idx + 1, 0, rows)
            
            if result is not None:
                return result
            
            rows[row_idx].pop()
            for d in new_diffs:
                used_diffs.discard(d)
        
        return None
    
    rows = []
    result = backtrack(0, 0, rows)
    
    if result is not None:
        if _validate(result, n_rows, k):
            return result
    
    return None


def _validate(rows, n_rows, k):
    """Verify that the rows are valid."""
    if len(rows) != n_rows:
        return False
    
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        for i in range(k):
            if row[i + 1] <= row[i]:
                return False
    
    return True


def _simple_greedy(n_rows, k):
    """Simple greedy fallback construction."""
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
