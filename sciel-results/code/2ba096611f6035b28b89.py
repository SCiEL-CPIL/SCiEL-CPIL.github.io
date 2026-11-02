"""Difference triangle set (4,7): backtracking with MRV and time guard."""
import time
import sys
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # We need total_diffs distinct positive differences.
    # The differences range from 1 to scope.
    # So scope must be at least total_diffs.
    
    # Lower bound: the smallest possible scope is at least total_diffs
    # (since we need total_diffs distinct positive integers, each >= 1)
    # Actually, the differences are a_j - a_i for 0 <= i < j <= k.
    # The smallest difference is at least 1, and we need total_diffs distinct values.
    # So the maximum difference (which is at most the scope) must be at least total_diffs.
    # But also, the scope itself must be large enough to accommodate the structure.
    
    # Simple lower bound: scope >= total_diffs
    lb = total_diffs
    
    # Try to find a valid configuration with scope = lb, lb+1, ...
    # Use backtracking with MRV (Minimum Remaining Values) heuristic
    
    best_rows = None
    best_scope = None
    
    for scope in range(lb, lb + 200):  # try up to 200 above lower bound
        if time.time() > deadline - 5.0:
            break
        result = _backtrack_solve(n_rows, k, scope, start_time, deadline)
        if result is not None:
            best_rows = result
            best_scope = scope
            break
    
    if best_rows is None:
        # Fallback: greedy construction
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows


def _backtrack_solve(n_rows, k, scope, start_time, deadline):
    """
    Backtracking solver with MRV heuristic.
    We assign values to each position in each row.
    Row r has positions 0..k, with row[r][0] = 0.
    We need row[r][j] < row[r][j+1] for all j, and row[r][k] <= scope.
    All differences row[r][j] - row[r][i] for i < j must be distinct across all rows.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # used_diffs: set of differences already used
    used_diffs = set()
    
    # rows: list of lists, each row is a list of k+1 values
    # We'll build rows one position at a time
    # State: (row_idx, pos_idx, current_row_values)
    
    # For efficiency, we track for each row the set of differences already in that row
    # And we need to check against global used_diffs
    
    # MRV: at each step, choose the next (row, pos) to fill that has the fewest valid options
    # But since rows must be filled in order (increasing), we fill row by row, position by position.
    
    # Actually, let's fill all positions of row 0, then row 1, etc.
    # Within a row, positions must be increasing.
    
    # Let's use a recursive backtracking approach.
    # State: which row we're on, which position in that row, and the values chosen so far.
    
    # For MRV, we can look ahead: for the current position, count how many valid values exist.
    
    sys.setrecursionlimit(10000)
    
    def backtrack(row_idx, pos_idx, rows, used_diffs, row_diffs):
        """
        row_idx: current row index (0..n_rows-1)
        pos_idx: current position in the row (0..k)
        rows: list of rows, each a list of values
        used_diffs: set of all differences used so far
        row_diffs: list of sets, row_diffs[r] is the set of differences in row r so far
        """
        if time.time() > deadline:
            return None
        
        if row_idx == n_rows:
            return rows
        
        if pos_idx == 0:
            # Start of a new row, value must be 0
            rows.append([0])
            row_diffs.append(set())
            return backtrack(row_idx, 1, rows, used_diffs, row_diffs)
        
        # We need to choose rows[row_idx][pos_idx]
        # It must be > rows[row_idx][pos_idx - 1]
        # And <= scope
        # And all new differences must not conflict with used_diffs
        
        prev_val = rows[row_idx][pos_idx - 1]
        min_val = prev_val + 1
        max_val = scope
        
        # MRV: try values that have the fewest conflicts
        # For efficiency, just iterate from min_val to max_val
        # But we can prune: if a value creates a difference that's already used, skip it
        
        found = False
        for val in range(min_val, max_val + 1):
            if time.time() > deadline:
                return None
            
            # Check if all new differences are valid
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
            
            # Also check within this row: new differences must not duplicate existing ones in this row
            # But since we're checking against used_diffs which includes row_diffs, this is covered.
            # Wait, used_diffs is global, and row_diffs is per-row. Let me reconsider.
            
            # Actually, the constraint is: all differences across all rows must be distinct.
            # So used_diffs should contain all differences from all rows so far.
            # When we add a new value to the current row, the new differences are val - rows[row_idx][i] for i < pos_idx.
            # These must not be in used_diffs.
            
            # But we also need to ensure that within the same row, differences are distinct.
            # Since rows[row_idx] is strictly increasing, and we're adding val > all previous,
            # the new differences val - rows[row_idx][i] for i < pos_idx are all distinct from each other
            # (because rows[row_idx][i] are distinct). And they are new, so they don't conflict
            # with existing differences in this row (which involve smaller values).
            
            # So we just need to check against used_diffs.
            
            # Commit
            rows[row_idx].append(val)
            used_diffs.update(new_diffs)
            
            # Recurse
            if pos_idx + 1 <= k:
                result = backtrack(row_idx, pos_idx + 1, rows, used_diffs, row_diffs)
            else:
                # Finished this row
                result = backtrack(row_idx + 1, 0, rows, used_diffs, row_diffs)
            
            if result is not None:
                return result
            
            # Undo
            rows[row_idx].pop()
            for d in new_diffs:
                used_diffs.discard(d)
        
        return None
    
    # Initial call
    rows = []
    row_diffs = []
    result = backtrack(0, 0, rows, used_diffs, row_diffs)
    
    if result is not None:
        # Verify
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
        # Check strictly increasing
        for i in range(k):
            if row[i + 1] <= row[i]:
                return False
    
    return True


def _greedy_construct(n_rows, k):
    """Greedy fallback construction."""
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
