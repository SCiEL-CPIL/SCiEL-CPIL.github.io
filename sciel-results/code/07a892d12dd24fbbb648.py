import time
import random
import math
from itertools import combinations
from collections import defaultdict

_DEADLINE = 145.0

def _row_diffs(row):
    """Generate all pairwise differences for a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _scope(rows):
    return max(r[-1] for r in rows)

def _backtracking_search(n_rows, k, max_scope, time_limit=60.0):
    """
    Backtracking search for difference triangle set.
    Build rows one at a time, elements within a row left to right.
    Prune aggressively using the used-differences set.
    """
    t0 = time.time()
    used_diffs = set()
    rows = []
    n_elements = k + 1  # each row has k+1 elements
    
    # Order of rows to build (try to build the "hardest" rows first for pruning)
    # For now, build in order 0, 1, 2, 3
    
    def backtrack(row_idx, pos_in_row, current_row):
        nonlocal used_diffs, rows, t0
        
        if time.time() - t0 > time_limit:
            return False
        
        if row_idx == n_rows:
            return True
        
        if pos_in_row == n_elements:
            # Row complete, commit it
            rows.append(list(current_row))
            # Update used_diffs
            for d in _row_diffs(current_row):
                used_diffs.add(d)
            
            # Recurse to next row
            if backtrack(row_idx + 1, 0, [0]):
                return True
            
            # Backtrack: remove this row
            for d in _row_diffs(current_row):
                used_diffs.discard(d)
            rows.pop()
            return False
        
        # Determine range for next element
        if pos_in_row == 0:
            # First element is always 0
            current_row.append(0)
            if backtrack(row_idx, 1, current_row):
                return True
            current_row.pop()
            return False
        
        # For subsequent elements, try values from current_row[-1] + 1 up to max_scope
        start_val = current_row[-1] + 1
        end_val = max_scope
        
        # Pruning: if we're at the last position, the value must be <= max_scope
        # Also, we need enough "room" for remaining elements
        remaining = n_elements - pos_in_row - 1
        # Minimum possible value for last element given strict increase
        # current_row[-1] + 1 + 2 + ... + (remaining+1) is the min last value
        min_last = current_row[-1] + sum(range(1, remaining + 2))
        if min_last > max_scope:
            return False
        
        for val in range(start_val, end_val + 1):
            # Check if adding val creates any conflict
            new_diffs = set()
            for prev in current_row:
                d = val - prev
                if d in used_diffs:
                    break
                new_diffs.add(d)
            else:
                # All diffs are new
                current_row.append(val)
                # Temporarily add to used_diffs for deeper pruning
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx, pos_in_row + 1, current_row):
                    return True
                
                # Undo
                for d in new_diffs:
                    used_diffs.discard(d)
                current_row.pop()
        
        return False
    
    # Try different max_scope values starting from lower bound
    # Lower bound: need at least 112 distinct differences, so max diff >= 112
    # But scope is the max element, and max diff <= scope
    # Simple lower bound: scope >= 112 (counting bound)
    
    for scope in range(112, 160):
        if time.time() - t0 > time_limit:
            break
        used_diffs = set()
        rows = []
        if backtrack(0, 0, []):
            return rows
    
    return None

def _targeted_descent(rows, k, time_limit=30.0):
    """
    Targeted local search to reduce the scope (max element).
    Focus on rows that contain the maximum element.
    Try to reduce large elements by finding alternative values that don't create conflicts.
    """
    t0 = time.time()
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    n_elements = k + 1
    
    # Build difference counter for efficient conflict checking
    def build_diff_map(rows):
        """Map each difference to the set of (row_idx, pos_i, pos_j) that produce it."""
        diff_map = defaultdict(list)
        for ri, row in enumerate(rows):
            for i in range(n_elements):
                for j in range(i + 1, n_elements):
                    d = row[j] - row[i]
                    diff_map[d].append((ri, i, j))
        return diff_map
    
    diff_map = build_diff_map(current_rows)
    
    def get_conflict_count(diff_map):
        return sum(len(v) - 1 for v in diff_map.values() if len(v) > 1)
    
    def try_reduce_element(row_idx, pos, new_val):
        """Try to set current_rows[row_idx][pos] to new_val. Returns (success, new_diff_map)."""
        row = current_rows[row_idx]
        old_val = row[pos]
        
        if new_val <= row[pos - 1] or (pos + 1 < n_elements and new_val >= row[pos + 1]):
            return False, diff_map
        
        # Compute old diffs involving pos
        old_diffs = []
        for j in range(n_elements):
            if j != pos:
                if j < pos:
                    old_diffs.append((old_val - row[j], row_idx, j, pos))
                else:
                    old_diffs.append((row[j] - old_val, row_idx, pos, j))
        
        # Check new diffs
        new_diffs = []
        for j in range(n_elements):
            if j != pos:
                if j < pos:
                    new_diffs.append((new_val - row[j], row_idx, j, pos))
                else:
                    new_diffs.append((row[j] - new_val, row_idx, pos, j))
        
        # Check if any new diff conflicts with existing diffs (excluding old ones)
        # Build a temporary diff map
        temp_diff_map = defaultdict(list)
        for d, entries in diff_map.items():
            # Remove old diffs involving (row_idx, pos)
            filtered = [e for e in entries if not (e[0] == row_idx and (e[1] == pos or e[2] == pos))]
            if filtered:
                temp_diff_map[d] = filtered
        
        # Add new diffs
        for d, ri, i, j in new_diffs:
            temp_diff_map[d].append((ri, i, j))
        
        # Check for conflicts
        conflicts = sum(len(v) - 1 for v in temp_diff_map.values() if len(v) > 1)
        if conflicts > 0:
            return False, diff_map
        
        # Apply
        row[pos] = new_val
        return True, temp_diff_map
    
    # Iteratively try to reduce the scope
    improvement = True
    while improvement and time.time() - t0 < time_limit:
        improvement = False
        current_scope = _scope(current_rows)
        
        # Find rows with the maximum element
        max_val = current_scope
        max_rows = [ri for ri in range(n_rows) if current_rows[ri][-1] == max_val]
        
        # Try to reduce elements in max_rows, starting from the largest positions
        for ri in max_rows:
            if time.time() - t0 > time_limit:
                break
            row = current_rows[ri]
            # Try to reduce each element from right to left
            for pos in range(n_elements - 1, 0, -1):
                if time.time() - t0 > time_limit:
                    break
                old_val = row[pos]
                # Try values smaller than old_val
                for new_val in range(row[pos - 1] + 1, old_val):
                    if pos + 1 < n_elements and new_val >= row[pos + 1]:
                        break
                    success, new_diff_map = try_reduce_element(ri, pos, new_val)
                    if success:
                        diff_map = new_diff_map
                        new_scope = _scope(current_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in current_rows]
                        improvement = True
                        break  # Restart loop with new state
        
        # Also try reducing non-maximum rows if it helps overall
        if not improvement:
            for ri in range(n_rows):
                if time.time() - t0 > time_limit:
                    break
                row = current_rows[ri]
                for pos in range(n_elements - 1, 0, -1):
                    if time.time() - t0 > time_limit:
                        break
                    old_val = row[pos]
                    for new_val in range(row[pos - 1] + 1, old_val):
                        if pos + 1 < n_elements and new_val >= row[pos + 1]:
                            break
                        success, new_diff_map = try_reduce_element(ri, pos, new_val)
                        if success:
                            diff_map = new_diff_map
                            new_scope = _scope(current_rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [list(r) for r in current_rows]
                            improvement = True
                            break
                    if improvement:
                        break
    
    return best_rows

def _greedy_valid(n_rows, k):
    """Generate a valid difference triangle set using greedy (always valid, may not be optimal)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def construct(n_rows, k):
    t_start = time.time()
    
    # Phase 1: Try backtracking search with increasing scope
    # Use a time budget for this phase
    bt_time = 40.0
    bt_rows = None
    
    # Start from the counting bound
    for scope in range(112, 145):
        if time.time() - t_start > bt_time:
            break
        bt_rows = _backtracking_search(n_rows, k, scope, time_limit=bt_time - (time.time() - t_start))
        if bt_rows is not None:
            break
    
    if bt_rows is not None:
        # Phase 2: Polish with targeted descent
        polished = _targeted_descent(bt_rows, k, time_limit=30.0)
        if _valid(polished):
            return polished
        return bt_rows
    else:
        # Fallback: greedy + targeted descent
        greedy_rows = _greedy_valid(n_rows, k)
        polished = _targeted_descent(greedy_rows, k, time_limit=40.0)
        if _valid(polished):
            return polished
        return greedy_rows
