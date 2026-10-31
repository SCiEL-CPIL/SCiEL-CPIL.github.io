"""Difference triangle set (4,7): backtracking with forward checking and MRV."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across rows.
    Minimize the largest entry.
    
    Strategy: Backtracking search with forward checking and dynamic variable
    ordering (MRV - minimum remaining values) to construct rows that minimize
    the maximum difference by prioritizing high-impact placements.
    """
    deadline = time.time() + 135  # 135s to leave safety margin before 150s
    
    def _row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _total_diffs_needed():
        """Total number of distinct differences needed."""
        return n_rows * k * (k + 1) // 2
    
    def _greedy_construct(n_rows, k):
        """Greedy construction: each row takes smallest next mark with no conflict."""
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
    
    def _backtrack_construct(n_rows, k, max_scope):
        """
        Backtracking with forward checking to construct n_rows rows
        with all differences distinct and max entry <= max_scope.
        Uses MRV (minimum remaining values) heuristic.
        """
        used_diffs = set()
        rows = []
        
        def _get_available_values(current_row, scope_limit):
            """Get available values for next position, considering increasing order."""
            if not current_row:
                return list(range(1, scope_limit + 1))
            last = current_row[-1]
            return list(range(last + 1, scope_limit + 1))
        
        def _forward_check(current_row, candidate, used_diffs, remaining_positions):
            """
            Check if adding candidate to current_row is feasible.
            Returns True if no immediate conflict and enough room for remaining positions.
            """
            new_diffs = {candidate - a for a in current_row}
            if new_diffs & used_diffs:
                return False
            
            # Check if we can still place remaining positions
            if remaining_positions > 0:
                # Need enough space for remaining positions
                # Each subsequent position must be at least 1 more than previous
                min_needed = candidate + remaining_positions
                if min_needed > scope_limit:
                    return False
            
            return True
        
        def _solve(row_idx, current_row, used_diffs, scope_limit):
            """Recursive backtracking solver."""
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            if not current_row:
                # Start new row
                current_row = [0]
            
            if len(current_row) == k + 1:
                # Row complete, move to next row
                rows.append(current_row)
                used_diffs_copy = used_diffs | _row_diffs(current_row)
                result = _solve(row_idx + 1, [], used_diffs_copy, scope_limit)
                if result is not None:
                    return result
                rows.pop()
                return None
            
            # Try to extend current row
            pos = len(current_row)
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # MRV: try values that leave most options for future positions
            # Try smaller values first to keep scope small
            candidates = []
            for val in range(last + 1, scope_limit + 1):
                if _forward_check(current_row, val, used_diffs, remaining - 1):
                    candidates.append(val)
            
            if not candidates:
                return None
            
            # Try candidates in order (smaller first for scope minimization)
            for val in candidates:
                new_diffs = {val - a for a in current_row}
                new_used = used_diffs | new_diffs
                new_row = current_row + [val]
                
                result = _solve(row_idx, new_row, new_used, scope_limit)
                if result is not None:
                    return result
            
            return None
        
        return _solve(0, [0], used_diffs, max_scope)
    
    def _binary_search_min_scope(n_rows, k, lo, hi):
        """Binary search for minimum scope."""
        best = None
        while lo <= hi and time.time() < deadline:
            mid = (lo + hi) // 2
            result = _backtrack_construct(n_rows, k, mid)
            if result is not None:
                best = result
                hi = mid - 1
            else:
                lo = mid + 1
        return best
    
    # Step 1: Greedy initialization
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = max(max(row) for row in greedy_rows)
    
    # Step 2: Try backtracking with binary search for smaller scope
    # Lower bound: need at least total_diffs distinct values in [1, scope]
    total_diffs = _total_diffs_needed()
    lower_bound = total_diffs  # At minimum, scope >= total_diffs
    upper_bound = greedy_scope
    
    # Try binary search
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # First try a few specific scopes
    test_scopes = []
    for s in range(max(lower_bound, 1), upper_bound + 1, 5):
        if s not in test_scopes:
            test_scopes.append(s)
    test_scopes.append(upper_bound)
    test_scopes.sort()
    
    for scope in test_scopes:
        if time.time() > deadline - 10:
            break
        result = _backtrack_construct(n_rows, k, scope)
        if result is not None:
            best_rows = result
            best_scope = scope
            break
    
    # Step 3: If we found a solution, try to improve with local search
    def _local_search(rows, k, n_rows, time_limit):
        """Local search to reduce scope."""
        start_time = time.time()
        current_rows = [list(r) for r in rows]
        current_scope = max(max(row) for row in current_rows)
        
        def _is_valid(rows):
            all_diffs = []
            for row in rows:
                all_diffs.extend(_row_diffs(row))
            return len(all_diffs) == len(set(all_diffs))
        
        if not _is_valid(current_rows):
            return current_rows, current_scope
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = current_rows[r_idx]
                for pos in range(k, 0, -1):  # Try to reduce larger positions first
                    old_val = row[pos]
                    for new_val in range(row[pos-1] + 1, old_val):
                        row[pos] = new_val
                        if _is_valid(current_rows):
                            new_scope = max(max(r) for r in current_rows)
                            if new_scope < current_scope:
                                current_scope = new_scope
                                improved = True
                                break
                        row[pos] = old_val
                        if time.time() - start_time > time_limit:
                            break
                    if time.time() - start_time > time_limit:
                        break
                if time.time() - start_time > time_limit:
                    break
        
        return current_rows, current_scope
    
    remaining = deadline - time.time()
    if remaining > 5:
        polished_rows, pol_scope = _local_search(best_rows, k, n_rows, min(remaining - 5, 60))
        if pol_scope < best_scope:
            best_rows = polished_rows
            best_scope = pol_scope
    
    return best_rows
