"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct
    across all rows. Minimize the largest entry (scope).
    
    Uses a row-level iterated local search:
    1. Greedy initialization.
    2. Iteratively try to reduce the scope by reconstructing rows that currently
       have the maximum value, forcing them to fit within a smaller scope.
    3. If stuck, shuffle rows to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # 1. Greedy Initialization
    rows, used_diffs = _greedy_construct(n_rows, k)
    current_scope = _get_scope(rows)
    best_rows = [row[:] for row in rows]
    best_scope = current_scope
    
    rng = random.Random(42)
    
    # 2. Iterative Improvement Loop
    while time.time() < deadline:
        # Identify rows contributing to the current scope
        max_rows = [i for i, r in enumerate(rows) if max(r) == current_scope]
        
        if not max_rows:
            break
        
        # Try to reduce the scope by one
        # Pick a random row from the max_rows to reconstruct
        target_scope = current_scope - 1
        
        # If target scope is too small (must be at least k), break
        if target_scope < k:
            break
            
        success = False
        # Try a few random max rows if the first one fails
        attempts = 0
        max_attempts = 10
        while not success and attempts < max_attempts and time.time() < deadline:
            idx = rng.choice(max_rows)
            row_to_replace = rows[idx]
            
            # Remove the row's diffs from used_diffs
            row_diffs = _get_diffs(row_to_replace)
            used_diffs -= row_diffs
            
            # Try to construct a new row with max <= target_scope
            new_row = _backtrack_row(k, target_scope, used_diffs, deadline)
            
            if new_row is not None:
                # Success: Update rows and used_diffs
                rows[idx] = new_row
                used_diffs |= _get_diffs(new_row)
                current_scope = target_scope
                best_rows = [row[:] for row in rows]
                best_scope = current_scope
                success = True
            else:
                # Failure: Revert
                used_diffs |= row_diffs
                attempts += 1
        
        if success:
            continue
            
        # If we couldn't reduce scope, try to shuffle to escape local minimum
        # Pick a random row (not necessarily max) and try to reconstruct it with same scope
        # This might free up differences that allow a max row to shrink later
        if time.time() < deadline:
            idx = rng.randint(0, n_rows - 1)
            row_to_replace = rows[idx]
            row_diffs = _get_diffs(row_to_replace)
            used_diffs -= row_diffs
            
            # Try to reconstruct with current scope (or slightly larger to allow flexibility? No, keep scope)
            # Actually, if we are stuck at M, we can try to reconstruct a row with max <= M.
            # If we find a *different* valid row, it might change the landscape.
            new_row = _backtrack_row(k, current_scope, used_diffs, deadline)
            
            if new_row is not None and new_row != row_to_replace:
                rows[idx] = new_row
                used_diffs |= _get_diffs(new_row)
            else:
                # Revert if no change or failure
                used_diffs |= row_diffs

    return best_rows


def _get_scope(rows):
    """Get the maximum entry across all rows."""
    return max(max(row) for row in rows)


def _get_diffs(row):
    """Get the set of positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows, n_rows, k):
    """Check if the rows form a valid difference triangle set."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff = row[j] - row[i]
                if diff <= 0:
                    return False
                if diff in used:
                    return False
                used.add(diff)
    return len(rows) == n_rows


def _greedy_construct(n_rows, k):
    """Greedy construction: build each row with smallest possible marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows, used


def _backtrack_row(k, max_scope, used_diffs, deadline):
    """
    Backtracking search to find a row of length k+1 starting at 0,
    strictly increasing, max <= max_scope, with no diffs in used_diffs.
    Returns the row or None.
    """
    # Heuristic: Try to fill the row
    # State: current_row
    # We want to minimize the max element, but we are constrained by max_scope.
    # To make it fast, we try candidates in increasing order.
    
    def backtrack(current_row):
        if time.time() > deadline:
            return None
        
        if len(current_row) == k + 1:
            return current_row[:]
            
        last_val = current_row[-1]
        # Next value must be > last_val and <= max_scope
        # Also, we need enough "space" for remaining elements.
        # Remaining elements: k + 1 - len(current_row)
        # Min possible next values: last_val + 1, last_val + 2, ...
        # Max possible value for the last element is max_scope.
        # So the last element (current_row[-1]) must be <= max_scope - (k + 1 - len(current_row) - 1)
        # i.e., current_row[-1] + (k + 1 - len(current_row)) <= max_scope + 1? 
        # Let's just check bounds in the loop.
        
        remaining = k + 1 - len(current_row)
        if remaining == 0:
            return current_row[:]
            
        # The smallest possible value for the last element is last_val + remaining
        # So if last_val + remaining > max_scope, we are dead.
        if last_val + remaining > max_scope:
            return None
            
        start_candidate = last_val + 1
        # End candidate: We need to leave room for 'remaining - 1' more elements after this one.
        # So x + (remaining - 1) <= max_scope => x <= max_scope - (remaining - 1)
        end_candidate = max_scope - (remaining - 1)
        
        for x in range(start_candidate, end_candidate + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            # Forward check: Are there enough unused diffs available?
            # This is hard to check efficiently. Just try.
            
            current_row.append(x)
            # Temporarily add to used? No, we can just check against the passed set.
            # Wait, the set `used_diffs` is fixed for this row construction (it represents diffs from OTHER rows).
            # We must ensure internal diffs don't conflict with `used_diffs`.
            # We don't need to update `used_diffs` locally because we are only checking against external conflicts.
            # However, we must ensure internal diffs are distinct (which they are by construction of differences from a set of unique numbers? No, differences within a row are always distinct if the numbers are distinct? 
            # Wait. a, b, c. a-b, b-c, a-c. a-c = (a-b)+(b-c). They are distinct values? 
            # Example: 0, 1, 3. Diffs: 1, 3, 2. Distinct.
            # Example: 0, 2, 4. Diffs: 2, 4, 2. NOT distinct.
            # So we MUST check if new_diffs has duplicates? 
            # No, new_diffs is a set, so duplicates are collapsed. 
            # But we need the *multiset* of differences to be distinct? 
            # The problem says "all positive differences ... are distinct over all rows".
            # This implies differences within a row must also be distinct from each other?
            # "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
            # Yes, if a row has duplicate differences, they are not distinct.
            # So we must check if len(new_diffs) == len(current_row) * (len(current_row)-1) // 2? 
            # No, new_diffs is the set of diffs for the NEW element x against all previous elements.
            # The total set of diffs for the row so far is the union of previous diffs and new_diffs.
            # We need to ensure no diff in new_diffs is already in the row's diff set.
            
            # Let's maintain a local set of diffs for the current row.
            
            # Optimization: Pass local_diffs as argument.
            
            return None # Placeholder, need to restructure
            
    # Restructure backtrack to track local_diffs
    
    def backtrack_with_local(current_row, local_diffs):
        if time.time() > deadline:
            return None
        
        if len(current_row) == k + 1:
            return current_row[:]
            
        last_val = current_row[-1]
        remaining = k + 1 - len(current_row)
        
        if last_val + remaining > max_scope:
            return None
            
        start_candidate = last_val + 1
        end_candidate = max_scope - (remaining - 1)
        
        for x in range(start_candidate, end_candidate + 1):
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if not valid:
                continue
                
            # Check if new_diffs has internal duplicates? 
            # Since x is fixed and a are distinct, x-a are distinct. 
            # So new_diffs size is len(current_row).
            
            current_row.append(x)
            local_diffs |= new_diffs
            
            res = backtrack_with_local(current_row, local_diffs)
            if res is not None:
                return res
                
            current_row.pop()
            local_diffs -= new_diffs
            
        return None
        
    return backtrack_with_local([0], set())


# EVOLVE-BLOCK-END
