import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set (rulers) with n_rows rows and k marks per row (excluding 0).
    Returns a list of rows, each a list of k+1 integers.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Get a baseline solution using the greedy method
    def greedy_construct():
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

    baseline_rows = greedy_construct()
    best_scope = max(max(row) for row in baseline_rows)
    
    # If we already have a very good solution, we might skip search?
    # No, we want to improve.
    
    # 2. Backtracking search for smaller scopes
    # Total differences required per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_required = n_rows * diffs_per_row
    
    # Lower bound on scope M
    # The differences must be distinct integers from 1..M.
    # So M must be at least total_diffs_required.
    min_scope = total_diffs_required
    
    # We search for M from min_scope to best_scope - 1
    # If we find a solution for M, it is optimal (since we go up from min).
    
    def backtrack_search(M):
        """
        Attempts to find a valid set of rows with max entry <= M.
        Returns the list of rows if found, else None.
        """
        # State:
        # rows: list of lists, partially filled
        # used_diffs: set of integers
        # row_idx: current row index
        # col_idx: current column index in that row
        
        rows = [[0] + [None] * k for _ in range(n_rows)]
        used_diffs = set()
        
        # To speed up, we can pre-check if M is large enough.
        # M >= total_diffs_required is already checked.
        
        def is_valid_placement(r, c, val):
            # Check if placing val at rows[r][c] creates duplicate diffs
            # Diffs are val - rows[r][i] for i in 0..c-1
            for i in range(c):
                d = val - rows[r][i]
                if d in used_diffs:
                    return False
            return True

        def add_diffs(r, c, val):
            for i in range(c):
                used_diffs.add(val - rows[r][i])
        
        def remove_diffs(r, c, val):
            for i in range(c):
                used_diffs.remove(val - rows[r][i])

        def forward_check(r, c):
            # Check if remaining positions can be filled
            # Remaining positions:
            # In current row r: columns c+1 to k
            # In rows r+1 to n_rows-1: columns 0 to k (but col 0 is 0, so 1 to k)
            
            # Simple bound:
            # Available diffs = M - len(used_diffs)
            # Required diffs:
            # For current row r, remaining cols c+1..k will generate diffs.
            # Specifically, if we place values v_{c+1}...v_k, the new diffs are v_j - v_i for i<j in the full row.
            # This is complex to count exactly without knowing values.
            # However, the maximum number of new diffs we *can* generate is bounded by the range.
            # A simpler heuristic:
            # The remaining "slots" for differences is M - len(used_diffs).
            # The minimum number of differences we *must* generate is:
            # Diffs already in row r: c*(c+1)/2? No, we have placed c values (indices 0..c-1 are filled, index c is being considered? No, c is the index we are filling).
            # Let's say we are filling index c.
            # Filled indices in row r: 0 to c-1.
            # Diffs generated so far in row r: (c-1)*c/2? No.
            # Pairs (i,j) with 0<=i<j<=c-1. Count is (c-1)*c/2.
            # Wait, the loop `for i in range(c)` in add_diffs adds c diffs.
            # So after filling index c, total diffs in row r is c*(c+1)/2?
            # If c=1 (filling index 1), diffs added: 1-0=1. Total diffs in row: 1. Formula: 1*2/2 = 1. Correct.
            # If c=2 (filling index 2), diffs added: 2-0, 2-1. Total diffs: 1 + 2 = 3. Formula: 2*3/2 = 3. Correct.
            
            # So, diffs used in row r so far (before placing c) is (c-1)*c/2.
            # Diffs used in previous rows 0..r-1 is r * diffs_per_row.
            # Total used_diffs size should be r * diffs_per_row + (c-1)*c/2.
            
            # We need to place values for:
            # Row r: indices c to k.
            # Rows r+1 to n_rows-1: indices 1 to k.
            
            # Minimum scope check:
            # The largest value in row r must be <= M.
            # The smallest possible value for index c is rows[r][c-1] + 1.
            # The largest possible value is M.
            
            # A stronger bound:
            # The number of available differences is M - len(used_diffs).
            # The number of differences *required* to complete the grid is:
            # rem_diffs = total_diffs_required - len(used_diffs)
            # If M - len(used_diffs) < rem_diffs, it's impossible.
            # This is equivalent to M < total_diffs_required, which is always false if we start M >= total_diffs_required.
            # So this check is trivial and always passes.
            
            # Better check:
            # Can we fit the remaining values?
            # For the current row r, we need to place k - c + 1 values (indices c..k).
            # They must be strictly increasing and <= M.
            # The smallest possible sequence is rows[r][c-1]+1, ..., rows[r][c-1] + (k-c+1).
            # If rows[r][c-1] + (k-c+1) > M, then it's impossible to fill this row.
            if c > 0:
                min_last_val = rows[r][c-1] + (k - c + 1)
                if min_last_val > M:
                    return False
            
            # For subsequent rows, we need to fit k values in 1..M.
            # Since they start at 0, the smallest max value is k.
            # If k > M, impossible (but M >= 112, k=7, so always true).
            
            return True

        def search(r, c):
            if time.time() > deadline:
                return None
            
            if r == n_rows:
                # All rows filled
                return [row[:] for row in rows]
            
            if c == k + 1:
                # Move to next row
                return search(r + 1, 1) # Start next row at index 1 (index 0 is 0)
            
            # Determine range of values for rows[r][c]
            if c == 1:
                # First value in row (after 0)
                start_val = 1
            else:
                start_val = rows[r][c-1] + 1
            
            # Upper bound: M
            # Also, we must leave enough room for the rest of the row.
            # Max value for index c is M - (k - c)
            # Because we need k-c more values strictly larger than this.
            max_val = M - (k - c)
            
            # Iterate possible values
            # To improve pruning, we might want to try values that leave more room or satisfy constraints better.
            # Usually, smaller values are preferred to keep the scope small, but we are fixing M.
            # Trying smaller values first is a good heuristic.
            
            for val in range(start_val, max_val + 1):
                if not is_valid_placement(r, c, val):
                    continue
                
                # Tentative placement
                rows[r][c] = val
                add_diffs(r, c, val)
                
                if forward_check(r, c):
                    result = search(r, c + 1)
                    if result is not None:
                        return result
                
                # Backtrack
                remove_diffs(r, c, val)
                rows[r][c] = None
            
            return None

        # Start search at row 0, col 1
        return search(0, 1)

    # Iterate M from min_scope to best_scope - 1
    for M in range(min_scope, best_scope):
        if time.time() > deadline:
            break
        
        # Optional: Add a small randomization or restart if stuck?
        # For now, deterministic.
        
        solution = backtrack_search(M)
        if solution is not None:
            return solution

    # If no better solution found, return baseline
    return baseline_rows
