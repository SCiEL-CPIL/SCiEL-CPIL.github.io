import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # State:
    # rows: list of lists, each list is a row of numbers
    # used_diffs: set of all differences used so far
    # best_solution: the best valid solution found (minimized scope)
    
    best_solution = None
    best_scope = float('inf')
    
    # We will construct rows one by one.
    # To improve performance, we can try to construct rows in a way that keeps numbers small.
    # A common heuristic for DTS is to try to keep the rows "balanced" or "tight".
    
    def is_valid_placement(row, idx, val, used_diffs, current_rows):
        """
        Check if placing 'val' at index 'idx' in 'row' is valid.
        1. It must be greater than the previous number in the row (if any).
        2. All differences between 'val' and previous numbers in 'row' must not be in 'used_diffs'.
        """
        if idx > 0 and val <= row[idx - 1]:
            return False
        
        # Check differences with previous numbers in this row
        for i in range(idx):
            diff = val - row[i]
            if diff in used_diffs:
                return False
        return True

    def backtrack(rows, used_diffs, row_idx, col_idx):
        """
        Recursive backtracking search.
        rows: list of current rows
        used_diffs: set of used differences
        row_idx: current row index
        col_idx: current column index (position in the row)
        """
        nonlocal best_solution, best_scope, deadline
        
        # Time check
        if time.time() > deadline:
            return False
        
        # If we have constructed all rows
        if row_idx == n_rows:
            # Calculate scope (max entry in the last row, since rows are increasing)
            # Actually, we need the max entry over ALL rows.
            # Since we build rows sequentially and don't constrain cross-row max strictly until the end,
            # we should track the global max or just compute it here.
            # For efficiency, we can assume the last row built might not have the max, 
            # but usually, later rows in a greedy/backtracking construction tend to be larger 
            # if we don't enforce a global bound. However, the problem asks to minimize the 
            # largest entry *overall*.
            
            # Let's compute the actual max
            current_max = 0
            for r in rows:
                if r:
                    current_max = max(current_max, r[-1])
            
            if current_max < best_scope:
                best_scope = current_max
                best_solution = [r[:] for r in rows]
            
            return True
        
        # If we are starting a new row
        if col_idx == 0:
            rows.append([0])
            # The first element is always 0. No differences generated yet for this row.
            return backtrack(rows, used_diffs, row_idx, 1)
        
        # If we are filling a position in the current row
        current_row = rows[row_idx]
        
        # Determine candidate values
        # The next value must be > current_row[col_idx - 1]
        min_val = current_row[col_idx - 1] + 1
        
        # We want to try small values first to keep scope low, but randomized to avoid local minima.
        # Generate a list of potential candidates.
        # How far should we look? 
        # If we look too far, it's expensive. If too short, we might miss solutions.
        # A dynamic window or a fixed reasonable window (e.g., 50-100) might work.
        # Let's try a window that expands if we fail, or just a fixed set of smallest valid candidates.
        
        candidates = []
        # We generate candidates starting from min_val upwards.
        # To avoid infinite loops or huge search spaces, we limit the number of candidates we consider in one go.
        # However, we need to ensure we don't miss a valid small number.
        # Strategy: Try the smallest valid numbers first, but with some randomization in order.
        
        # Let's generate the first N valid candidates.
        N_CANDIDATES = 20
        count = 0
        val = min_val
        
        # Optimization: If we already have a best_solution, we can prune values 
        # that would immediately exceed the best_scope found so far?
        # Not exactly, because a smaller value now might allow a larger value later, 
        # but generally keeping numbers small is good.
        
        while count < N_CANDIDATES:
            # Check validity
            if is_valid_placement(current_row, col_idx, val, used_diffs, rows):
                candidates.append(val)
                count += 1
            
            # Safety: Don't search forever for candidates
            if val > min_val + 10000:
                break
            val += 1
            
            if time.time() > deadline:
                return False

        if not candidates:
            # Dead end
            return False
        
        # Shuffle candidates to explore different branches
        # But we want to bias towards smaller values to find small scopes quickly.
        # A common trick: Sort by value, then shuffle the top few, or use a probability distribution.
        # Let's just shuffle them. Since we generated them in increasing order, 
        # shuffling ensures we don't always pick the smallest.
        random.shuffle(candidates)
        
        for val in candidates:
            # Place the value
            current_row.append(val)
            
            # Update used_diffs
            new_diffs = set()
            for i in range(col_idx):
                diff = val - current_row[i]
                new_diffs.add(diff)
            
            used_diffs |= new_diffs
            
            # Recurse
            if backtrack(rows, used_diffs, row_idx, col_idx + 1):
                return True
            
            # Backtrack
            used_diffs -= new_diffs
            current_row.pop()
            
            if time.time() > deadline:
                return False
        
        return False

    # Initial setup
    rows = []
    used_diffs = set()
    
    # Run the search
    # We might want to run multiple restarts with different random seeds if time permits
    # to escape local minima more effectively.
    
    num_restarts = 5
    for _ in range(num_restarts):
        if time.time() > deadline - 10: # Leave time for final checks
            break
        
        # Reset state for new restart
        rows = []
        used_diffs = set()
        
        # Optional: Randomize the initial row construction order? 
        # No, the structure is fixed: n_rows rows of k+1 elements.
        # We can randomize the candidate selection within the backtracking.
        
        backtrack(rows, used_diffs, 0, 0)
        
        # If we found a solution, we can try to improve it?
        # The backtracking already tries to find *a* solution. 
        # To minimize scope, we should ideally search for the *minimum* scope.
        # However, finding the exact minimum is hard. 
        # The current approach finds *a* solution. 
        # To improve the score (minimize scope), we should prefer solutions with smaller max entries.
        
        # Improvement: Instead of just returning True on the first valid solution,
        # we could continue searching for a better one if time permits?
        # Or, we can bias the candidate selection towards smaller numbers more strongly.
        
        # Let's stick with the current backtracking which finds a valid solution.
        # The randomization helps find *different* valid solutions, some of which might be smaller.
        
    # If no solution was found (should not happen for small n,k, but possible for large),
    # fall back to a simple greedy to ensure a valid output is returned.
    if best_solution is None:
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
        best_solution = rows

    return best_solution
# EVOLVE-BLOCK-END
