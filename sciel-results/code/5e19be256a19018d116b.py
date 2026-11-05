# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a CSP approach with MRV (Minimum Remaining Values) heuristic and
    bitset-based difference tracking for efficient pruning.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Greedy baseline for upper bound
    def greedy_construct(n_rows, k):
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    # CSP Solver with MRV
    def solve_csp_mrv(n_rows, k, max_entry, timeout, seed=0):
        """
        Solves the CSP using MRV heuristic.
        State:
        - rows: list of lists (partial rows)
        - used_bits: integer bitset of used differences
        - row_lengths: list of current lengths of each row (implicitly in rows)
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        
        # Initial state: n_rows empty rows (each starts with 0)
        initial_rows = [[0] for _ in range(n_rows)]
        initial_used = 0
        
        # We will use a recursive backtracking search.
        # To avoid deep recursion limits and overhead, we might implement iteratively 
        # or just use recursion with a small k (k is usually small in these problems, e.g., 7).
        # For k=7, depth is at most n_rows * 7.
        
        def dfs(rows, used_bits, depth):
            if time.time() > deadline:
                return None
            
            # Check if all rows are complete
            if all(len(r) == k + 1 for r in rows):
                return [r[:] for r in rows] # Return a copy
            
            # Find the row with Minimum Remaining Values (MRV)
            # For each incomplete row, calculate valid candidates for the next element
            best_row_idx = -1
            best_candidates = []
            min_candidate_count = float('inf')
            
            for i, row in enumerate(rows):
                if len(row) == k + 1:
                    continue
                
                last_val = row[-1]
                remaining_slots = k + 1 - len(row)
                
                # The next value x must be > last_val and <= max_entry - (remaining_slots - 1)
                min_x = last_val + 1
                max_x = max_entry - (remaining_slots - 1)
                
                if min_x > max_x:
                    # No possible value, this row is dead
                    return None
                
                candidates = []
                # Iterate through possible values
                # Optimization: We can break early if we find enough? No, we need the min count.
                # But we can stop if count exceeds current min_candidate_count? 
                # Actually, we just need to know which row has the *fewest*.
                # If we find a row with 0, we return None immediately.
                
                for x in range(min_x, max_x + 1):
                    # Check conflicts with used_bits
                    conflict = False
                    for a in row:
                        d = x - a
                        if (used_bits >> d) & 1:
                            conflict = True
                            break
                    if not conflict:
                        candidates.append(x)
                        
                count = len(candidates)
                if count == 0:
                    return None
                
                if count < min_candidate_count:
                    min_candidate_count = count
                    best_row_idx = i
                    best_candidates = candidates
                    
                    # If count is 1, we can't do better than this
                    if count == 1:
                        break
            
            if best_row_idx == -1:
                # Should not happen if not all complete
                return None
            
            # Heuristic for ordering candidates:
            # We want to find a valid solution quickly.
            # Usually, smaller values pack tighter, so try smaller x first?
            # Or randomize to escape local minima?
            # Let's try a mix: sort by value, but with a slight randomization or just reverse?
            # For minimization of max entry, compactness is key.
            # Let's try ascending order (smallest x first) as it tends to leave more room for others?
            # Actually, using small differences early blocks small differences for others.
            # But we are constrained by the global max entry.
            # Let's try ascending.
            
            # To add diversity in restarts, we can shuffle if count is large?
            # For now, deterministic ascending for the first attempt.
            # If we do multiple restarts, we can vary this.
            
            # Apply LCV (Least Constraining Value) heuristic?
            # This would require simulating the next step for each candidate, which is expensive.
            # Let's stick to MRV + Ascending Value for now.
            
            for x in best_candidates:
                # Calculate new used bits
                new_diff_bits = 0
                for a in rows[best_row_idx]:
                    d = x - a
                    new_diff_bits |= (1 << d)
                
                # Make a copy of the row to modify
                # Optimization: We can modify in place and backtrack
                old_row = rows[best_row_idx]
                rows[best_row_idx] = old_row + [x]
                
                result = dfs(rows, used_bits | new_diff_bits, depth + 1)
                if result is not None:
                    return result
                
                # Backtrack
                rows[best_row_idx] = old_row
            
            return None
        
        # Run the search
        return dfs(initial_rows, initial_used, 0)

    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Search for minimal M
    # We start from lb.
    # If lb is close to greedy_max, we check all.
    # If lb is much smaller, we might need to jump? No, linear is safer for correctness.
    
    # To speed up, we can check if the problem is "dense".
    # For small n_rows and k, the search space is small.
    
    for max_entry in range(lb, greedy_max + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Allocate time per M value
        # Give more time to smaller M values as they are harder
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
        
        # Try a few restarts with different random seeds if the first deterministic one fails?
        # The MRV + Ascending is deterministic.
        # If it fails, it's likely infeasible for this M? Not necessarily, backtracking might have missed a path due to heuristic order?
        # No, DFS is complete. If it returns None, no solution exists for this M with this row ordering?
        # Wait, the rows are distinguishable in the input (row 0, row 1...), but symmetric in constraints.
        # My MRV picks the "most constrained" row dynamically. This handles the symmetry.
        # So if DFS returns None, there is NO solution for this M.
        # Therefore, we don't need multiple restarts for a single M if the search is complete.
        # The search IS complete because it explores all valid assignments consistent with the constraints.
        
        # However, Python recursion might be slow.
        # If we have time, we can try.
        
        result = solve_csp_mrv(n_rows, k, max_entry, budget, seed=0)
        if result is not None:
            best_rows = result
            best_max = max_entry
            break
    
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
