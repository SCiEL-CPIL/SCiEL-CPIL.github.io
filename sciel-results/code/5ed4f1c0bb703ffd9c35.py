"""Difference triangle set construction using backtracking with heuristic row ordering."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Total number of differences needed: n_rows * C(k+1, 2)
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # Simple counting bound: need at least total_diffs distinct differences
    # The differences range from 1 to max_entry
    # So max_entry >= total_diffs (but actually differences are a_j - a_i where a_0=0)
    # Actually the differences can be up to max_entry, and we need total_diffs distinct ones
    # A tighter bound: the differences are subsets of {1, ..., M} where M is the max entry
    
    # Start with a reasonable upper bound
    # Greedy gives some value, let's start a bit above the counting bound
    # Counting bound: if we use differences 1..M, we can have at most M distinct diffs
    # But each row uses n_diffs_per_row diffs, so we need M >= total_diffs
    # However, the structure constrains which differences can appear together
    
    # Let's use the greedy solution as an initial upper bound
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
        return rows, max(max(r) for r in rows)
    
    greedy_rows, greedy_max = greedy_construct()
    
    # Try to improve using backtracking search
    # We'll search for a solution with max_entry < greedy_max
    # Use iterative deepening on the max_entry bound
    
    def row_differences(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def backtrack_search(max_entry, row_idx, current_rows, used_diffs, deadline):
        """
        Backtracking search for the next row given used differences.
        Returns a valid row or None.
        """
        if time.time() > deadline:
            return None
        
        # If this is the last row, we can be more aggressive
        is_last_row = (row_idx == n_rows - 1)
        
        # Generate candidate rows of length k+1 starting with 0, all <= max_entry
        # Use a heuristic: try rows that use smaller differences first
        # But we need to ensure all differences are unused
        
        # For efficiency, we'll build the row incrementally
        # State: current partial row, next value to try
        best_row = None
        
        def build_row(partial_row, used_so_far, deadline):
            """Recursively build a row with the given partial row."""
            if time.time() > deadline:
                return None
            
            if len(partial_row) == k + 1:
                return partial_row[:]
            
            last_val = partial_row[-1]
            
            # Try next values starting from last_val + 1
            # Heuristic: try smaller values first to keep the max small
            for next_val in range(last_val + 1, max_entry + 1):
                # Check if adding next_val creates any difference already in used_diffs
                new_diffs = set()
                for a in partial_row:
                    d = next_val - a
                    if d in used_diffs or d in used_so_far:
                        break
                    new_diffs.add(d)
                else:
                    # All differences are new
                    new_used = used_so_far | new_diffs
                    partial_row.append(next_val)
                    result = build_row(partial_row, new_used, deadline)
                    partial_row.pop()
                    if result is not None:
                        return result
            
            return None
        
        result = build_row([0], set(), deadline)
        return result
    
    # Try to find a solution with progressively smaller max_entry
    # Start just below the greedy solution and go down
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Try values from greedy_max - 1 down to a reasonable lower bound
    # Lower bound: we need total_diffs distinct differences, each <= max_entry
    # But also, the structure of rows constrains things
    # A simple lower bound: max_entry >= total_diffs (since we need that many distinct positive integers)
    # Actually, differences are positive integers, so we need at least total_diffs distinct values in {1, ..., max_entry}
    # So max_entry >= total_diffs
    
    lower_bound = max(total_diffs, k)  # at least k because we need k+1 distinct values starting from 0
    
    # Try to improve from greedy_max - 1 downward
    for target_max in range(greedy_max - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random orderings of rows for diversity
        # For now, just try the natural order
        rows = []
        used_diffs = set()
        success = True
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                success = False
                break
            
            row = backtrack_search(target_max, row_idx, rows, used_diffs, deadline)
            if row is None:
                success = False
                break
            
            rows.append(row)
            used_diffs |= row_differences(row)
        
        if success and all(len(r) == k + 1 for r in rows):
            # Verify the solution
            all_diffs = set()
            valid = True
            for r in rows:
                if r[0] != 0:
                    valid = False
                    break
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                actual_max = max(max(r) for r in rows)
                if actual_max < best_max:
                    best_rows = rows
                    best_max = actual_max
                # If we found a solution at target_max, we can try to go lower
                # But let's continue to see if we can do even better
                # Actually, we're iterating downward, so we'll try lower targets next
    
    return best_rows
# EVOLVE-BLOCK-END
