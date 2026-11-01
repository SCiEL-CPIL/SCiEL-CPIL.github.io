"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a backtracking search with strong pruning.
    """
    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on M: total_diffs
    # Also, for a single row, the max element M_row must be at least k (trivial) 
    # but more tightly, the sum of k smallest distinct positive diffs is k(k+1)/2.
    # So M_row >= k(k+1)/2.
    # Global M >= max(M_row).
    
    # We will search for the minimum M.
    # We can do a binary search on M, or just incremental search.
    # Given the small size (7 rows, 5 cols), we can just search for the first valid solution
    # with increasing M bounds.
    
    # Optimization: Pre-compute valid rows? No, too many.
    
    # State:
    # rows: list of lists
    # used_diffs: set of integers
    
    # Pruning strategy:
    # 1. If current max > best found, prune.
    # 2. If we are building row i, and we have placed j elements, 
    #    the remaining elements must allow the remaining diffs to be placed.
    
    # Let's use a recursive backtracker that tries to build rows one by one.
    # To speed up, we can generate candidate rows for the next position.
    
    # Heuristic: 
    # The "greedy" approach often gets stuck. 
    # We will use a depth-first search with the following order:
    # - Try to keep the max entry as small as possible.
    
    # Since we need to minimize the global max, we can do a binary search on the global max M.
    # For a fixed M, is there a valid configuration?
    
    def is_valid_config(M):
        """
        Check if there exists a configuration with max entry <= M.
        Returns the configuration if found, else None.
        """
        used_diffs = set()
        rows = []
        
        # We need to place n_rows rows.
        # Each row is a subset of {0, ..., M} of size k+1, containing 0.
        # To make it easier, we can iterate through possible rows.
        
        # However, generating all possible rows is expensive.
        # Better: Build the row element by element.
        
        def backtrack(row_idx, current_row, current_used):
            if row_idx == n_rows:
                return True
            
            # If current row is complete, move to next row
            if len(current_row) == k + 1:
                # Check if current_row is valid (differences not in used_diffs)
                # We check this incrementally, but let's double check or just trust incremental.
                # Incremental check: when adding x, we checked x-a for all a in current_row.
                # So current_row is valid with respect to current_used.
                
                # Save row
                rows.append(list(current_row))
                
                # Recurse to next row
                if backtrack(row_idx + 1, [0], current_used):
                    return True
                
                rows.pop()
                return False
            
            # Determine next element
            last_val = current_row[-1]
            
            # The next element x must be > last_val and <= M.
            # Also, x - a for all a in current_row must not be in current_used.
            
            # Optimization: 
            # The smallest possible next value is last_val + 1.
            # The largest possible is M.
            
            # Pruning: 
            # If we are at the last element (index k), the value is determined? No.
            # But we must ensure that the remaining slots can be filled.
            
            # Let's just iterate x from last_val + 1 to M.
            # To improve speed, we can try to find a valid x quickly.
            
            for x in range(last_val + 1, M + 1):
                # Check constraints
                valid = True
                for a in current_row:
                    d = x - a
                    if d in current_used:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Add x
                new_used = current_used | {x - a for a in current_row}
                
                if backtrack(row_idx, current_row + [x], new_used):
                    return True
            
            return False
        
        # Start backtracking
        if backtrack(0, [0], set()):
            return rows
        return None

    # Binary search on M
    # Lower bound: total_diffs
    # Upper bound: We can estimate. For (7,5), 112 is optimal. 
    # Let's start high and go down, or start low and go up.
    # Since we want the *minimum* M, we can binary search.
    
    low = total_diffs
    # Upper bound heuristic: 
    # A greedy solution usually provides a good upper bound.
    # Let's try to find a quick greedy solution to set the upper bound.
    
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Try to build a row greedily
            # This might fail or be suboptimal, but gives an upper bound.
            # We'll just use a large number for high bound initially.
            pass
        # Actually, let's just set high = 200 for (7,5) case generally?
        # No, let's compute a safe upper bound.
        # M >= total_diffs.
        # Let's try M = total_diffs + 20.
        # If that fails, increase.
        
        # Better: Just run the search with an increasing M limit.
        # But binary search is faster if the range is known.
        
        # Let's use a heuristic upper bound:
        # For k=5, n=7, M=112.
        # total_diffs = 105.
        # Let's try from total_diffs to total_diffs + 50.
        
        # To be safe, let's just do a linear search from low to high, 
        # but with a step. Or just binary search between low and a high estimate.
        
        # High estimate: 
        # The greedy algorithm picks the smallest available numbers.
        # Let's simulate a very greedy run to get an upper bound.
        
        used_g = set()
        max_g = 0
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    if x > 500: # Safety break
                        return None
                    valid = True
                    for a in row:
                        if (x - a) in used_g:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_g.update({x - a for a in row[:-1]})
                        max_g = max(max_g, x)
                        break
                    x += 1
        return max_g

    upper_bound = greedy_solution()
    if upper_bound is None:
        upper_bound = total_diffs + 100
    
    # Ensure upper bound is at least lower bound
    if upper_bound < low:
        upper_bound = low
        
    # Binary search
    best_solution = None
    while low <= upper_bound:
        mid = (low + upper_bound) // 2
        sol = is_valid_config(mid)
        if sol is not None:
            best_solution = sol
            upper_bound = mid - 1
        else:
            low = mid + 1
            
    if best_solution is not None:
        return best_solution
    
    # Fallback if binary search failed (shouldn't happen)
    return is_valid_config(upper_bound)

# EVOLVE-BLOCK-END
