import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of size k+1.
    Minimizes the largest entry using backtracking with symmetry breaking.
    """
    
    # Fallback greedy
    def greedy_construct(n, k):
        used = set()
        rows = []
        for _ in range(n):
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

    best_scope = [float('inf')]
    best_rows = [None]
    
    # rows is a list of lists, used to store the current path
    current_rows = [[] for _ in range(n_rows)]
    
    def backtrack(row_idx, cur_row, used_bits, global_max, prev_row_max):
        # Pruning: if global max is already >= best, stop
        if global_max >= best_scope[0]:
            return
        
        # If we are starting a new row, cur_row is [0]
        # prev_row_max is the max of the previous row, or 0 if first row.
        
        if len(cur_row) == k + 1:
            # Row is complete
            # Symmetry breaking: max of this row must be >= prev_row_max
            if cur_row[-1] < prev_row_max:
                return
                
            current_rows[row_idx] = cur_row
            
            if row_idx == n_rows - 1:
                # All rows complete
                best_scope[0] = global_max
                best_rows[0] = [list(r) for r in current_rows]
            else:
                # Next row
                # The next row's max must be >= cur_row[-1]
                backtrack(row_idx + 1, [0], used_bits, global_max, cur_row[-1])
            return
        
        last = cur_row[-1]
        
        # Determine range for x
        # x must be > last
        # x must be < best_scope[0]
        # Also, if this is the last element of the row (len(cur_row) == k),
        # we must ensure x >= prev_row_max (if we haven't already satisfied it? 
        # No, the check is done when row is complete).
        # But we can prune earlier: if len(cur_row) == k and last < prev_row_max,
        # we must pick x >= prev_row_max.
        
        start_x = last + 1
        if len(cur_row) == k and start_x < prev_row_max:
            start_x = prev_row_max
            
        # Also, we can bound x by best_scope[0]
        end_x = best_scope[0]
        
        for x in range(start_x, end_x):
            # Check differences
            valid = True
            new_diffs_bits = 0
            for a in cur_row:
                d = x - a
                if (used_bits >> d) & 1:
                    valid = False
                    break
                new_diffs_bits |= (1 << d)
            
            if not valid:
                continue
                
            new_used_bits = used_bits | new_diffs_bits
            new_global_max = max(global_max, x)
            
            backtrack(row_idx, cur_row + [x], new_used_bits, new_global_max, prev_row_max)
            
            # Optional: if we found a solution, we can break early?
            # No, we want the minimum. The loop tries smaller x first.
            # So the first solution found in this branch will have the smallest x for this step.
            # But other branches (different previous rows) might yield better global max.
            # So we must continue searching.

    backtrack(0, [0], 0, 0, 0)
    
    if best_rows[0] is None:
        return greedy_construct(n_rows, k)
        
    return best_rows[0]

