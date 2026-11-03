# EVOLVE-BLOCK-START
import sys
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct over all rows.
    Minimize the largest entry (scope).
    
    Uses backtracking with forward checking and bitset optimization.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for scope
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    lb = max(lb, k)
    
    rows = []
    
    def solve(scope, deadline):
        """Try to construct n_rows rows with max entry <= scope."""
        used_bits = 0
        
        # Mask for valid differences 1..scope
        full_mask = (1 << (scope + 1)) - 2
        
        def backtrack(row_idx, current_row):
            """
            row_idx: which row we're constructing
            current_row: list of entries in current row
            """
            # Check timeout
            if time.time() > deadline:
                return False

            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            current_max = current_row[-1]
            min_next = current_max + 1
            max_next = scope
            
            # Try values in increasing order
            for x in range(min_next, max_next + 1):
                # Calculate differences
                new_bits = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        valid = False
                        break
                    new_bits |= (1 << d)
                
                if not valid:
                    continue
                
                # Forward Checking
                updated_used = used_bits | new_bits
                available_mask = full_mask & ~updated_used
                available_count = bin(available_mask).count('1')
                
                r_after = len(current_row) + 1
                rem_entries = k + 1 - r_after
                curr_row_rem_diffs = rem_entries * r_after + rem_entries * (rem_entries - 1) // 2
                future_rows_diffs = (n_rows - row_idx - 1) * diffs_per_row
                total_rem_diffs = curr_row_rem_diffs + future_rows_diffs
                
                if available_count < total_rem_diffs:
                    continue
                
                # Place x
                current_row.append(x)
                used_bits |= new_bits
                
                if backtrack(row_idx, current_row):
                    return True
                
                # Undo
                current_row.pop()
                used_bits &= ~new_bits
            
            return False
        
        return backtrack(0, [0])
    
    start_time = time.time()
    timeout = 110
    deadline = start_time + timeout
    
    for scope in range(lb, lb + 100):
        if time.time() > deadline:
            break
        
        # Clear rows before trying a new scope
        rows.clear()
        
        if solve(scope, deadline):
            return rows

    # Fallback: Greedy
    rows.clear()
    used = 0
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_bits = 0
            valid = True
            for a in row:
                d = x - a
                if (used >> d) & 1:
                    valid = False
                    break
                new_bits |= (1 << d)
            if valid:
                row.append(x)
                used |= new_bits
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
