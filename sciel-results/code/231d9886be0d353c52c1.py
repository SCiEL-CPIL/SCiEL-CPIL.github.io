import time
import sys

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

def construct(n_rows, k):
    # Baseline
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(row) for row in greedy_rows)
    
    # Deadline
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Lower bound for M
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    min_m = total_diffs
    
    # If min_m is already >= greedy_max, greedy is optimal (or close)
    if min_m >= greedy_max:
        return greedy_rows
        
    # Try to find a better solution
    # We iterate M from min_m to greedy_max - 1
    # But min_m can be large. For n_rows=4, k=7, num_diffs=28, total=112.
    # greedy_max for 4,7 is likely around 134-150.
    # So we try M = 112, 113, ...
    
    # Helper for backtracking
    def solve(rows, used_diffs_int, m_limit):
        if time.time() > deadline:
            return None
        
        if len(rows) == n_rows:
            return rows
        
        # Symmetry breaking: rows must be lexicographically sorted
        # Since we build them in order, we just need to ensure the new row 
        # is >= the previous one.
        # Actually, since the set of differences is global, the "value" of a row
        # is determined by its elements.
        # We will enforce that the new row is lexicographically greater than the last one.
        
        last_row = rows[-1] if rows else None
        
        # Build the next row
        # We use a recursive function to build the row
        def build_row(prefix, current_used):
            if time.time() > deadline:
                return None
            
            if len(prefix) == k + 1:
                # Check symmetry breaking
                if last_row is not None:
                    # Lexicographical comparison
                    if prefix < last_row:
                        return None
                # Success
                new_rows = rows + [prefix]
                return solve(new_rows, current_used, m_limit)
            
            last_val = prefix[-1]
            # Try next values
            # Heuristic: try smaller values first
            # Range: last_val + 1 to m_limit
            for x in range(last_val + 1, m_limit + 1):
                # Calculate new differences
                mask = 0
                valid = True
                for y in prefix:
                    d = x - y
                    # d is between 1 and m_limit
                    bit = 1 << d
                    if current_used & bit:
                        valid = False
                        break
                    mask |= bit
                
                if valid:
                    res = build_row(prefix + [x], current_used | mask)
                    if res is not None:
                        return res
            return None
        
        return build_row([0], used_diffs_int)

    # We need to convert used_diffs to int for the bitset approach
    # But greedy uses a set. We can just start with 0 for the backtracking search.
    
    # We try increasing M
    # Note: The search space is huge. We need to be careful.
    # For M=112, the bitset is 113 bits. Python int handles this fine.
    
    # Optimization: If we find a solution for M, it's valid for any M' > M.
    # So we just want the smallest M.
    
    for m in range(min_m, greedy_max):
        # Check time
        if time.time() > deadline:
            break
            
        # Initialize
        # We start with no rows, no used diffs
        res = solve([], 0, m)
        if res is not None:
            return res
            
    return greedy_rows
