import sys
import time

# Increase recursion limit
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    # Calculate total differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for M
    low = total_diffs
    if low < 1:
        low = 1
        
    # Get an initial upper bound using the greedy algorithm
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    high = max(max(r) for r in rows)
    
    # If the greedy solution is already at the lower bound, return it
    if high == low:
        return rows
        
    best_rows = rows
    best_max = high
    
    start_time = time.time()
    TIME_LIMIT = 110
    
    # Try to find a solution with smaller M
    # Search from high-1 down to low
    for M in range(high - 1, low - 1, -1):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Try to construct solution for this M
        # We use backtracking
        used_diffs = set()
        solution_rows = []
        node_count = 0
        NODE_LIMIT = 100000 # Adjust if needed
        
        def backtrack(row_idx):
            nonlocal node_count
            if node_count > NODE_LIMIT:
                return False
            node_count += 1
            
            if row_idx == n_rows:
                return True
            
            def build_row(current_elements, local_diffs):
                nonlocal node_count
                if node_count > NODE_LIMIT:
                    return False
                node_count += 1
                
                if len(current_elements) == k + 1:
                    used_diffs.update(local_diffs)
                    solution_rows.append(list(current_elements))
                    if backtrack(row_idx + 1):
                        return True
                    # Backtrack
                    used_diffs.difference_update(local_diffs)
                    solution_rows.pop()
                    return False
                
                last = current_elements[-1]
                
                # Iterate possible next values
                # Optimization: We can bound the max x based on remaining slots?
                # Not easily.
                
                for x in range(last + 1, M + 1):
                    new_diffs = []
                    valid = True
                    for a in current_elements:
                        d = x - a
                        if d in used_diffs or d in local_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    current_elements.append(x)
                    local_diffs.update(new_diffs)
                    
                    if build_row(current_elements, local_diffs):
                        return True
                    
                    current_elements.pop()
                    local_diffs.difference_update(new_diffs)
                
                return False
            
            return build_row([0], set())
        
        if backtrack(0):
            best_rows = solution_rows
            best_max = M
            # We can continue to search for even smaller M
            # But if we find one, it's a valid improvement.
            
    return best_rows
