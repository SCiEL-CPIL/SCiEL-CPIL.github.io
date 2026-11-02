import sys
import random

def construct(n_rows, k):
    # Set recursion limit
    sys.setrecursionlimit(10000)
    
    # Lower bound based on counting
    lower_bound = n_rows * (k * (k + 1)) // 2
    
    # Helper to check feasibility for a given max value M
    def check_feasibility(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, col_idx):
            if row_idx == n_rows:
                return True
            
            if col_idx == 0:
                rows.append([0])
                return backtrack(row_idx, 1)
            
            if col_idx > k:
                return backtrack(row_idx + 1, 0)
            
            current_row = rows[row_idx]
            prev_val = current_row[col_idx - 1]
            
            # Heuristic: Try values in a random order? 
            # No, let's try increasing order first. 
            # If we want to minimize the max, we are fixing M.
            # To find *any* solution, order doesn't matter for correctness, 
            # but affects speed.
            
            # Pruning: 
            # The smallest possible next value is prev_val + 1.
            # The largest is M.
            
            # We can also prune based on the fact that we need to fit 
            # (k - col_idx) more elements in this row, and 
            # (n_rows - row_idx - 1) full rows.
            # This is complex. Let's stick to simple bounds.
            
            for x in range(prev_val + 1, M + 1):
                # Check conflicts with used_diffs
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Commit
                current_row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx, col_idx + 1):
                    return True
                
                # Undo
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        success = backtrack(0, 0)
        if success:
            return True, rows
        else:
            return False, []

    # Find the minimum M
    # Start with a high bound
    high = lower_bound + 50
    found, _ = check_feasibility(high)
    while not found:
        high *= 2
        if high > 1000:
            # Fallback to greedy if CSP is too hard (unlikely for 4,7)
            # But let's just return a greedy solution to be safe
            # Actually, the prompt says best known is 134. 
            # If we can't find it, we should return something valid.
            # Let's implement a greedy fallback.
            break
        found, _ = check_feasibility(high)
        
    if not found:
        # Greedy fallback
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

    # Binary search
    low = lower_bound
    while low < high:
        mid = (low + high) // 2
        found, _ = check_feasibility(mid)
        if found:
            high = mid
        else:
            low = mid + 1
            
    _, best_rows = check_feasibility(low)
    return best_rows
