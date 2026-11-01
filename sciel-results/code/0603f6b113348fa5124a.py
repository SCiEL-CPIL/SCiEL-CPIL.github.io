import time
import random

def construct(n_rows, k):
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # Calculate lower bound for scope
    total_diffs_needed = n_rows * (k + 1) * k // 2
    min_scope = total_diffs_needed
    
    # Initial greedy solution
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

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    # Try a few random greedy runs to improve initial bound
    for _ in range(5):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            # Randomized choice? No, greedy is deterministic given the "first available" rule.
            # To randomize, we can shuffle the order of candidates?
            # But candidates are just integers.
            # We can try to pick a random valid x instead of the smallest.
            while len(row) < k + 1:
                candidates = []
                # Look ahead a bit?
                # Just picking the smallest is standard.
                # Let's stick to deterministic greedy for simplicity, 
                # but maybe try a different strategy?
                # Actually, the deterministic greedy is fine.
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
        scope = max(max(r) for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    row_diffs_count = (k + 1) * k // 2
    
    found_rows = []
    node_count = 0
    
    def solve_for_scope(M):
        nonlocal found_rows, node_count
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, last_val):
            nonlocal used_diffs, rows, found_rows, node_count
            
            node_count += 1
            if node_count % 1000 == 0:
                if time.time() - start_time > timeout:
                    return False
            
            if row_idx == n_rows:
                found_rows = [r[:] for r in rows]
                return True
            
            if len(current_row) == 0:
                current_row = [0]
                last_val = 0
                return backtrack(row_idx, current_row, last_val)
            
            if len(current_row) == k + 1:
                if rows:
                    if current_row < rows[-1]:
                        return False
                rows.append(current_row)
                return backtrack(row_idx + 1, [], 0)
            
            L = len(current_row)
            r = (k + 1) - L
            diffs_for_current_completion = r * L + (r * (r - 1)) // 2
            diffs_for_remaining_rows = (n_rows - row_idx - 1) * row_diffs_count
            total_needed = diffs_for_current_completion + diffs_for_remaining_rows
            
            if len(used_diffs) + total_needed > M:
                return False

            lower_bound = last_val + 1
            if rows and len(current_row) > 0:
                prev_row = rows[-1]
                if current_row < prev_row[:L]:
                    return False
                if current_row == prev_row[:L]:
                    lower_bound = max(lower_bound, prev_row[L])
            
            for x in range(lower_bound, M + 1):
                new_diffs = {x - y for y in current_row}
                if new_diffs & used_diffs:
                    continue
                
                new_used = used_diffs | new_diffs
                new_row = current_row + [x]
                
                if backtrack(row_idx, new_row, x):
                    used_diffs = new_used
                    return True
            
            return False
        
        return backtrack(0, [], 0)

    current_M = min_scope
    while current_M < best_scope:
        if time.time() - start_time > timeout:
            break
            
        found_rows = []
        
        success = solve_for_scope(current_M)
        
        if success:
            best_rows = found_rows
            best_scope = current_M
            break
        
        if time.time() - start_time > timeout:
            break
            
        current_M += 1
        
    return best_rows
