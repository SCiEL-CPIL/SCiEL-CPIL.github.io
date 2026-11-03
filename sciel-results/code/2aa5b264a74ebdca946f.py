import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Lower bound: counting bound
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    # Greedy to get an initial upper bound
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return max(max(r) for r in rows), rows
    
    upper_bound, greedy_rows = greedy_solution()
    
    # If we are very close to the limit, just return greedy
    if time.time() > deadline - 5:
        return greedy_rows

    # We will search for m from lower_bound to upper_bound
    sys.setrecursionlimit(10000)
    
    def solve_for_m(m):
        rows = []
        
        def dfs_opt(row_idx, current_row, used_bits):
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                res = dfs_opt(row_idx + 1, [0], used_bits)
                if res is not None:
                    return res
                rows.pop()
                return None
            
            last = current_row[-1]
            min_x = last + 1
            
            # Determine max_x
            max_x = m
            if row_idx == 0 and len(current_row) < k:
                # Need to leave room for remaining elements to reach m
                # Remaining elements to place: k - len(current_row)
                # One of them is m (the last one).
                # So we need k - len(current_row) - 1 elements strictly between x and m.
                # So x + (k - len(current_row) - 1) < m
                # x < m - (k - len(current_row) - 1)
                # x <= m - (k - len(current_row))
                max_x = m - (k - len(current_row))
            
            # If we are at the last position of the first row, x MUST be m
            if row_idx == 0 and len(current_row) == k:
                x = m
                mask = 0
                valid = True
                for p in current_row:
                    d = x - p
                    if d <= 0 or d > m:
                        valid = False
                        break
                    mask |= (1 << d)
                
                if not valid or (used_bits & mask):
                    return None
                
                new_used_bits = used_bits | mask
                current_row.append(x)
                
                res = dfs_opt(row_idx, current_row, new_used_bits)
                if res is not None:
                    return res
                
                current_row.pop()
                return None
            
            for x in range(min_x, max_x + 1):
                mask = 0
                valid = True
                for p in current_row:
                    d = x - p
                    if d <= 0 or d > m:
                        valid = False
                        break
                    mask |= (1 << d)
                
                if not valid:
                    continue
                
                if used_bits & mask:
                    continue
                
                new_used_bits = used_bits | mask
                current_row.append(x)
                
                res = dfs_opt(row_idx, current_row, new_used_bits)
                if res is not None:
                    return res
                
                current_row.pop()
                
            return None
        
        # Initial call
        res = dfs_opt(0, [0], 0)
        return res
        
    for m in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break
        res = solve_for_m(m)
        if res is not None:
            return res
        
    return greedy_rows

# EVOLVE-BLOCK-END
