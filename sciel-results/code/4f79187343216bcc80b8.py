import time
import math

def construct(n_rows, k):
    total_diffs_needed = n_rows * k * (k + 1) // 2
    M_min = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))
    
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

    initial_rows, M_max = greedy_construct()
    
    deadline = time.time() + 110
    
    def solve_for_M(M_val, local_deadline):
        used_mask = 0
        rows = [[0] for _ in range(n_rows)]
        count_used = 0
        
        def dfs(row_idx, mark_idx):
            nonlocal used_mask, count_used
            
            if time.time() > local_deadline:
                return None
            
            if row_idx == n_rows:
                return [r[:] for r in rows]
            
            if mark_idx == k + 1:
                return dfs(row_idx + 1, 1)
            
            # Calculate remaining differences needed
            # Current row: marks from mark_idx to k
            # Mark j creates j differences
            # Sum = sum_{j=mark_idx}^{k} j
            rem_row_diffs = (k + 1 + mark_idx) * (k - mark_idx + 1) // 2
            rem_rows_diffs = (n_rows - row_idx - 1) * k * (k + 1) // 2
            total_needed = rem_row_diffs + rem_rows_diffs
            
            if M_val - count_used < total_needed:
                return None
            
            start_val = rows[row_idx][mark_idx - 1] + 1 if mark_idx > 0 else 1
            end_val = M_val - (k - mark_idx)
            
            if start_val > end_val:
                return None
            
            for val in range(start_val, end_val + 1):
                conflict = False
                diffs_to_set = []
                for i in range(mark_idx):
                    diff = val - rows[row_idx][i]
                    if (used_mask >> diff) & 1:
                        conflict = True
                        break
                    diffs_to_set.append(diff)
                
                if conflict:
                    continue
                
                rows[row_idx][mark_idx] = val
                for d in diffs_to_set:
                    used_mask |= (1 << d)
                count_used += len(diffs_to_set)
                
                res = dfs(row_idx, mark_idx + 1)
                if res is not None:
                    return res
                
                for d in diffs_to_set:
                    used_mask &= ~(1 << d)
                count_used -= len(diffs_to_set)
                rows[row_idx][mark_idx] = 0
                
            return None
        
        return dfs(0, 1)

    per_m_limit = 5.0
    
    for M in range(M_min, M_max):
        if time.time() > deadline:
            break
        
        start_t = time.time()
        local_deadline = start_t + per_m_limit
        
        res = solve_for_M(M, local_deadline)
        
        if res is not None:
            return res
            
    return initial_rows
