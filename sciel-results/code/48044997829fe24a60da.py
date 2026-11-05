import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    best_solution = None
    best_scope = float('inf')
    
    def greedy_construct(n_rows, k):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    rows = []
    used_diffs_bitset = 0
    
    def backtrack(row_idx, current_row, current_max):
        nonlocal best_solution, best_scope, used_diffs_bitset
        
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            if current_max < best_scope:
                best_scope = current_max
                best_solution = [list(r) for r in rows]
            return True
        
        if current_max >= best_scope:
            return False
        
        if not current_row:
            current_row = [0]
            current_max = 0
            return backtrack(row_idx, current_row, current_max)
            
        if len(current_row) == k + 1:
            if row_idx > 0 and current_max < rows[row_idx-1][-1]:
                return False
                
            new_diffs_bitset = 0
            for i in range(len(current_row)):
                for j in range(i):
                    d = current_row[i] - current_row[j]
                    if (used_diffs_bitset >> d) & 1:
                        return False
                    new_diffs_bitset |= (1 << d)
                    
            used_diffs_bitset |= new_diffs_bitset
            rows.append(current_row)
            
            success = backtrack(row_idx + 1, [], 0)
            
            rows.pop()
            used_diffs_bitset ^= new_diffs_bitset # Remove the differences
            
            return success
        
        for next_val in range(current_max + 1, best_scope):
            valid = True
            for prev_val in current_row:
                d = next_val - prev_val
                if (used_diffs_bitset >> d) & 1:
                    valid = False
                    break
            
            if valid:
                new_row = current_row + [next_val]
                if backtrack(row_idx, new_row, next_val):
                    return True
        
        return False

    backtrack(0, [], 0)
    
    if best_solution is None:
        return greedy_construct(n_rows, k)
        
    return best_solution
