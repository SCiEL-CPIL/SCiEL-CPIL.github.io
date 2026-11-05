import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    lower_bound = total_diffs_needed

    def get_greedy_rows():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < num_elements:
                last = row[-1]
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                    if x > 1000:
                        return None
                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    greedy_sol = get_greedy_rows()
    if greedy_sol is None:
        return [] # Should not happen
        
    initial_ub = max(row[-1] for row in greedy_sol)
    
    def solve_for_M(M):
        used_diffs = set()
        rows = []
        current_row = [0]
        
        def backtrack(row_idx, pos_in_row):
            if time.time() - start_time > time_limit:
                return False
            
            if pos_in_row == num_elements:
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        row_diffs.add(current_row[j] - current_row[i])
                
                if row_diffs & used_diffs:
                    return False
                
                old_used = used_diffs.copy()
                old_rows_len = len(rows)
                
                used_diffs.update(row_diffs)
                rows.append(current_row[:])
                
                if row_idx == n_rows - 1:
                    return True
                else:
                    res = backtrack(row_idx + 1, 1)
                    if res:
                        return True
                    
                    rows.pop()
                    used_diffs.clear()
                    used_diffs.update(old_used)
                    return False
            
            last_val = current_row[-1]
            remaining_slots = num_elements - pos_in_row - 1
            
            if last_val + 1 > M - remaining_slots:
                return False

            for x in range(last_val + 1, M - remaining_slots + 1):
                valid = True
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                if backtrack(row_idx, pos_in_row + 1):
                    return True
                current_row.pop()
            
            return False
        
        if backtrack(0, 1):
            return [r[:] for r in rows]
        return None

    lo = lower_bound
    hi = initial_ub
    found_solution = None
    
    while lo <= hi:
        if time.time() - start_time > time_limit:
            break
        mid = (lo + hi) // 2
        sol = solve_for_M(mid)
        if sol is not None:
            found_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1
            
    if found_solution is None:
        return greedy_sol
        
    return found_solution
