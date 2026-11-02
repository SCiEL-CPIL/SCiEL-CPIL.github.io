import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    num_elements = k + 1
    
    # Lower bound for M
    lower_bound = (n_rows * k * (k + 1)) // 2
    
    # Global max to generate rows. 
    # Buffer of 100 above lower bound.
    max_gen_val = lower_bound + 100
    
    all_rows = []
    all_diffs = []
    
    def generate_rows(current_row, used_diffs_in_row, max_val):
        if len(current_row) == num_elements:
            all_rows.append(tuple(current_row))
            all_diffs.append(frozenset(used_diffs_in_row))
            return
        
        start = current_row[-1] + 1 if current_row else 0
        
        for x in range(start, max_val + 1):
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs_in_row or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                generate_rows(current_row + [x], used_diffs_in_row | new_diffs, max_val)
                
    generate_rows([0], set(), max_gen_val)
    
    # Sort rows by max element
    # This allows us to efficiently filter by M if we wanted, 
    # but we just use list comprehension.
    
    def find_solution(M):
        candidates = [(r, d) for r, d in zip(all_rows, all_diffs) if max(r) <= M]
        
        def backtrack(idx, used_global_diffs, selected_rows):
            if len(selected_rows) == n_rows:
                return selected_rows
            
            if len(candidates) - idx < n_rows - len(selected_rows):
                return None
            
            for i in range(idx, len(candidates)):
                r, d = candidates[i]
                
                if not (d & used_global_diffs):
                    res = backtrack(i + 1, used_global_diffs | d, selected_rows + [list(r)])
                    if res is not None:
                        return res
            return None
        
        return backtrack(0, set(), [])

    for M in range(lower_bound, max_gen_val + 1):
        if time.time() - start_time > time_limit - 5:
            break
            
        sol = find_solution(M)
        if sol:
            return sol
            
    def greedy_fallback():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < num_elements:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows

    return greedy_fallback()
