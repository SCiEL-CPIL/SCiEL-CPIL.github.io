import time

def construct(n_rows, k):
    start_time = time.time()
    TIME_LIMIT = 115
    
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_solution = greedy_solution()
    best_max = max(max(r) for r in best_solution)
    
    def build_row(forbidden, max_val):
        row = [0]
        nodes_explored = [0]
        
        def backtrack():
            if time.time() - start_time > TIME_LIMIT:
                return False
            nodes_explored[0] += 1
            if nodes_explored[0] > 200000:
                return False
            
            if len(row) == k + 1:
                return True
            
            last = row[-1]
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden:
                    continue
                
                row.append(x)
                forbidden.update(new_diffs)
                
                if backtrack():
                    return True
                
                forbidden -= new_diffs
                row.pop()
            
            return False
        
        if backtrack():
            return list(row)
        return None
    
    for iteration in range(20):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        improved = False
        for row_idx in range(n_rows):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            fixed_rows = [best_solution[i] for i in range(n_rows) if i != row_idx]
            forbidden = set()
            for r in fixed_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        forbidden.add(r[j] - r[i])
            
            current_max = best_solution[row_idx][-1]
            
            for bound in range(current_max - 1, 105, -1):
                if time.time() - start_time > TIME_LIMIT:
                    break
                
                new_row = build_row(set(forbidden), bound)
                if new_row is not None:
                    new_solution = [best_solution[i] for i in range(n_rows)]
                    new_solution[row_idx] = new_row
                    new_max = max(max(r) for r in new_solution)
                    if new_max < best_max:
                        best_solution = new_solution
                        best_max = new_max
                        improved = True
                    break
        
        if not improved:
            break
    
    return best_solution
