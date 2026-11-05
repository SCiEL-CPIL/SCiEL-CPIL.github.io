import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: 150s limit. Keep 5s buffer.
    deadline = start_time + 145
    
    # Problem parameters
    # n_rows=4, k=7 (8 marks) based on "8 increasing integers" and "counting bound 112"
    # If k is passed as 8, it means 9 marks. But the prompt says "8 increasing integers".
    # The interface says "list of k+1 integers".
    # If the evaluator passes k=7, we get 8 marks.
    # If the evaluator passes k=8, we get 9 marks.
    # The prompt says "Choose 4 rows of 8 increasing integers".
    # This implies the number of integers is 8.
    # So k+1 = 8 => k=7.
    # I will assume the evaluator passes k=7.
    # However, to be safe, I should handle the k passed.
    
    n_marks = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Theoretical lower bound for max entry
    # If total_diffs distinct integers are needed, max entry >= total_diffs
    lower_bound = total_diffs
    
    # Best known is 134 for k=7 (8 marks).
    # Wait, if k=7, total_diffs = 4 * 28 = 112.
    # If k=8, total_diffs = 4 * 36 = 144.
    # The prompt says "counting bound is 112". This confirms k=7.
    
    # Greedy Initialization
    def greedy_init():
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_marks:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Compute max entry
    def get_max_entry(rows):
        return max(max(r) for r in rows)

    # Check validity
    def is_valid(rows):
        used = set()
        for r in range(n_rows):
            row = rows[r]
            # Check strictly increasing
            for i in range(1, n_marks):
                if row[i] <= row[i-1]:
                    return False
            for i in range(n_marks):
                for j in range(i + 1, n_marks):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # DFS to find solution with max_entry <= target
    def dfs_solve(target_max):
        used_diffs = set()
        rows = []
        
        # Node limit to prevent infinite loops or excessive time
        node_limit = 500000
        nodes_visited = [0]
        
        def backtrack_row(row_idx, pos, current_row, current_diffs):
            if time.time() > deadline - 10:
                return False
            nodes_visited[0] += 1
            if nodes_visited[0] > node_limit:
                return False
            
            if pos == n_marks:
                # Row complete
                rows.append(list(current_row))
                if row_idx == n_rows - 1:
                    return True
                return backtrack_row(row_idx + 1, 1, [0], set())
            
            last_mark = current_row[-1]
            min_val = last_mark + 1
            max_val = target_max
            
            # Pruning: ensure we can fit remaining marks
            remaining_marks = n_marks - pos - 1
            if min_val + remaining_marks > target_max:
                return False
            
            for val in range(min_val, max_val + 1):
                new_diffs = {val - a for a in current_row}
                # Check conflicts with current row diffs and global used diffs
                if new_diffs & current_diffs or new_diffs & used_diffs:
                    continue
                
                # Check if new_diffs are distinct among themselves?
                # Yes, because val is unique and a's are distinct.
                
                # Apply
                current_row.append(val)
                current_diffs.update(new_diffs)
                
                if backtrack_row(row_idx, pos + 1, current_row, current_diffs):
                    return True
                
                # Backtrack
                current_row.pop()
                current_diffs.difference_update(new_diffs)
                
        # Start DFS
        # We can try to optimize row order, but fixed is fine for now
        # To break symmetry, we can fix the first row's last mark? No.
        
        return backtrack_row(0, 1, [0], set())

    # Simulated Annealing
    def simulated_annealing(rows, time_budget):
        current_rows = [list(r) for r in rows]
        best_rows = [list(r) for r in current_rows]
        
        def compute_cost(rows):
            used = set()
            max_entry = 0
            conflicts = 0
            for r in range(n_rows):
                row = rows[r]
                for m in range(n_marks):
                    if row[m] > max_entry:
                        max_entry = row[m]
                for i in range(n_marks):
                    for j in range(i + 1, n_marks):
                        d = row[j] - row[i]
                        if d <= 0:
                            conflicts += 1000
                            continue
                        if d in used:
                            conflicts += 1
                        used.add(d)
            return max_entry + conflicts * 100

        current_cost = compute_cost(current_rows)
        best_cost = current_cost
        
        T_start = 50.0
        T_end = 0.01
        steps = 0
        max_steps = 200000
        
        while time.time() < time_budget and steps < max_steps:
            steps += 1
            T = T_start * ((T_end / T_start) ** (steps / max_steps))
            
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = current_rows[r][pos]
            min_val = current_rows[r][pos - 1] + 1
            max_val = current_rows[r][pos + 1] - 1 if pos < k else 200
            
            if min_val > max_val:
                continue
            
            if random.random() < 0.5:
                delta = random.randint(-5, 5)
                new_val = old_val + delta
            else:
                new_val = random.randint(min_val, max_val)
            
            if new_val < min_val or new_val > max_val or new_val == old_val:
                continue
            
            current_rows[r][pos] = new_val
            new_cost = compute_cost(current_rows)
            
            delta_cost = new_cost - current_cost
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 1e-10)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(row) for row in current_rows]
            else:
                current_rows[r][pos] = old_val
        
        return best_rows

    # 1. Try DFS to find optimal (or near-optimal) solution
    # Target the lower bound
    target_max = lower_bound
    # Allow a small slack if exact bound is hard? 
    # Let's try exact first.
    
    # Run DFS with a time limit
    dfs_start = time.time()
    dfs_budget = 20 # seconds
    found = dfs_solve(target_max)
    
    if found:
        # Verify
        if is_valid(rows):
            return rows
    
    # 2. Fallback to SA
    greedy_rows = greedy_init()
    sa_budget = deadline - 5
    sa_rows = simulated_annealing(greedy_rows, sa_budget)
    
    return sa_rows
