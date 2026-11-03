import time
import random
import copy

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Greedy Initialization
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < row_len:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    break
                x += 1
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
        
    best_rows = copy.deepcopy(rows)
    best_span = max(max(r) for r in rows)
    
    # 2. Simulated Annealing
    current_rows = copy.deepcopy(rows)
    current_used = set()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                current_used.add(r[j] - r[i])
        
    row_maxes = [max(r) for r in current_rows]
    current_span = max(row_maxes)
    current_conflicts = 0
    current_cost = current_span
    
    T = 10.0
    cooling = 0.999
    
    # Pre-calculate max for other rows to speed up span calc?
    # Since N is small, O(N) is fine.
    
    while time.time() < deadline and T > 0.1:
        # Choose move type: 0 for point, 1 for row shift
        move_type = random.randint(0, 1)
        i = random.randint(0, n_rows - 1)
        
        if move_type == 0:
            # Point move
            j = random.randint(1, row_len - 1)
            delta = random.choice([-2, -1, 1, 2])
            if random.random() < 0.2:
                delta = random.randint(-5, 5)
                
            new_val = current_rows[i][j] + delta
            
            # Bounds check
            if new_val <= current_rows[i][j-1]:
                continue
            if j < row_len - 1 and new_val >= current_rows[i][j+1]:
                continue
                
            old_row = current_rows[i]
            # Remove old diffs
            for a in range(len(old_row)):
                for b in range(a+1, len(old_row)):
                    current_used.remove(old_row[b] - old_row[a])
                    
            new_row = old_row.copy()
            new_row[j] = new_val
            
            # Calc new diffs
            diffs_list = []
            for a in range(len(new_row)):
                for b in range(a+1, len(new_row)):
                    diffs_list.append(new_row[b] - new_row[a])
            
            c_inter = sum(1 for d in diffs_list if d in current_used)
            c_intra = len(diffs_list) - len(set(diffs_list))
            total_conflicts = c_inter + c_intra
            
            new_max_row = max(new_row)
            other_max = 0
            for idx in range(n_rows):
                if idx != i:
                    if row_maxes[idx] > other_max:
                        other_max = row_maxes[idx]
            new_span = max(new_max_row, other_max)
            
            new_cost = new_span + 50 * total_conflicts
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                current_rows[i] = new_row
                row_maxes[i] = new_max_row
                current_span = new_span
                current_conflicts = total_conflicts
                current_cost = new_cost
                current_used.update(diffs_list)
                
                if total_conflicts == 0 and new_span < best_span:
                    best_span = new_span
                    best_rows = copy.deepcopy(current_rows)
            else:
                # Restore old diffs
                for a in range(len(old_row)):
                    for b in range(a+1, len(old_row)):
                        current_used.add(old_row[b] - old_row[a])
                        
        else:
            # Row Shift Move
            delta = random.choice([-1, 1])
            old_row = current_rows[i]
            
            # Check if shift is valid (maintains increasing order starting at 0)
            # row[0] is 0. row[1] must be > 0.
            if old_row[1] + delta <= 0:
                continue
                
            # Remove old diffs
            for a in range(len(old_row)):
                for b in range(a+1, len(old_row)):
                    current_used.remove(old_row[b] - old_row[a])
                    
            new_row = [0] + [x + delta for x in old_row[1:]]
            
            # Internal diffs are unchanged, so c_intra = 0 (assuming old row was valid internally)
            # But wait, if old row had internal conflicts, they persist.
            # However, we start with valid rows and only accept valid or slightly invalid.
            # If we assume current state has 0 intra conflicts (which we try to maintain), then intra is 0.
            # Let's be safe and calculate.
            diffs_list = []
            for a in range(len(new_row)):
                for b in range(a+1, len(new_row)):
                    diffs_list.append(new_row[b] - new_row[a])
            
            c_inter = sum(1 for d in diffs_list if d in current_used)
            c_intra = len(diffs_list) - len(set(diffs_list))
            total_conflicts = c_inter + c_intra
            
            new_max_row = max(new_row)
            other_max = 0
            for idx in range(n_rows):
                if idx != i:
                    if row_maxes[idx] > other_max:
                        other_max = row_maxes[idx]
            new_span = max(new_max_row, other_max)
            
            new_cost = new_span + 50 * total_conflicts
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                current_rows[i] = new_row
                row_maxes[i] = new_max_row
                current_span = new_span
                current_conflicts = total_conflicts
                current_cost = new_cost
                current_used.update(diffs_list)
                
                if total_conflicts == 0 and new_span < best_span:
                    best_span = new_span
                    best_rows = copy.deepcopy(current_rows)
            else:
                # Restore old diffs
                for a in range(len(old_row)):
                    for b in range(a+1, len(old_row)):
                        current_used.add(old_row[b] - old_row[a])

        T *= cooling
        
    return best_rows
