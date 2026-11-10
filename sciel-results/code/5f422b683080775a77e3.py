import time
import random
import sys

sys.setrecursionlimit(100000)

DEADLINE = 0
TIME_LIMIT = 115

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # 1. Greedy Baseline
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
    
    best_rows = baseline_rows
    best_scope = max(max(row) for row in baseline_rows)
    
    # 2. Calculate Lower Bound
    diffs_per_row = k * (k + 1) // 2
    generic_lb = n_rows * diffs_per_row
    lower_bound = generic_lb
    
    # 3. Try Backtracking for M from LB to Best Scope
    max_bt_attempts = 15 
    attempts = 0
    
    for M in range(lower_bound, best_scope + 1):
        if time.time() > DEADLINE:
            break
        if attempts >= max_bt_attempts:
            break
            
        sol = solve_backtracking(n_rows, k, M)
        attempts += 1
        if sol is not None:
            return sol
            
    # 4. Simulated Annealing to improve best_scope
    if time.time() < DEADLINE - 5:
        sa_result = simulated_annealing(n_rows, k, best_rows, best_scope)
        if sa_result:
            sa_scope = max(max(row) for row in sa_result)
            if sa_scope < best_scope:
                best_rows = sa_result
                best_scope = sa_scope

    return best_rows

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        local_used = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & local_used):
                row.append(x)
                local_used |= new_diffs
        rows.append(row)
        used |= local_used
    return rows

def solve_backtracking(n_rows, k, M):
    diffs_per_row = k * (k + 1) // 2
    bit_of = [1 << d for d in range(M + 1)]
    used_mask = 0
    rows = []
    
    def build_row(current_row, current_row_mask, rows_remaining):
        nonlocal used_mask
        if time.time() > DEADLINE:
            return False
        
        if len(current_row) == k + 1:
            rows.append(list(current_row))
            used_mask |= current_row_mask
            return True
            
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = bit_of[d]
                if (used_mask | current_row_mask) & bit:
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        if not candidates:
            return False
            
        candidates.sort(key=lambda t: t[0])
        
        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask, rows_remaining - 1):
                return True
            current_row.pop()
            
        return False

    def run_search():
        nonlocal used_mask, rows
        used_mask = 0
        rows = []
        for i in range(n_rows):
            if time.time() > DEADLINE:
                return False
            if not build_row([0], 0, n_rows - i - 1):
                return False
        return True

    if run_search():
        return rows
    return None

def simulated_annealing(n_rows, k, initial_rows, initial_scope):
    rows = [r[:] for r in initial_rows]
    
    row_diffs = []
    global_diffs = set()
    for r in rows:
        d = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d.add(r[j] - r[i])
        row_diffs.append(d)
        global_diffs |= d
        
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)
    
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() < DEADLINE:
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        pos = random.randint(1, k)
        old_val = row[pos]
        
        low = row[pos-1] + 1
        high = row[pos+1] - 1 if pos < k else best_scope + 10
        
        if low > high:
            T *= cooling
            continue
            
        current_max = max(max(r) for r in rows)
        max_search = min(high, current_max + 5)
        if low > max_search:
            T *= cooling
            continue
            
        new_val = random.randint(low, max_search)
        
        if new_val == old_val:
            continue
            
        old_local_diffs = set()
        for j in range(len(row)):
            if j != pos:
                if j < pos:
                    old_local_diffs.add(row[pos] - row[j])
                else:
                    old_local_diffs.add(row[j] - row[pos])
                    
        new_local_diffs = set()
        for j in range(len(row)):
            if j != pos:
                if j < pos:
                    new_local_diffs.add(new_val - row[j])
                else:
                    new_local_diffs.add(row[j] - new_val)
                    
        other_local_diffs = row_diffs[r_idx] - old_local_diffs
        
        if new_local_diffs & other_local_diffs:
            T *= cooling
            continue
            
        global_other_diffs = global_diffs - row_diffs[r_idx]
        
        if new_local_diffs & global_other_diffs:
            T *= cooling
            continue
            
        old_row_max = max(row)
        new_row_max = max(old_row_max, new_val)
        
        other_max = 0
        for i in range(n_rows):
            if i != r_idx:
                if max(rows[i]) > other_max:
                    other_max = max(rows[i])
                    
        old_global_max = max(max(r) for r in rows)
        new_global_max = max(new_row_max, other_max)
        
        delta_cost = new_global_max - old_global_max
        
        if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
            global_diffs.difference_update(row_diffs[r_idx])
            row[pos] = new_val
            row_diffs[r_idx] = other_local_diffs | new_local_diffs
            global_diffs.update(row_diffs[r_idx])
            
            current_scope = new_global_max
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
                
        T *= cooling
        
    return best_rows
