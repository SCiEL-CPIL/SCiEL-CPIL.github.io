import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    diffs_per_row = row_len * (row_len - 1) // 2
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(row_len) for j in range(i + 1, row_len)}

    def greedy_init(deterministic=False):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                x = row[-1] + 1
                found = False
                # Search for next valid element
                # Limit search to avoid infinite loops in bad states, though greedy should always find one
                while not found:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                    else:
                        x += 1
            rows.append(row)
        return rows, used

    def anneal(rows, used, time_limit):
        current_rows = [r[:] for r in rows]
        current_used = set(used)
        current_cost = max(max(r) for r in current_rows)
        best_rows = [r[:] for r in current_rows]
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9999
        start_time = time.time()
        
        while time.time() - start_time < time_limit and T > T_min:
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            p_idx = random.randint(1, row_len - 1)
            
            # Determine range for new value
            # Must be strictly between neighbors
            low = row[p_idx - 1] + 1
            high = row[p_idx + 1] - 1 if p_idx < row_len - 1 else current_cost + 20
            
            if high < low:
                T *= cooling
                continue
                
            # Randomly pick a new value
            new_val = random.randint(low, high)
            if new_val == row[p_idx]:
                T *= cooling
                continue
                
            # Calculate old diffs involving p_idx
            old_diffs = set()
            for j in range(row_len):
                if j != p_idx:
                    old_diffs.add(abs(row[p_idx] - row[j]))
            
            # Calculate new diffs involving p_idx
            new_diffs = set()
            for j in range(row_len):
                if j != p_idx:
                    new_diffs.add(abs(new_val - row[j]))
            
            # Check validity
            # New diffs must not be in current_used unless they are in old_diffs
            valid = True
            for d in new_diffs:
                if d in current_used and d not in old_diffs:
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
                
            # Apply move
            old_val = row[p_idx]
            row[p_idx] = new_val
            current_used -= old_diffs
            current_used |= new_diffs
            
            new_cost = max(max(r) for r in current_rows)
            delta = new_cost - current_cost
            
            if delta <= 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [r[:] for r in current_rows]
            else:
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    row[p_idx] = old_val
                    current_used -= new_diffs
                    current_used |= old_diffs
            
            T *= cooling
            
        return best_rows, best_cost

    # Main Logic
    start_global = time.time()
    max_total_time = 110
    
    # 1. Generate several initial candidates
    candidates = []
    
    # Deterministic greedy
    rows_det, used_det = greedy_init(deterministic=True)
    candidates.append((rows_det, used_det))
    
    # Randomized greedy
    for _ in range(5):
        if time.time() - start_global > max_total_time - 10:
            break
        # Simple randomization: shuffle order of rows? No, rows are symmetric.
        # Randomize the search for the next element?
        # Let's just run the deterministic one again? It's deterministic.
        # We need randomness in init.
        # Modify greedy to pick randomly from valid candidates in a small window
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                x = row[-1] + 1
                candidates_x = []
                # Look ahead a bit
                limit = x + 20
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates_x.append(x)
                    x += 1
                if not candidates_x:
                    # Fallback
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates_x.append(x)
                            break
                        x += 1
                chosen = random.choice(candidates_x)
                new_diffs = {chosen - a for a in row}
                row.append(chosen)
                used.update(new_diffs)
            rows.append(row)
        candidates.append((rows, used))

    # 2. Anneal the best candidates
    best_overall_rows = None
    best_overall_cost = float('inf')
    
    # Sort candidates by initial cost
    candidates.sort(key=lambda x: max(max(r) for r in x[0]))
    
    # Allocate time to top candidates
    time_per_candidate = (max_total_time - (time.time() - start_global)) / len(candidates)
    
    for i, (rows, used) in enumerate(candidates):
        if time.time() - start_global > max_total_time - 1:
            break
            
        time_left = max_total_time - (time.time() - start_global)
        t_alloc = min(time_per_candidate, time_left * 0.5) # Use half of remaining for this, leave buffer
        if t_alloc < 1:
            t_alloc = time_left * 0.1
            
        res_rows, res_cost = anneal(rows, used, t_alloc)
        
        if res_cost < best_overall_cost:
            best_overall_cost = res_cost
            best_overall_rows = res_rows
            
    return best_overall_rows
