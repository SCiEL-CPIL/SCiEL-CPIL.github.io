import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(seed_val):
        """Construct rows using greedy with lookahead."""
        random.seed(seed_val)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            
            while len(row) < k + 1:
                last = row[-1]
                found = False
                best_candidate = None
                best_score = float('inf')
                
                # Search window: try values from last+1 upward
                # Use a wider window but evaluate by "tightness"
                search_end = min(last + 80, lb + 100)
                
                candidates = list(range(last + 1, search_end + 1))
                random.shuffle(candidates)
                
                for x in candidates:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        # Score: prefer smaller x (tighter packing)
                        # but also consider future flexibility
                        score = x
                        if score < best_score:
                            best_score = score
                            best_candidate = (x, new_diffs)
                
                if best_candidate is not None:
                    x, new_diffs = best_candidate
                    row.append(x)
                    used.update(new_diffs)
                else:
                    # Fallback: search further
                    for x in range(search_end + 1, lb + 500):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        
                        if valid:
                            row.append(x)
                            used.update(new_diffs)
                            break
                    else:
                        return None, float('inf')
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope

    # 1. Get a valid starting point using multiple greedy seeds
    best_scope = float('inf')
    best_rows = None
    
    for s in range(30):
        rows, scope = greedy_construct(s)
        if rows and scope < best_scope:
            best_scope = scope
            best_rows = rows
        if time.time() > deadline - 110:
            break
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)
        best_scope = max(max(r) for r in best_rows)

    # 2. Local search with swap/shift operations
    current_rows = [list(r) for r in best_rows]
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    def try_shift_row(row, p_idx, new_val, used_diffs, exclude_row_diffs):
        """Try to shift element at p_idx to new_val. Returns (new_row, new_diffs) or None."""
        old_val = row[p_idx]
        if new_val == old_val:
            return None
        
        # Check strict ordering
        if p_idx > 0 and new_val <= row[p_idx - 1]:
            return None
        if p_idx < len(row) - 1 and new_val >= row[p_idx + 1]:
            return None
        if new_val <= 0:
            return None
        
        # Check no duplicate in row
        if new_val in row:
            return None
        
        # Check differences
        new_diffs = set()
        for i in range(len(row)):
            if i != p_idx:
                d = abs(row[i] - new_val)
                if d == 0:
                    return None
                if d in used_diffs and d not in exclude_row_diffs:
                    return None
                new_diffs.add(d)
        
        new_row = list(row)
        new_row[p_idx] = new_val
        return new_row, new_diffs

    current_scope = get_scope(current_rows)
    best_cost = current_scope
    best_rows_ls = [list(r) for r in current_rows]
    
    T = 5.0
    T_min = 0.001
    cooling = 0.9997
    
    steps = 0
    while time.time() < deadline - 10:
        if T < T_min:
            T = 0.5
            cooling = 0.9998
            if steps > 30000:
                break
        
        move_type = random.random()
        
        if move_type < 0.8:
            # Shift one element in one row
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            p_idx = random.randint(1, len(row) - 1)  # Not the first element (must be 0)
            
            old_val = row[p_idx]
            
            # Compute diffs to remove
            diffs_to_remove = set()
            for i in range(len(row)):
                if i != p_idx:
                    diffs_to_remove.add(abs(row[i] - old_val))
            
            # Compute all used diffs
            used_all = get_used_diffs(current_rows)
            exclude = diffs_to_remove  # These are the row's own diffs being removed
            
            # Try a few candidate new values
            found_move = False
            for _ in range(8):
                # Try values near old_val
                delta = random.choice([-3, -2, -1, 1, 2, 3, 5, -5, 7, -7])
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                
                result = try_shift_row(row, p_idx, new_val, used_all - exclude, exclude)
                if result is not None:
                    new_row, new_diffs = result
                    # Check if new_diffs conflict with other rows
                    conflict = False
                    for d in new_diffs:
                        if d in used_all and d not in exclude:
                            conflict = True
                            break
                    if not conflict:
                        # Apply move
                        current_rows[r_idx] = new_row
                        new_scope = get_scope(current_rows)
                        new_cost = new_scope
                        delta_cost = new_cost - current_scope
                        
                        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                            current_scope = new_scope
                            if current_scope < best_cost:
                                best_cost = current_scope
                                best_rows_ls = [list(r) for r in current_rows]
                        else:
                            # Revert
                            current_rows[r_idx] = row
                        found_move = True
                        break
            
        else:
            # Try to compress a row by shifting the last element
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            if len(row) < 2:
                continue
            last_idx = len(row) - 1
            old_val = row[last_idx]
            
            used_all = get_used_diffs(current_rows)
            diffs_to_remove = set()
            for i in range(len(row)):
                if i != last_idx:
                    diffs_to_remove.add(row[last_idx] - row[i])
            exclude = diffs_to_remove
            
            # Try smaller values
            for new_val in range(old_val - 1, 0, -1):
                if new_val in row:
                    continue
                if new_val <= row[last_idx - 1]:
                    break
                
                valid = True
                for i in range(len(row)):
                    if i != last_idx:
                        d = new_val - row[i]
                        if d in used_all and d not in exclude:
                            valid = False
                            break
                
                if valid:
                    new_row = list(row)
                    new_row[last_idx] = new_val
                    current_rows[r_idx] = new_row
                    new_scope = get_scope(current_rows)
                    delta_cost = new_scope - current_scope
                    
                    if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                        current_scope = new_scope
                        if current_scope < best_cost:
                            best_cost = current_scope
                            best_rows_ls = [list(r) for r in current_rows]
                    else:
                        current_rows[r_idx] = row
                    break
        
        T *= cooling
        steps += 1

    # 3. Final compaction: try to shift each element to a smaller value
    final_rows = [list(r) for r in best_rows_ls]
    final_scope = get_scope(final_rows)
    
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        improved = False
        for r_idx in range(n_rows):
            row = final_rows[r_idx]
            for p_idx in range(len(row) - 1, 0, -1):
                old_val = row[p_idx]
                if p_idx == 0:
                    continue
                used_all = get_used_diffs(final_rows)
                diffs_to_remove = set()
                for i in range(len(row)):
                    if i != p_idx:
                        diffs_to_remove.add(abs(row[i] - old_val))
                exclude = diffs_to_remove
                
                for new_val in range(old_val - 1, row[p_idx - 1] if p_idx > 0 else 0, -1):
                    if new_val in row:
                        continue
                    if p_idx > 0 and new_val <= row[p_idx - 1]:
                        break
                    
                    valid = True
                    for i in range(len(row)):
                        if i != p_idx:
                            d = abs(row[i] - new_val)
                            if d in used_all and d not in exclude:
                                valid = False
                                break
                    
                    if valid:
                        row[p_idx] = new_val
                        improved = True
                        break
        
        if not improved:
            break
    
    final_scope = get_scope(final_rows)
    
    if is_valid(final_rows) and final_scope < best_cost:
        best_rows_ls = final_rows
        best_cost = final_scope

    # Final safety check
    if not is_valid(best_rows_ls):
        if is_valid(best_rows):
            best_rows_ls = best_rows
        else:
            used = set()
            best_rows_ls = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used:
                                valid = False
                                break
                        if valid:
                            break
                    row.append(x)
                    used.update({x - a for a in row if a != x})
                best_rows_ls.append(row)

    return best_rows_ls
