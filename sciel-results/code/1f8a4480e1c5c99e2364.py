import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
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

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def generate_row(used_diffs, k_val, randomize=False, p_smallest=0.7, window=5, max_search=200):
        """Generate a single row given the set of already-used differences."""
        row = [0]
        current_row_diffs = set()
        
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            forbidden = used_diffs | current_row_diffs
            
            while x < max_search:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]
                
            row.append(next_x)
            current_row_diffs |= {next_x - a for a in row[:-1]}
            
        return row

    # 1. Initial Solution: Deterministic Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_row_diffs = set()
        for _ in range(k):
            x = row[-1] + 1
            forbidden = used | current_row_diffs
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    row.append(x)
                    current_row_diffs |= new_diffs
                    break
                x += 1
        rows.append(row)
        used |= current_row_diffs
        
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # 2. Simulated Annealing with Row Regeneration
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    while time.time() < deadline - 30:
        if T < T_min:
            T = 5.0
            cooling = 0.998
            
        r_idx = random.randint(0, n_rows - 1)
        
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        new_row = generate_row(used, k, randomize=True, p_smallest=0.8, window=3, max_search=best_max + 20)
        
        if new_row is None:
            used |= old_diffs
            continue
        
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r_idx] = new_row
            used |= set(new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i+1, len(new_row)))
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            used |= old_diffs
            
        T *= cooling

    # 3. Local Search with Individual Element Moves
    current_rows = [list(r) for r in best_rows]
    
    diff_counts = {}
    current_conflict_count = 0
    
    def add_diff(d):
        nonlocal current_conflict_count
        if d in diff_counts:
            if diff_counts[d] == 1:
                current_conflict_count += 1
            diff_counts[d] += 1
        else:
            diff_counts[d] = 1
            
    def remove_diff(d):
        nonlocal current_conflict_count
        if d in diff_counts:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            elif diff_counts[d] == 1:
                current_conflict_count -= 1

    for r in current_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                add_diff(d)
                
    current_scope = max(max(r) for r in current_rows)
    current_cost = current_scope + 1000 * current_conflict_count
    best_cost = current_cost
    best_rows_final = [list(r) for r in current_rows]
    best_scope_final = current_scope
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline - 5:
        if T < T_min:
            T = 2.0
            cooling = 0.9995
            if steps > 50000:
                break
                
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        p_idx = random.randint(1, len(row) - 1)
        
        old_val = row[p_idx]
        
        diffs_to_remove = []
        for i in range(len(row)):
            if i != p_idx:
                diffs_to_remove.append(abs(row[i] - old_val))
        
        for d in diffs_to_remove:
            remove_diff(d)
        
        row.pop(p_idx)
        
        new_val = None
        new_r_idx = r_idx
        
        if random.random() < 0.8:
            max_search = min(best_scope_final + 10, 250)
            found = False
            for _ in range(10):
                cand = random.randint(1, max_search)
                if cand in row:
                    continue
                
                valid = True
                new_diffs = []
                for a in row:
                    d = abs(cand - a)
                    if d in diff_counts:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    pos = 0
                    while pos < len(row) and row[pos] < cand:
                        pos += 1
                    row.insert(pos, cand)
                    for d in new_diffs:
                        add_diff(d)
                    new_val = cand
                    found = True
                    break
            
            if not found:
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        else:
            new_r_idx = random.randint(0, n_rows - 1)
            while new_r_idx == r_idx:
                new_r_idx = random.randint(0, n_rows - 1)
            
            cand = old_val
            target_row = current_rows[new_r_idx]
            
            if cand in target_row:
                cand = random.randint(1, best_scope_final + 10)
                if cand in target_row:
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
                    continue
            
            valid = True
            new_diffs = []
            for a in target_row:
                d = abs(cand - a)
                if d in diff_counts:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                pos = 0
                while pos < len(target_row) and target_row[pos] < cand:
                    pos += 1
                target_row.insert(pos, cand)
                for d in new_diffs:
                    add_diff(d)
                new_val = cand
            else:
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        
        new_scope = max(max(r) for r in current_rows)
        new_cost = new_scope + 1000 * current_conflict_count
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            current_scope = new_scope
            
            if is_valid(current_rows):
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope_final = current_scope
                    best_rows_final = [list(r) for r in current_rows]
        else:
            if new_r_idx != r_idx:
                target_row = current_rows[new_r_idx]
                idx_new = target_row.index(new_val)
                target_row.pop(idx_new)
                for a in target_row:
                    remove_diff(abs(a - new_val))
            else:
                row = current_rows[r_idx]
                idx_new = row.index(new_val)
                row.pop(idx_new)
                for a in row:
                    remove_diff(abs(a - new_val))
            
            row = current_rows[r_idx]
            pos = 0
            while pos < len(row) and row[pos] < old_val:
                pos += 1
            row.insert(pos, old_val)
            for d in diffs_to_remove:
                add_diff(d)
        
        T *= cooling
        steps += 1

    # 4. Enhanced Final Polish: Multi-pass systematic compaction with pull-to-smallest
    final_rows = [list(r) for r in best_rows_final]
    used_set = get_used_diffs(final_rows)
    
    # Multiple passes of systematic compaction - process from right to left
    for pass_num in range(50):
        if time.time() > deadline - 2:
            break
        
        improved = False
        
        # Process rows in order of their max element (smallest first for more freedom)
        row_order = sorted(range(n_rows), key=lambda i: max(final_rows[i]))
        
        for row_idx in row_order:
            row = final_rows[row_idx]
            if len(row) < 2:
                continue
            
            # Try to compact each element (except 0) to the smallest valid position
            # Process from right to left so we don't disturb earlier positions
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline - 2:
                    break
                
                val = row[pos]
                if val <= pos:  # Already as small as possible
                    continue
                
                # Remove val and its differences
                row.pop(pos)
                removed_diffs = set()
                for a in row:
                    d = abs(val - a)
                    removed_diffs.add(d)
                    used_set.discard(d)
                
                # Find smallest valid position
                new_val = None
                for candidate in range(pos + 1, val):
                    if candidate in row:
                        continue
                    valid = True
                    for a in row:
                        d = abs(candidate - a)
                        if d == 0 or d in used_set:
                            valid = False
                            break
                    if valid:
                        new_val = candidate
                        break
                
                if new_val is not None and new_val < val:
                    # Insert at new position
                    insert_pos = 0
                    while insert_pos < len(row) and row[insert_pos] < new_val:
                        insert_pos += 1
                    row.insert(insert_pos, new_val)
                    for a in row:
                        if a != new_val:
                            used_set.add(abs(new_val - a))
                    improved = True
                else:
                    # Revert
                    row.insert(pos, val)
                    for d in removed_diffs:
                        used_set.add(d)
        
        if not improved:
            break
    
    # 5. Additional polish: Try pairwise swaps between rows to free up difference slots
    if time.time() < deadline - 3:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            
            improved_swap = False
            for i in range(n_rows):
                for j in range(i + 1, n_rows):
                    if time.time() > deadline - 3:
                        break
                    
                    row_i = final_rows[i]
                    row_j = final_rows[j]
                    
                    # Try swapping elements between rows
                    for pi in range(1, len(row_i)):
                        for pj in range(1, len(row_j)):
                            val_i = row_i[pi]
                            val_j = row_j[pj]
                            
                            if val_i == val_j:
                                continue
                            
                            # Check if swap is valid
                            # Remove val_i from row_i and val_j from row_j
                            row_i_copy = row_i[:]
                            row_j_copy = row_j[:]
                            row_i_copy.pop(pi)
                            row_j_copy.pop(pj)
                            
                            # Try inserting val_j into row_i
                            valid_i = True
                            for a in row_i_copy:
                                d = abs(val_j - a)
                                if d == 0 or d in used_set:
                                    valid_i = False
                                    break
                            
                            # Try inserting val_i into row_j
                            valid_j = True
                            for a in row_j_copy:
                                d = abs(val_i - a)
                                if d == 0 or d in used_set:
                                    valid_j = False
                                    break
                            
                            if valid_i and valid_j:
                                # Check that the new differences don't conflict with each other
                                new_diffs_i = {abs(val_j - a) for a in row_i_copy}
                                new_diffs_j = {abs(val_i - a) for a in row_j_copy}
                                
                                if not (new_diffs_i & new_diffs_j):
                                    # Perform the swap
                                    row_i.pop(pi)
                                    row_j.pop(pj)
                                    
                                    # Insert val_j into row_i
                                    pos = 0
                                    while pos < len(row_i) and row_i[pos] < val_j:
                                        pos += 1
                                    row_i.insert(pos, val_j)
                                    
                                    # Insert val_i into row_j
                                    pos = 0
                                    while pos < len(row_j) and row_j[pos] < val_i:
                                        pos += 1
                                    row_j.insert(pos, val_i)
                                    
                                    # Update used_set
                                    for a in row_i:
                                        if a != val_j:
                                            used_set.add(abs(val_j - a))
                                    for a in row_j:
                                        if a != val_i:
                                            used_set.add(abs(val_i - a))
                                    
                                    improved_swap = True
                                    break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
            
            if not improved_swap:
                break
    
    final_scope = max(max(r) for r in final_rows)
    
    if is_valid(final_rows) and final_scope < best_scope_final:
        best_rows_final = final_rows
        best_scope_final = final_scope
        
    if not is_valid(best_rows_final):
        if is_valid(best_rows):
            best_rows_final = best_rows
        else:
            used = set()
            best_rows_final = []
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
                best_rows_final.append(row)

    return best_rows_final
