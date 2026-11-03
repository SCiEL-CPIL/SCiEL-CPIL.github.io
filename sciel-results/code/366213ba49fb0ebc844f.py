import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    def get_row_diffs(row):
        """Get all pairwise differences for a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def generate_row(used_diffs, k_val, randomize=False, p_smallest=0.7, window=5, max_search=None):
        """Generate a single row given the set of already-used differences."""
        if max_search is None:
            max_search = lb + 100
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
    # This approach regenerates entire rows, which is robust for maintaining
    # the "increasing integers" constraint and allows finding compact configurations.
    
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    while time.time() < deadline - 10:
        if T < T_min:
            T = 5.0  # Reheat
            cooling = 0.998
            
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove diffs of row r_idx from used
        old_row = rows[r_idx]
        old_diffs = get_row_diffs(old_row)
        used -= old_diffs
        
        # Regenerate row with randomization
        # Limit search space to slightly above current best to encourage compactness
        # but allow some room for improvement
        new_row = generate_row(used, k, randomize=True, p_smallest=0.8, window=3, max_search=best_max + 20)
        
        if new_row is None:
            # Rollback
            used |= old_diffs
            continue
        
        # Calculate new max
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            rows[r_idx] = new_row
            used |= get_row_diffs(new_row)
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Rollback
            used |= old_diffs
            
        T *= cooling

    # 3. Final Local Search: Individual Element Moves for Fine-Tuning
    # After the coarse row-regeneration, we do fine-tuning by moving individual elements.
    # This can squeeze out the last few units.
    
    current_rows = [list(r) for r in best_rows]
    
    # Build diff counts for efficient conflict checking
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
    while time.time() < deadline - 2:
        if T < T_min:
            T = 2.0
            cooling = 0.9995
            if steps > 50000:
                break
                
        # Pick a random element to modify
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        p_idx = random.randint(1, len(row) - 1)  # Don't move 0
        
        old_val = row[p_idx]
        
        # Remove old_val's contributions
        diffs_to_remove = []
        for i in range(len(row)):
            if i != p_idx:
                diffs_to_remove.append(abs(row[i] - old_val))
        
        for d in diffs_to_remove:
            remove_diff(d)
        
        row.pop(p_idx)
        
        # Try to find a new valid value
        new_val = None
        new_r_idx = r_idx
        
        if random.random() < 0.8:
            # Value change in same row
            max_search = min(best_scope_final + 10, 250)
            found = False
            # Try a few random candidates
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
                    # Check if it maintains sorted order
                    insert_pos = 0
                    while insert_pos < len(row) and row[insert_pos] < cand:
                        insert_pos += 1
                    
                    # We need to ensure the row remains strictly increasing
                    # Since we removed an element, we can insert at any valid position
                    # But we must ensure no duplicate values
                    if cand in row:
                        continue
                        
                    # Check if inserting at insert_pos is valid
                    # Actually, we just need to add it back to the row and sort
                    # But we must ensure uniqueness
                    row.append(cand)
                    row.sort()
                    
                    # Verify no duplicates
                    if len(row) != len(set(row)):
                        row.remove(cand)
                        continue
                        
                    # Check differences again
                    valid = True
                    for a in row[:-1]: # Exclude the newly added cand itself
                        d = cand - a
                        if d in diff_counts:
                            valid = False
                            break
                        # Also check against other elements in the row
                        # Wait, diff_counts contains global diffs.
                        # We need to check if the new diffs conflict with global used diffs
                        # AND with other diffs in the same row (which are not in diff_counts yet)
                        
                    # Simpler: just check against global diff_counts
                    # The row itself will be checked for internal consistency later
                    # But we must ensure the new diffs don't conflict with global
                    valid = True
                    for a in row:
                        if a == cand:
                            continue
                        d = abs(cand - a)
                        if d in diff_counts:
                            valid = False
                            break
                    
                    if valid:
                        new_val = cand
                        found = True
                        break
                    else:
                        row.remove(cand)
            
            if found:
                # Add new diffs
                for a in current_rows[r_idx]:
                    if a != new_val:
                        add_diff(abs(new_val - a))
                current_rows[r_idx] = row
            else:
                # Rollback
                row.append(old_val)
                row.sort()
                current_rows[r_idx] = row
                for d in diffs_to_remove:
                    add_diff(d)
        else:
            # Rollback
            row.append(old_val)
            row.sort()
            current_rows[r_idx] = row
            for d in diffs_to_remove:
                add_diff(d)
        
        # Update cost
        current_scope = max(max(r) for r in current_rows)
        current_cost = current_scope + 1000 * current_conflict_count
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_rows_final = [list(r) for r in current_rows]
            best_scope_final = current_scope
            
        delta = current_cost - best_cost
        if delta > 0:
            if random.random() >= math.exp(-delta / T):
                # Rollback to best? No, SA usually rolls back to previous state.
                # But we are modifying in place. We need to keep track of previous state.
                # This is getting complicated. Let's just keep the best found so far.
                pass
        
        T *= cooling
        steps += 1

    # Return the best valid result found
    # Ensure the final result is valid
    used_check = set()
    valid = True
    for r in best_rows_final:
        if len(r) != k + 1:
            valid = False
            break
        if r[0] != 0:
            valid = False
            break
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in used_check:
                    valid = False
                    break
                used_check.add(d)
        if not valid:
            break
    
    if valid:
        return best_rows_final
    else:
        # Fallback to the initial greedy solution if final search broke something
        return best_rows

