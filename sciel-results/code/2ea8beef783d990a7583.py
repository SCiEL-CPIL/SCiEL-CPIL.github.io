import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
            if r != sorted(r):
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

    # 1. Initial Solution: Deterministic Greedy + Randomized Regeneration
    # Start with a deterministic greedy solution to ensure we have a valid baseline
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
    # Instead of moving individual elements (which can break row structure easily),
    # we regenerate entire rows. This is more robust for maintaining the "increasing integers" constraint
    # and allows the row to find a compact configuration within the remaining difference space.
    
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    while time.time() < deadline - 5:
        if T < T_min:
            T = 5.0  # Reheat
            cooling = 0.998
            
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove diffs of row r_idx from used
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        # Regenerate row with randomization
        # We try to keep it compact, so we limit the search space slightly
        # but allow some randomness in choice.
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
            used |= set(new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i+1, len(new_row)))
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
                    # Find position
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
                # Revert
                pos = 0
                while pos < len(current_rows[r_idx]) and current_rows[r_idx][pos] < old_val:
                    pos += 1
                current_rows[r_idx].insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue # Skip this iteration

        else:
            # Move to another row
            new_r_idx = random.randint(0, n_rows - 1)
            while new_r_idx == r_idx:
                new_r_idx = random.randint(0, n_rows - 1)
            
            # Try to insert old_val into new_r_idx
            cand = old_val
            
            if cand in current_rows[new_r_idx]:
                # Conflict with existing element in new row
                # Try a nearby value
                cand = random.randint(1, best_scope_final + 10)
                if cand in current_rows[new_r_idx]:
                    # Revert
                    pos = 0
                    while pos < len(current_rows[r_idx]) and current_rows[r_idx][pos] < old_val:
                        pos += 1
                    current_rows[r_idx].insert(pos, old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
                    continue

            valid = True
            new_diffs = []
            for a in current_rows[new_r_idx]:
                d = abs(cand - a)
                if d in diff_counts:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                pos = 0
                while pos < len(current_rows[new_r_idx]) and current_rows[new_r_idx][pos] < cand:
                    pos += 1
                current_rows[new_r_idx].insert(pos, cand)
                for d in new_diffs:
                    add_diff(d)
                new_val = cand
            else:
                # Revert
                pos = 0
                while pos < len(current_rows[r_idx]) and current_rows[r_idx][pos] < old_val:
                    pos += 1
                current_rows[r_idx].insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue

        # Calculate new cost
        new_scope = max(max(r) for r in current_rows)
        new_cost = new_scope + 1000 * current_conflict_count
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_cost = new_cost
            current_scope = new_scope
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_scope_final = current_scope
                best_rows_final = [list(r) for r in current_rows]
                
        else:
            # Revert
            # Remove new_val from new_r_idx
            if new_r_idx != r_idx:
                # Find and remove new_val from new_r_idx
                row_new = current_rows[new_r_idx]
                idx_new = row_new.index(new_val)
                row_new.pop(idx_new)
                # Remove diffs
                for a in row_new:
                    remove_diff(abs(a - new_val))
            else:
                # Same row, different value?
                row_old = current_rows[r_idx]
                idx_new = row_old.index(new_val)
                row_old.pop(idx_new)
                for a in row_old:
                    remove_diff(abs(a - new_val))
            
            # Re-insert old_val into r_idx
            pos = 0
            while pos < len(current_rows[r_idx]) and current_rows[r_idx][pos] < old_val:
                pos += 1
            current_rows[r_idx].insert(pos, old_val)
            for d in diffs_to_remove:
                add_diff(d)
            
            # Restore current_cost
            # current_cost was not changed
            pass

        T *= cooling
        steps += 1
        
        if steps % 1000 == 0:
             # Optional: Local search / Compaction on best_rows
             if time.time() > deadline - 5:
                 break

    # Final Compaction on best_rows
    if best_rows_final is not None and best_scope_final > lb:
        # Use the old try_compact_rows logic on the best solution found
        used_set = get_used_diffs(best_rows_final)
        
        def quick_compact(rows, used_set, max_iter=100):
            scope = max(max(r) for r in rows)
            for _ in range(max_iter):
                if time.time() > deadline - 1:
                    break
                improved = False
                row_idx = random.randint(0, n_rows - 1)
                row = rows[row_idx]
                if len(row) < 2:
                    continue
                last = row[-1]
                if last <= 1:
                    continue
                removed = row.pop()
                for a in row:
                    used_set.discard(removed - a)
                
                found = False
                for x in range(last - 1, 0, -1):
                    if x in row:
                        continue
                    valid = True
                    for a in row:
                        d = abs(x - a)
                        if d == 0 or d in used_set:
                            valid = False
                            break
                    if valid:
                        pos = 0
                        while pos < len(row) and row[pos] < x:
                            pos += 1
                        row.insert(pos, x)
                        for a in row:
                            if a != x:
                                used_set.add(abs(x - a))
                        found = True
                        improved = True
                        break
                
                if not found:
                    row.append(removed)
                    row.sort()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            used_set.add(row[j] - row[i])
                
                if improved:
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope_final:
                        best_scope_final = new_scope
                        best_rows_final = [list(r) for r in rows]

        quick_compact(best_rows_final, used_set, 200)

    if best_rows_final is None:
        # Fallback
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
    
    # Final validity check and fix if needed
    if not is_valid(best_rows_final):
        # If invalid, regenerate deterministically
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
