import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    best_rows = None
    best_scope = float('inf')
    best_cost = float('inf')
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct(seed_val, strategy='compact'):
        """Construct rows using greedy with various strategies."""
        random.seed(seed_val)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                elements_placed = len(row)
                if strategy == 'compact':
                    window = 30 + elements_placed * 10
                elif strategy == 'spread':
                    window = 50 + elements_placed * 15
                else:
                    window = 40 + elements_placed * 12
                
                search_end = min(last + window, lb + 200)
                
                candidates = list(range(last + 1, search_end + 1))
                
                if strategy == 'compact':
                    random.shuffle(candidates)
                    candidates.sort(key=lambda x: x + random.random() * 3)
                elif strategy == 'spread':
                    random.shuffle(candidates)
                else:
                    if random.random() < 0.5:
                        candidates.sort(key=lambda x: x + random.random() * 10)
                    else:
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
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    for x in range(last + 1, lb + 500):
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
                            found = True
                            break
                    if not found:
                        return None, float('inf')
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope

    # 1. Get a valid starting point using greedy
    # Try a few seeds to find a good initial solution
    initial_rows = None
    for s in range(20):
        rows, scope = greedy_construct(s, 'compact')
        if rows:
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
            if initial_rows is None:
                initial_rows = rows
        if time.time() > deadline - 100:
            break
    
    if initial_rows is None:
        # Fallback if greedy fails (unlikely)
        used = set()
        initial_rows = []
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
            initial_rows.append(row)
        best_rows = initial_rows
        best_scope = max(max(r) for r in initial_rows)

    # 2. Simulated Annealing
    # State: rows (list of lists)
    # We maintain a global set of used differences for efficiency? 
    # Actually, updating the set for every move is O(k). 
    # Let's maintain the set.
    
    current_rows = [list(r) for r in initial_rows]
    current_used = get_used_diffs(current_rows)
    current_scope = max(max(r) for r in current_rows)
    current_conflicts = 0 # Should be 0 initially
    
    # Cost function: Scope + Penalty * Conflicts
    # We want to minimize Scope. Conflicts must be 0.
    # If we allow conflicts, we need a high penalty to ensure we return to validity.
    # Let's define cost = scope + 1000 * conflicts
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute indices for elements
    # Total elements = n_rows * k (excluding 0s)
    
    def get_cost(rows, used_set, conflicts):
        scope = max(max(r) for r in rows)
        return scope + 1000 * conflicts

    # To speed up conflict checking, we can just count how many times a diff appears.
    # But maintaining a count dict is expensive to update if we remove/add many.
    # Since k is small (e.g., 8), a row has ~28 diffs.
    # Total diffs ~ n_rows * 28.
    # A move affects O(k) diffs.
    
    # Let's use a dictionary of diff -> count
    diff_counts = {}
    for d in current_used:
        diff_counts[d] = 1 # Initially valid, so all unique
    # Wait, if it's valid, all counts are 1.
    # Conflicts = number of d where count[d] > 1.
    
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

    # Initialize diff_counts properly (in case initial was not perfect, but greedy ensures valid)
    # Just to be safe, rebuild
    diff_counts = {}
    current_conflict_count = 0
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                add_diff(d)
                
    current_scope = max(max(r) for r in current_rows)
    current_cost = current_scope + 1000 * current_conflict_count
    best_cost = current_cost
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope

    # SA Loop
    steps = 0
    while time.time() < deadline - 2:
        if T < T_min:
            T = 1.0 # Reheat
            cooling = 0.9995 # Slower cooling after reheat?
            # Or just break and do more compaction?
            # Let's try reheating a few times.
            if steps > 50000:
                break
        
        # Generate Move
        move_type = random.random()
        
        # Pick a random element (row_idx, pos_idx) where pos_idx > 0
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, len(current_rows[r_idx]) - 1)
        
        old_val = current_rows[r_idx][p_idx]
        
        # Calculate diffs to remove
        row = current_rows[r_idx]
        diffs_to_remove = []
        for i in range(len(row)):
            if i != p_idx:
                diffs_to_remove.append(abs(row[i] - old_val))
        
        # Remove old val from row and diff_counts
        for d in diffs_to_remove:
            remove_diff(d)
        
        current_rows[r_idx].pop(p_idx)
        
        # Determine new value and position
        new_val = None
        new_pos = None
        new_r_idx = r_idx
        
        if move_type < 0.3:
            # Shift within same row
            # Try to move to adjacent position? Or swap?
            # Let's try swapping with another element in the same row
            if len(row) > 1:
                other_p = random.randint(0, len(row) - 1)
                # Swap values
                val2 = current_rows[r_idx][other_p]
                # We already popped old_val.
                # We need to re-insert old_val at other_p? No, swap means exchange.
                # This logic is getting complex with pop.
                # Let's stick to simpler moves:
                # 1. Change value in same position (random nearby)
                # 2. Move element to different position in same row
                # 3. Move element to different row
                
                # Let's abort this specific move type if it's complex and just do value change
                pass
        
        # Simplified Move Set:
        # 1. Value Change: Pick a new random value in a range.
        # 2. Row Move: Pick a different row and a position.
        
        # Let's restart the move generation logic to be cleaner.
        # We already popped old_val.
        
        if move_type < 0.5:
            # Value Change
            # Search range: [1, best_scope + 10]
            # Bias towards smaller values
            max_search = min(best_scope + 20, 200)
            # Try a few random candidates
            found = False
            for _ in range(5):
                cand = random.randint(1, max_search)
                # Check if cand is already in the row
                if cand in current_rows[r_idx]:
                    continue
                
                # Check conflicts
                # Tentatively add cand
                valid = True
                new_diffs = []
                for a in current_rows[r_idx]:
                    d = abs(cand - a)
                    if d in diff_counts and diff_counts[d] >= 1:
                        # It's a conflict if it's already used by ANOTHER pair
                        # Wait, diff_counts includes the diffs of the current row?
                        # We removed old_val's diffs. So diff_counts reflects the state WITHOUT old_val.
                        # So if d is in diff_counts, it means some OTHER pair uses d.
                        # So cand would create a conflict.
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    # Find position
                    pos = 0
                    while pos < len(current_rows[r_idx]) and current_rows[r_idx][pos] < cand:
                        pos += 1
                    current_rows[r_idx].insert(pos, cand)
                    for d in new_diffs:
                        add_diff(d)
                    new_val = cand
                    found = True
                    break
            
            if not found:
                # Revert
                # Put old_val back
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
            if new_r_idx == r_idx:
                # Move to different position in same row
                # Find a random valid position
                # This is hard to do efficiently.
                # Let's just try to insert at a random position in the same row
                # But we already popped it.
                # We can try to re-insert at a different position.
                # However, the value might be invalid.
                # Let's just try to change the value in the same row (like above) but maybe shift?
                # For simplicity, if move_type > 0.5, we pick a new row.
                # If it ends up being the same row, we treat it as a value change attempt?
                # No, let's force new_r_idx != r_idx
                while new_r_idx == r_idx:
                    new_r_idx = random.randint(0, n_rows - 1)
            
            # Try to insert old_val into new_r_idx
            # Or generate a new value?
            # Moving the same value is a good move to explore row interactions.
            cand = old_val
            
            if cand in current_rows[new_r_idx]:
                # Conflict with existing element in new row
                # Try a nearby value
                cand = random.randint(1, best_scope + 10)
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
                if d in diff_counts and diff_counts[d] >= 1:
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
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
                # If we found a valid solution with better scope, we can update lb?
                # No, lb is fixed.
                
        else:
            # Revert
            # Remove new_val from new_r_idx
            if new_r_idx != r_idx:
                # Find and remove new_val from new_r_idx
                # It should be there
                row_new = current_rows[new_r_idx]
                # Find index
                idx_new = row_new.index(new_val)
                row_new.pop(idx_new)
                # Remove diffs
                for a in row_new:
                    remove_diff(abs(a - new_val))
            else:
                # Same row, different value?
                # In our logic, if move_type < 0.5, we inserted cand into r_idx.
                # If move_type >= 0.5, we inserted cand into new_r_idx (which != r_idx).
                # So if we are reverting, we need to undo the insertion.
                # If new_r_idx == r_idx, it means we did a value change.
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
    if best_rows is not None and best_scope > lb:
        # Use the old try_compact_rows logic on the best solution found
        used_set = get_used_diffs(best_rows)
        # We need to adapt try_compact_rows to not use nonlocal best_rows if we are just local
        # Let's just run a few iterations of the old compaction
        # Re-implementing a quick compaction here to save space/time
        
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
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]

        quick_compact(best_rows, used_set, 200)

    if best_rows is None:
        # Fallback
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
    
    return best_rows
