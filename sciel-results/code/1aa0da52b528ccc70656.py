import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n_marks = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Helper to get differences for a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(n_marks):
            for j in range(i + 1, n_marks):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to compute used diffs excluding a specific row
    def get_used_diffs_excluding(rows, exclude_row):
        used = set()
        for r in range(n_rows):
            if r == exclude_row:
                continue
            used |= get_row_diffs(rows[r])
        return used

    # Greedy initialization
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

    # Compute cost: prioritize no conflicts, then minimize max entry
    def compute_cost(rows):
        used = set()
        max_entry = 0
        conflicts = 0
        for r in range(n_rows):
            row = rows[r]
            if row[-1] > max_entry:
                max_entry = row[-1]
            for i in range(n_marks):
                for j in range(i + 1, n_marks):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    used.add(d)
        # Heavily penalize conflicts to ensure validity first
        return conflicts * 10000 + max_entry

    # Rebuild a single row using backtracking, given other rows are fixed
    # Returns a new row list or None if no valid row found within max_val_bound
    def rebuild_row(row_idx, rows, max_val_bound=None):
        if max_val_bound is None:
            max_val_bound = rows[row_idx][-1] + 20 # Allow some growth
        
        used_diffs = get_used_diffs_excluding(rows, row_idx)
        
        new_row = [0]
        
        def backtrack(pos, last_val):
            if pos == n_marks:
                return True
            
            # Heuristic: try smaller values first to keep scope small
            # But we need to ensure we can fill the rest.
            # Simple bound: we need k - pos + 1 more marks.
            # Max possible value is max_val_bound.
            # Min possible next value is last_val + 1.
            
            start_val = last_val + 1
            # We can prune if start_val is too high
            if start_val > max_val_bound:
                return False
            
            # Try values from start_val to max_val_bound
            # To be faster, we can limit the range or use randomization if stuck
            # But for k=8, linear scan is usually fine if we prune well.
            
            for v in range(start_val, max_val_bound + 1):
                # Check conflicts with new_row
                new_diffs = {v - a for a in new_row}
                if new_diffs & used_diffs:
                    continue
                # Check internal conflicts (new_diffs must be unique)
                if len(new_diffs) != len(new_row):
                    continue
                
                new_row.append(v)
                # Recurse
                if backtrack(pos + 1, v):
                    return True
                new_row.pop()
                
            return False
        
        if backtrack(1, 0):
            return new_row
        return None

    # Simulated Annealing with Conflict-Based Moves
    def simulated_annealing(rows, time_budget):
        current_rows = [list(r) for r in rows]
        current_cost = compute_cost(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        
        T_start = 500.0
        T_end = 1.0
        steps = 0
        max_steps = 100000
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_steps - 1))
        
        while time.time() < time_budget and steps < max_steps:
            steps += 1
            T = T_start * (cooling_rate ** steps)
            if T < T_end:
                T = T_end
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Conflict-based move: Find a conflict and try to resolve it
                # Get all diffs and their sources
                diff_sources = {}
                for r in range(n_rows):
                    row = current_rows[r]
                    for i in range(n_marks):
                        for j in range(i + 1, n_marks):
                            d = row[j] - row[i]
                            if d not in diff_sources:
                                diff_sources[d] = []
                            diff_sources[d].append((r, i, j))
                
                conflicts = [d for d, srcs in diff_sources.items() if len(srcs) > 1]
                
                if conflicts:
                    d = random.choice(conflicts)
                    srcs = diff_sources[d]
                    # Pick two sources that share this diff
                    # Actually, pick one source and try to move one of its marks
                    # To resolve conflict d, we need to change one of the marks in one of the pairs
                    # Let's pick a random pair (r, i, j) from srcs
                    r, i, j = random.choice(srcs)
                    
                    # Try to move row[r][i] or row[r][j]
                    # Moving j is usually easier as it affects more future diffs?
                    # Let's try moving j (the larger index)
                    pos = j
                    old_val = current_rows[r][pos]
                    
                    min_val = current_rows[r][pos - 1] + 1
                    max_val = current_rows[r][pos + 1] - 1 if pos < k else old_val + 10
                    
                    # Try a few random values in range
                    found_better = False
                    for _ in range(10):
                        new_val = random.randint(min_val, max_val)
                        if new_val == old_val:
                            continue
                        if new_val < min_val or new_val > max_val:
                            continue
                        
                        current_rows[r][pos] = new_val
                        new_cost = compute_cost(current_rows)
                        
                        delta_cost = new_cost - current_cost
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 1e-10)):
                            current_cost = new_cost
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_rows = [list(row) for row in current_rows]
                            found_better = True
                            break
                        else:
                            current_rows[r][pos] = old_val
                    
                    if not found_better:
                        current_rows[r][pos] = old_val
                else:
                    # No conflicts, try to reduce max entry
                    # Find row with max entry
                    max_r = max(range(n_rows), key=lambda r: current_rows[r][-1])
                    # Try to compress this row
                    # Pick a random pos > 0 and try to decrease it
                    pos = random.randint(1, k)
                    old_val = current_rows[max_r][pos]
                    min_val = current_rows[max_r][pos - 1] + 1
                    max_val = old_val - 1 # Try to decrease
                    if min_val <= max_val:
                        new_val = random.randint(min_val, max_val)
                        current_rows[max_r][pos] = new_val
                        new_cost = compute_cost(current_rows)
                        delta_cost = new_cost - current_cost
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 1e-10)):
                            current_cost = new_cost
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_rows = [list(row) for row in current_rows]
                        else:
                            current_rows[max_r][pos] = old_val

            elif move_type < 0.7:
                # Rebuild a row
                r = random.randint(0, n_rows - 1)
                # Try to rebuild with a slightly larger bound to allow escape, 
                # but prefer smaller
                # Use current max as a soft bound, allow +10
                bound = current_rows[r][-1] + 10
                new_row = rebuild_row(r, current_rows, bound)
                if new_row:
                    old_row = current_rows[r]
                    current_rows[r] = new_row
                    new_cost = compute_cost(current_rows)
                    delta_cost = new_cost - current_cost
                    if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 1e-10)):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_rows = [list(row) for row in current_rows]
                    else:
                        current_rows[r] = old_row

            else:
                # Standard random move
                r = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                old_val = current_rows[r][pos]
                min_val = current_rows[r][pos - 1] + 1
                max_val = current_rows[r][pos + 1] - 1 if pos < k else old_val + 20
                
                if min_val > max_val:
                    continue
                
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
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

        return best_rows, best_cost

    # Local Search: Targeted conflict resolution and compression
    def local_search(rows, time_budget):
        current_rows = [list(r) for r in rows]
        current_cost = compute_cost(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < time_budget and no_improve < max_no_improve:
            improved = False
            
            # 1. Resolve conflicts
            diff_sources = {}
            for r in range(n_rows):
                row = current_rows[r]
                for i in range(n_marks):
                    for j in range(i + 1, n_marks):
                        d = row[j] - row[i]
                        if d not in diff_sources:
                            diff_sources[d] = []
                        diff_sources[d].append((r, i, j))
            
            conflicts = [d for d, srcs in diff_sources.items() if len(srcs) > 1]
            
            if conflicts:
                # Try to fix each conflict
                for d in conflicts:
                    if time.time() > time_budget: break
                    srcs = diff_sources[d]
                    # Pick a random source
                    r, i, j = random.choice(srcs)
                    # Try to move j
                    pos = j
                    old_val = current_rows[r][pos]
                    min_val = current_rows[r][pos - 1] + 1
                    max_val = current_rows[r][pos + 1] - 1 if pos < k else old_val + 5
                    
                    best_local_val = old_val
                    best_local_cost = current_cost
                    
                    # Try a range of values
                    for v in range(min_val, min(max_val, old_val + 5) + 1):
                        if v == old_val: continue
                        current_rows[r][pos] = v
                        new_cost = compute_cost(current_rows)
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_val = v
                            
                    current_rows[r][pos] = best_local_val
                    if best_local_cost < current_cost:
                        current_cost = best_local_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_rows = [list(row) for row in current_rows]
                        no_improve = 0
                        break # Restart conflict scan if improved
            
            # 2. If no conflicts, try to reduce max entry
            else:
                max_r = max(range(n_rows), key=lambda r: current_rows[r][-1])
                # Try to compress the max row
                for pos in range(k, 0, -1): # From end to start
                    if time.time() > time_budget: break
                    old_val = current_rows[max_r][pos]
                    min_val = current_rows[max_r][pos - 1] + 1
                    max_val = old_val - 1
                    if min_val > max_val: continue
                    
                    best_local_val = old_val
                    best_local_cost = current_cost
                    
                    for v in range(max_val, min_val - 1, -1):
                        current_rows[max_r][pos] = v
                        new_cost = compute_cost(current_rows)
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_val = v
                            
                    current_rows[max_r][pos] = best_local_val
                    if best_local_cost < current_cost:
                        current_cost = best_local_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_rows = [list(row) for row in current_rows]
                        no_improve = 0
                        break # Restart if improved
            
            if not improved:
                no_improve += 1
                # If stuck, try a rebuild on the worst row
                if no_improve % 10 == 0:
                    max_r = max(range(n_rows), key=lambda r: current_rows[r][-1])
                    bound = current_rows[max_r][-1] + 5
                    new_row = rebuild_row(max_r, current_rows, bound)
                    if new_row:
                        old_row = current_rows[max_r]
                        current_rows[max_r] = new_row
                        new_cost = compute_cost(current_rows)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_rows = [list(row) for row in current_rows]
                            no_improve = 0
                        else:
                            current_rows[max_r] = old_row

        return best_rows, best_cost

    # Get initial solution
    greedy_rows = greedy_init()
    
    # Run simulated annealing
    sa_rows, sa_cost = simulated_annealing(greedy_rows, time.time() + 50)
    
    # Run local search from the SA result
    ls_rows, ls_cost = local_search(sa_rows, time.time() + 100)
    
    # Final validation
    for r in range(n_rows):
        for m in range(1, n_marks):
            if ls_rows[r][m] <= ls_rows[r][m - 1]:
                ls_rows[r][m] = ls_rows[r][m - 1] + 1
                
    return ls_rows
