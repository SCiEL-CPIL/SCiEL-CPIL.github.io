import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 132s to be safe.
    deadline = start_time + 132.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check if rows are valid (strictly increasing, start at 0, unique diffs)."""
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.1, window=20, seed=None):
        """
        Generates a valid configuration using a randomized greedy approach.
        Optimized: precompute row elements for O(1) diff checks.
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                row_set = set(row)
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def local_search(rows, time_limit):
        """
        Iterated Local Search (ILS) with Targeted Descent and Row Reconstruction.
        1. Try to reduce the global maximum via single-element moves.
        2. If stuck, perturb a random element.
        3. If stuck longer, reconstruct a random row entirely to escape deep local minima.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        def get_all_diffs(rows):
            diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs

        stuck_counter = 0
        no_improve_counter = 0

        while time.time() < start + time_limit:
            # 1. Targeted Descent: Try to reduce the global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break
            
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            
            # Build set of all diffs
            all_diffs = get_all_diffs(best_rows)
            
            # Diffs to remove (involving target)
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, len(row)):
                remove_set.add(row[j] - current_val)
            
            fixed_diffs = all_diffs - remove_set
            
            found_improvement = False
            # Try values from current_val - 1 down to prev_val + 1
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                    
                for j in range(max_c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, len(row))]
                
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                row[max_c_idx] = new_val
                found_improvement = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    stuck_counter = 0
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                break
            
            if not found_improvement:
                stuck_counter += 1
                no_improve_counter += 1
                
                # If stuck for a while, try a more aggressive perturbation: Row Reconstruction
                if (stuck_counter > 20 or no_improve_counter > 50) and random.random() < 0.3:
                    # Pick a random row to reconstruct
                    r_idx = random.randint(0, n_rows - 1)
                    
                    # Remove the row from consideration
                    other_rows = [best_rows[i] for i in range(n_rows) if i != r_idx]
                    
                    # Compute diffs used by other rows
                    fixed_diffs = set()
                    for r in other_rows:
                        for i in range(len(r)):
                            for j in range(i+1, len(r)):
                                fixed_diffs.add(r[j] - r[i])
                    
                    # Greedily reconstruct the row
                    new_row = [0]
                    # Try to keep the last element close to the original to minimize scope impact
                    # Or just build from scratch. Let's build from scratch but biased towards small values.
                    
                    # We can try a few random seeds for this reconstruction
                    success = False
                    for attempt in range(5):
                        temp_row = [0]
                        temp_used = set()
                        
                        # We must avoid fixed_diffs
                        # We also need to ensure internal uniqueness
                        
                        valid_row = True
                        while len(temp_row) < row_len:
                            last = temp_row[-1]
                            candidates = []
                            x = last + 1
                            # Limit search scope to current best_scope or slightly less
                            limit = min(best_scope, last + 50)
                            
                            while x <= limit and len(candidates) < 20:
                                valid = True
                                for a in temp_row:
                                    d = x - a
                                    if d in fixed_diffs or d in temp_used:
                                        valid = False
                                        break
                                if valid:
                                    candidates.append(x)
                                x += 1
                            
                            if not candidates:
                                # Expand search
                                while True:
                                    x += 1
                                    valid = True
                                    for a in temp_row:
                                        d = x - a
                                        if d in fixed_diffs or d in temp_used:
                                            valid = False
                                            break
                                    if valid:
                                        candidates.append(x)
                                        break
                            
                            if random.random() < 0.2 and len(candidates) > 1:
                                x = random.choice(candidates)
                            else:
                                x = candidates[0]
                            
                            temp_row.append(x)
                            for a in temp_row[:-1]:
                                temp_used.add(x - a)
                        
                        # Check if this new row is valid with respect to fixed_diffs
                        # (It is by construction, but let's double check internal uniqueness)
                        # temp_used ensures internal uniqueness.
                        # fixed_diffs check ensures global uniqueness.
                        
                        # If the new row's max is less than or equal to best_scope, we accept it
                        # Even if it doesn't improve the global max immediately, it might allow future improvements.
                        # However, to be safe, we only accept if it doesn't increase the global max.
                        if max(temp_row) <= best_scope:
                            best_rows[r_idx] = temp_row
                            success = True
                            break
                        else:
                            # If it increases scope, we might still accept it if it's a "perturbation" 
                            # but strictly speaking, we want to minimize.
                            # Let's only accept if it's strictly better or equal.
                            pass

                    if success:
                        stuck_counter = 0
                        no_improve_counter = 0
                        # Re-evaluate scope
                        new_scope = get_scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                else:
                    # Standard perturbation: Move a random element
                    r_idx = random.randint(0, n_rows-1)
                    c_idx = random.randint(1, row_len-1)
                    
                    curr_val = best_rows[r_idx][c_idx]
                    prev_v = best_rows[r_idx][c_idx-1]
                    next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                    
                    all_diffs = get_all_diffs(best_rows)
                    remove_set = set()
                    row = best_rows[r_idx]
                    for i in range(c_idx):
                        remove_set.add(curr_val - row[i])
                    for j in range(c_idx + 1, len(row)):
                        remove_set.add(row[j] - curr_val)
                    
                    fixed_diffs = all_diffs - remove_set
                    
                    low = prev_v + 1
                    high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                    
                    if low > high:
                        continue
                    
                    applied = False
                    for _ in range(10):
                        cand = random.randint(low, high)
                        if cand == curr_val:
                            continue
                        
                        valid = True
                        for i in range(c_idx):
                            d = cand - row[i]
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        for j in range(c_idx + 1, len(row)):
                            d = row[j] - cand
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_left = [cand - row[i] for i in range(c_idx)]
                        new_right = [row[j] - cand for j in range(c_idx+1, len(row))]
                        if len(set(new_left + new_right)) != len(new_left + new_right):
                            continue
                        
                        row[c_idx] = cand
                        applied = True
                        break
                    
                    if not applied:
                        pass

        return best_rows, best_scope

    def polish_pass(rows, time_limit):
        """
        Final high-precision polish pass: exhaustively try to reduce each element.
        For each element (in order of largest first), try all smaller values that
        maintain validity. This is a local search with tighter tolerance.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        def get_all_diffs(rws):
            diffs = set()
            for r in rws:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs
        
        # Sort elements by value descending to prioritize reducing large values
        while time.time() < start + time_limit:
            # Find all elements sorted by value descending
            elements = []
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if c_idx > 0:  # skip 0
                        elements.append((val, r_idx, c_idx))
            
            if not elements:
                break
            
            elements.sort(reverse=True)
            
            improved = False
            for val, r_idx, c_idx in elements:
                if time.time() > start + time_limit:
                    break
                
                row = best_rows[r_idx]
                prev_val = row[c_idx - 1]
                next_val = row[c_idx + 1] if c_idx < row_len - 1 else val + 100
                
                # Build all diffs
                all_diffs = get_all_diffs(best_rows)
                
                # Diffs to remove (involving target)
                remove_set = set()
                for i in range(c_idx):
                    remove_set.add(val - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove_set.add(row[j] - val)
                
                fixed_diffs = all_diffs - remove_set
                
                # Try values from val-1 down to prev_val+1
                for new_val in range(val - 1, prev_val, -1):
                    valid = True
                    for i in range(c_idx):
                        d = new_val - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - new_val
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_left = [new_val - row[i] for i in range(c_idx)]
                    new_right = [row[j] - new_val for j in range(c_idx+1, len(row))]
                    if len(set(new_left + new_right)) != len(new_left + new_right):
                        continue
                    
                    row[c_idx] = new_val
                    improved = True
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    break
            
            if not improved:
                break
        
        return best_rows, best_scope

    # Phase 1: Diverse Initialization
    best_rows = None
    best_scope = float('inf')
    
    # Run several greedy initializations with different seeds
    num_inits = 10
    init_time_limit = 5.0
    start_init = time.time()
    
    for i in range(num_inits):
        if time.time() > start_init + init_time_limit:
            break
        rows = greedy_init(rand_prob=0.1 + 0.01*i, window=20, seed=i)
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    # Phase 2: Local Search Refinement
    local_time = 60.0
    if time.time() > deadline - 70:
        local_time = deadline - time.time() - 10
        
    if local_time > 10 and best_rows:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 2.5: Polish pass for final optimization
    polish_time = min(20.0, max(0.0, deadline - time.time() - 15.0))
    if polish_time > 5.0 and best_rows:
        polished_rows, polished_scope = polish_pass(best_rows, polish_time)
        if validate(polished_rows):
            best_rows = polished_rows
            best_scope = polished_scope

    # Phase 3: Backtracking to try to beat the local search result
    current_target = best_scope - 1
    seed_counter = 0
    
    # Reserve time for backtracking
    backtracking_time_remaining = deadline - time.time() - 10.0
    if backtracking_time_remaining < 10.0:
        return best_rows
    
    def backtracking_search(n_rows, k, target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking with candidate windowing.
        """
        random.seed(seed)
        start = time.time()
        
        rows = [[0] * row_len for _ in range(n_rows)]
        used_diffs = set()
        
        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == row_len:
                return solve(row_idx + 1, 1)
            
            prev_val = rows[row_idx][pos - 1]
            lower = prev_val + 1
            upper = target_scope
            
            if lower > upper:
                return False
            
            candidates = []
            # Scan a window first
            scan_limit = min(upper, lower + 50)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 10:
                        break
            
            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 10:
                            break
            
            if not candidates:
                return False
            
            random.shuffle(candidates)
            
            for val in candidates:
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Run backtracking attempts
    while time.time() < deadline - 5:
        if current_target < 112:
            break
        
        remaining = deadline - time.time() - 5.0
        if remaining < 1.0:
            break
            
        search_time = min(5.0, remaining)
        if search_time < 1.0:
            break

        result = backtracking_search(n_rows, k, current_target, search_time, seed=seed_counter)
        seed_counter += 1
        
        if result:
            if validate(result):
                new_scope = get_scope(result)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = result
                    current_target = best_scope - 1
                else:
                    break
            else:
                break
        else:
            retry_remaining = deadline - time.time() - 5.0
            if retry_remaining > 1.0:
                result_retry = backtracking_search(n_rows, k, current_target, min(5.0, retry_remaining), seed=seed_counter)
                seed_counter += 1
                if result_retry and validate(result_retry):
                    new_scope = get_scope(result_retry)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = result_retry
                        current_target = best_scope - 1
                    else:
                        break
                else:
                    break
            else:
                break

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
