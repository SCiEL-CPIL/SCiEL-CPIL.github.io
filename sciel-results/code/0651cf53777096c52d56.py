import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before the 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Simple counting bound lower bound for scope
    # We need total_diffs distinct differences in [1, M], so M >= total_diffs
    # But also each row has k+1 elements, max element is at least k (if 0,1,2...k)
    # The "simple counting bound" mentioned in the task is 112 for n_rows=4, k=7.
    # total_diffs = 4 * 28 = 112. So M >= 112 is a very tight lower bound.
    
    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        """
        Counts the number of excess duplicates (collisions).
        """
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def get_all_diffs_set(rows):
        s = set()
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    s.add(r[j] - r[i])
        return s

    def greedy_init(rand_prob=0.1, window=20):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 300 
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
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

    def local_search_targeted(rows, time_limit):
        """
        Structural refinement: Targeted descent to minimize the global maximum (scope).
        We repeatedly identify the largest element in the entire configuration.
        We attempt to replace it with a smaller valid value.
        If successful, the scope might decrease.
        If stuck, we perform random perturbations to escape local minima.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        current_conflicts = count_conflicts(best_rows)
        
        while time.time() < start + time_limit:
            # 1. Try to reduce the global maximum
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
                break # Can't change 0
            
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            
            # Build set of all diffs
            all_diffs = set()
            for r in best_rows:
                for i in range(row_len):
                    for j in range(i + 1, row_len):
                        all_diffs.add(r[j] - r[i])
            
            # Diffs to remove (involving target)
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, row_len):
                remove_set.add(row[j] - current_val)
            
            fixed_diffs = all_diffs - remove_set
            
            found = False
            # Try values from current_val - 1 down to prev_val + 1
            # We want to find the smallest valid value to minimize scope, 
            # but reducing max is priority.
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                # Check diffs with left neighbors
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Check diffs with right neighbors
                for j in range(max_c_idx + 1, row_len):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check for duplicates within new diffs
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, row_len)]
                
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                # Valid!
                row[max_c_idx] = new_val
                found = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                break
            
            if not found:
                # 2. If we can't reduce the current max, try a random swap on a non-max element to "jolt" the system.
                # Pick a random row and column (not 0)
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)
                
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx - 1]
                next_v = best_rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 100
                
                # Re-compute fixed diffs for this specific change
                all_diffs = set()
                for r in best_rows:
                    for i in range(row_len):
                        for j in range(i + 1, row_len):
                            all_diffs.add(r[j] - r[i])
                
                remove_set = set()
                row = best_rows[r_idx]
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, row_len):
                    remove_set.add(row[j] - curr_val)
                
                fixed_diffs = all_diffs - remove_set
                
                # Try random new values
                for _ in range(10):
                    # Bias towards smaller values to potentially help reduce scope later
                    low = prev_v + 1
                    high = min(next_v - 1, curr_val + 10)
                    if low >= high:
                        continue
                    new_val = random.randint(low, high)
                    
                    valid = True
                    for i in range(c_idx):
                        d = new_val - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    for j in range(c_idx + 1, row_len):
                        d = row[j] - new_val
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_left_diffs = [new_val - row[i] for i in range(c_idx)]
                    new_right_diffs = [row[j] - new_val for j in range(c_idx + 1, row_len)]
                    all_new = new_left_diffs + new_right_diffs
                    if len(set(all_new)) != len(all_new):
                        continue
                    
                    row[c_idx] = new_val
                    break # Accept one move

        return best_rows

    # Main Loop: Multiple Restarts
    best_overall = None
    best_overall_score = float('inf')
    
    # We have ~135 seconds.
    # Let's do a few restarts.
    # Each restart: Greedy Init -> Local Search
    # We can also use the GA idea if we have time, but the inspiration suggests targeted local search is strong.
    # Let's stick to multiple restarts of (Greedy + Targeted Local Search) as it's robust and simple.
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        
        # Vary the greedy randomness
        # Early restarts: more deterministic (lower rand_prob)
        # Later restarts: more random (higher rand_prob)
        rand_prob = min(0.5, 0.05 * restarts)
        
        initial_rows = greedy_init(rand_prob=rand_prob, window=20)
        
        # Allocate time for local search
        # If we have lots of time left, give it more.
        remaining = deadline - time.time()
        ls_time = min(remaining - 5, 10.0) # Cap at 10s per restart to allow many restarts
        
        if ls_time <= 1:
            break
            
        refined_rows = local_search_targeted(initial_rows, ls_time)
        
        scope = get_scope(refined_rows)
        conflicts = count_conflicts(refined_rows)
        
        # Score: primarily scope, but only if valid. If invalid, score is huge.
        if conflicts > 0:
            score = scope + 1000000
        else:
            score = scope
            
        if score < best_overall_score:
            best_overall_score = score
            best_overall = [r[:] for r in refined_rows]
            
    return best_overall
