import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 135s to be safe.
    deadline = start_time + 135.0

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
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
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
        Iterated Local Search (ILS) with Targeted Descent.
        1. Try to reduce the global maximum.
        2. If stuck, perturb a random element to escape local minimum.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        # Helper to get all diffs for a configuration
        def get_all_diffs(rows):
            diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs

        # We will maintain a set of used diffs for the current state to speed up checks
        # But since we perturb, we might need to recompute or update carefully.
        # For simplicity and safety in this implementation, we recompute the 'fixed' diffs 
        # when we decide to move an element. This is O(N^2) per move, which is acceptable 
        # for small N (4 rows, 8 cols).
        
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
                # Can't reduce 0
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
                # Check diffs with left neighbors
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Check diffs with right neighbors
                for j in range(max_c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check for duplicates within new diffs
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, len(row))]
                
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                # Valid improvement
                row[max_c_idx] = new_val
                found_improvement = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                break
            
            if not found_improvement:
                # 2. Perturbation: If we can't reduce the max, we are likely stuck.
                # Pick a random element (not the first) and move it to a random valid position.
                # This might increase the scope temporarily, but it changes the structure.
                
                r_idx = random.randint(0, n_rows-1)
                c_idx = random.randint(1, row_len-1)
                
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx-1]
                next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                
                # Re-compute fixed diffs for this element
                all_diffs = get_all_diffs(best_rows)
                remove_set = set()
                row = best_rows[r_idx]
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove_set.add(row[j] - curr_val)
                
                fixed_diffs = all_diffs - remove_set
                
                # Try to find a valid new value
                # We want to try values that are close to the current one to maintain structure,
                # or random values to explore more.
                # Let's try a few random candidates in a range.
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                
                # If the range is too small, expand it slightly or skip
                if low > high:
                    continue
                
                # Try 10 random candidates
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
                    
                    # Apply the perturbation
                    row[c_idx] = cand
                    applied = True
                    break
                
                if not applied:
                    # If we can't even perturb, we might be in a very tight spot.
                    # Try a different random element or break.
                    # For now, break to avoid infinite loop if stuck completely.
                    # But usually, there's some freedom.
                    pass

        return best_rows, best_scope

    # Phase 1: Diverse Initialization
    best_rows = None
    best_scope = float('inf')
    
    # Run several greedy initializations with different seeds
    num_inits = 10
    init_time_limit = 5.0 # 5 seconds for init
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
    # Allocate a significant chunk of time to local search to tighten the solution.
    local_time = 60.0
    if time.time() > deadline - 70:
        local_time = deadline - time.time() - 10
        
    if local_time > 10 and best_rows:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 3: Backtracking to try to beat the local search result
    # We use the local search result as a guide for the target scope.
    # We try to find a solution with scope < best_scope.
    
    current_target = best_scope - 1
    seed_counter = 0
    
    # Define backtracking search
    def backtracking_search(n_rows, k, target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking.
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
            # We want to try smaller values first to keep the scope low
            # But we also need to ensure we don't paint ourselves into a corner.
            # Heuristic: Try values in a random order, but biased towards smaller ones?
            # Or just random shuffle.
            
            # To speed up, we can limit the number of candidates we try per node
            # if the search space is too large, but for small N, we can try more.
            
            # Generate candidates from lower to upper
            # If the range is large, we might want to sample.
            # For now, iterate all if range is small, else sample.
            
            range_size = upper - lower + 1
            if range_size < 50:
                cand_range = range(lower, upper + 1)
            else:
                # Sample a subset
                # This is risky for completeness, but backtracking is a last resort.
                # Actually, for correctness, we should check all valid candidates.
                # But if range is large, checking all is slow.
                # Let's just iterate all, but break early if we find a solution.
                cand_range = range(lower, upper + 1)
            
            for val in cand_range:
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    # Found a candidate
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
            
        # Adaptive time for backtracking
        # If we are close to the deadline, use less time.
        search_time = 5.0
        if time.time() > deadline - 15:
            search_time = deadline - time.time() - 2
            
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
                    # Found a valid solution but scope didn't improve? 
                    # This implies the search found a solution with scope <= target, 
                    # but maybe target was not tight enough or we made a mistake.
                    break
            else:
                # Invalid result from backtracking? Should not happen.
                break
        else:
            # Failed to find solution with current_target
            # Try one retry with different seed
            if time.time() < deadline - 10:
                result_retry = backtracking_search(n_rows, k, current_target, 5.0, seed=seed_counter)
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
    # Simple test
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    # Verify
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
