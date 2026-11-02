import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 145s to be safe.
    deadline = start_time + 145.0
    
    # Force deterministic behavior for reproducibility
    random.seed(42)

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
                limit = 300 
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
                break
            
            if not found_improvement:
                stuck_counter += 1
                
                # If stuck for a while, try a more aggressive perturbation: Row Reconstruction
                if stuck_counter > 20 and random.random() < 0.3:
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
                    success = False
                    for attempt in range(5):
                        temp_row = [0]
                        temp_used = set()
                        
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
                        
                        if max(temp_row) <= best_scope:
                            best_rows[r_idx] = temp_row
                            success = True
                            break
                        else:
                            pass

                    if success:
                        stuck_counter = 0
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
    local_time = 20.0
    if time.time() > deadline - 100:
        local_time = deadline - time.time() - 10
        
    if local_time > 5 and best_rows:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 3: Backtracking to try to beat the local search result
    # This is the structural change: A dedicated backtracking solver with forward checking
    # to systematically reduce the scope.
    
    def backtracking_search(target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses DFS with forward checking.
        """
        random.seed(seed)
        start = time.time()
        
        rows = [[0] * row_len for _ in range(n_rows)]
        used_diffs = set()
        
        # Pre-compute max depth for pruning
        max_depth = n_rows * (row_len - 1) # Total elements to place (excluding 0s)
        
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
            
            # Pruning: Check if we have enough room to place remaining elements in this row
            # Remaining elements in this row: row_len - pos
            # Minimum span needed: (row_len - pos) - 1
            if upper - prev_val < (row_len - pos) - 1:
                return False

            # Generate candidates
            # We prioritize smaller values to keep the row tight, but we must check validity.
            # To improve efficiency, we can check a window of candidates.
            candidates = []
            
            # Scan up to 50 candidates to avoid huge lists, but ensure we don't miss a valid tight solution
            # Since we want to minimize scope, trying small values first is good.
            for val in range(lower, upper + 1):
                valid = True
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    candidates.append((val, new_diffs))
                    if len(candidates) >= 50:
                        break
            
            if not candidates:
                return False
            
            # Shuffle slightly to break symmetry if needed, but for minimization, sorted is usually best
            # However, if we just take the first 50, we might miss a solution that requires a slightly larger
            # early value to allow a much smaller later value? Unlikely for strictly increasing.
            # So we try them in increasing order.
            
            for val, new_diffs in candidates:
                # Add diffs
                for d in new_diffs:
                    used_diffs.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                # Backtrack
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    current_target = best_scope - 1
    seed_counter = 0
    
    # Allocate remaining time to backtracking
    # We try to reduce the scope by 1 each time we find a solution.
    # If we fail to find a solution for a target, we stop (assuming it's hard).
    # To be safe, we limit the number of attempts or time per attempt.
    
    while time.time() < deadline - 5:
        if current_target < 112: # Theoretical lower bound
            break
            
        # Give each backtracking attempt a reasonable time slice
        # If we are close to the limit, give it the rest.
        search_time = 10.0
        if time.time() > deadline - 20:
            search_time = deadline - time.time() - 2
            
        if search_time < 1.0:
            break

        result = backtracking_search(current_target, search_time, seed=seed_counter)
        seed_counter += 1
        
        if result:
            if validate(result):
                new_scope = get_scope(result)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = result
                    current_target = best_scope - 1
                else:
                    # Found a solution but not better? Shouldn't happen if target < best_scope
                    break
            else:
                break
        else:
            # No solution found for current_target within time.
            # It's possible a solution exists but we didn't find it.
            # We can try a few more seeds, but if it's hard, we should stop.
            # Let's try 2 more quick attempts with different seeds if time permits.
            if time.time() < deadline - 10:
                found_any = False
                for _ in range(2):
                    if time.time() > deadline - 5:
                        break
                    res_retry = backtracking_search(current_target, 5.0, seed=seed_counter)
                    seed_counter += 1
                    if res_retry and validate(res_retry):
                        new_scope = get_scope(res_retry)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = res_retry
                            current_target = best_scope - 1
                            found_any = True
                            break
                if not found_any:
                    # If we can't beat it, and we've spent significant time, we stop.
                    # Since backtracking is expensive, failing once suggests the bound is tight.
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
