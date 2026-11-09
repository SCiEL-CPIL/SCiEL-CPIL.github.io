import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    k_plus_1 = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True

    def compute_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_solution(seed, target_max=None):
        random.seed(seed)
        used = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                x = current_max + 1
                if target_max is not None:
                    search_limit = target_max
                else:
                    search_limit = x + 500 + row_idx * 50
                
                candidates = []
                while x <= search_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        candidates.append(x)
                        if len(candidates) > 5:
                            break
                    x += 1
                
                if not candidates:
                    if target_max is not None:
                        return None
                    else:
                        # Unbounded search
                        x = current_max + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & used) and len(new_diffs) == len(row):
                                candidates.append(x)
                                break
                            x += 1
                            if time.time() - start_time > time_limit - 10:
                                return None
                
                candidates.sort()
                # Randomized choice to create diversity
                # Pick from top few
                top_cands = candidates[:min(10, len(candidates))]
                chosen = random.choice(top_cands)
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
            
            if time.time() - start_time > time_limit - 20:
                return None
        
        return rows

    def local_search(rows, time_budget):
        """
        Attempts to reduce the maximum element by checking if it can be replaced
        by a smaller value that maintains validity.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        end_time = time.time() + time_budget
        
        used_diffs = set()
        row_diffs = []
        for r in rows:
            rd = compute_row_diffs(r)
            row_diffs.append(rd)
            used_diffs.update(rd)
            
        # Identify the row and index of the global maximum
        # There might be multiple, we pick one arbitrarily or the first one found
        max_val = best_score
        target_row_idx = -1
        target_col_idx = -1
        
        for r_idx, row in enumerate(rows):
            if row[-1] == max_val:
                target_row_idx = r_idx
                target_col_idx = k # Last element is the max
                break
        
        if target_row_idx == -1:
            return best_rows

        row = rows[target_row_idx]
        current_max = row[target_col_idx]
        
        # Try to reduce current_max
        # We iterate downwards from current_max - 1
        # To make it efficient, we can jump or check specific candidates
        # But for a final polish, checking downwards is safe if we stop early
        
        # Optimization: We only need to check if a new value v < current_max is valid.
        # If we find one, we update and restart the search from the new max.
        
        while time.time() < end_time:
            # Find the current global max
            current_max_val = get_score(rows)
            # Find a row with this max
            target_row_idx = -1
            target_col_idx = -1
            for r_idx, row in enumerate(rows):
                if row[-1] == current_max_val:
                    target_row_idx = r_idx
                    target_col_idx = k
                    break
            
            if target_row_idx == -1:
                break
                
            row = rows[target_row_idx]
            old_val = row[target_col_idx]
            
            # Try values from old_val - 1 downwards
            # Limit the search range to avoid infinite loops if no improvement is found
            # If we check 1000 values and fail, we might be stuck.
            # But since we are reducing the max, the space is finite.
            found_improvement = False
            # We can try a random sample of lower values or sequential
            # Sequential is better for finding the *smallest* possible max eventually,
            # but might be slow. Let's try sequential but with a cap on checks per iteration
            # to allow time for other potential moves if we add them later.
            # For now, just try to find *any* valid lower value.
            
            # To speed up, we can check random values in [row[k-1]+1, old_val-1]
            # and also check old_val - 1, old_val - 2, etc.
            
            # Let's try a mix: first few sequential, then random
            checked = set()
            attempts = 0
            max_attempts = 5000 # Cap checks per max reduction attempt
            
            while attempts < max_attempts and time.time() < end_time:
                if attempts < 10:
                    # Sequential check for closest values
                    v = old_val - attempts - 1
                else:
                    # Random check in the valid range
                    low = row[target_col_idx - 1] + 1
                    high = old_val - 1
                    if low > high:
                        break
                    v = random.randint(low, high)
                
                if v in checked or v >= old_val or v <= row[target_col_idx - 1]:
                    attempts += 1
                    continue
                checked.add(v)
                
                # Check validity of replacing old_val with v in row[target_row_idx]
                # Remove old diffs involving old_val
                old_diffs_to_remove = set()
                for i in range(target_col_idx):
                    old_diffs_to_remove.add(old_val - row[i])
                # Note: old_val is the last element, so no diffs with elements after it.
                
                # Calculate new diffs involving v
                new_diffs_to_add = set()
                for i in range(target_col_idx):
                    new_diffs_to_add.add(v - row[i])
                
                # Check internal uniqueness
                if len(new_diffs_to_add) != target_col_idx:
                    attempts += 1
                    continue
                
                # Check global conflict
                # Conflict if d in new_diffs_to_add and d in (used_diffs - old_diffs_to_remove)
                conflict = False
                for d in new_diffs_to_add:
                    if d in used_diffs and d not in old_diffs_to_remove:
                        conflict = True
                        break
                
                if not conflict:
                    # Valid move! Apply it
                    row[target_col_idx] = v
                    used_diffs.difference_update(old_diffs_to_remove)
                    used_diffs.update(new_diffs_to_add)
                    row_diffs[target_row_idx] = (row_diffs[target_row_idx] - old_diffs_to_remove) | new_diffs_to_add
                    
                    new_score = get_score(rows)
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                    
                    found_improvement = True
                    break # Restart the outer loop to find the new max
                
                attempts += 1
            
            if not found_improvement:
                # No improvement found for this max value in this attempt
                # We could try other rows if there are multiple maxes, but 
                # reducing the global max requires reducing the specific element(s) holding it.
                # If we can't reduce the current global max, we are stuck in this local search phase.
                # Break to avoid wasting time.
                break
        
        return best_rows

    def simulated_annealing(rows, time_budget):
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        used_diffs = set()
        row_diffs = []
        for r in rows:
            rd = compute_row_diffs(r)
            row_diffs.append(rd)
            used_diffs.update(rd)
            
        current_score = best_score
        end_time = time.time() + time_budget
        
        # Parameters
        T = 20.0
        cooling_rate = 0.999
        min_T = 0.5
        
        # Outer loop for restarts
        while time.time() < end_time:
            # Reset T for each restart? Or keep it?
            # Restarting with a higher T helps escape local minima.
            T = max(T, 5.0)
            
            no_improve_steps = 0
            max_no_improve = 2000
            
            while time.time() < end_time and T > min_T and no_improve_steps < max_no_improve:
                r_idx = random.randint(0, n_rows - 1)
                j_idx = random.randint(1, k)
                
                row = rows[r_idx]
                old_val = row[j_idx]
                
                low_bound = row[j_idx - 1] + 1
                if j_idx < k:
                    high_bound = row[j_idx + 1] - 1
                else:
                    high_bound = old_val + 5 # Small window above for last element
                
                if low_bound > high_bound:
                    T *= cooling_rate
                    continue
                
                is_max_element = (old_val == current_score)
                
                if is_max_element:
                    if low_bound < old_val:
                        v = random.randint(low_bound, old_val - 1)
                    else:
                        T *= cooling_rate
                        continue
                else:
                    window = max(1, int(T * 3))
                    v_low = max(low_bound, old_val - window)
                    v_high = min(high_bound, old_val + window)
                    if v_low > v_high:
                        T *= cooling_rate
                        continue
                    v = random.randint(v_low, v_high)
                    
                if v == old_val:
                    T *= cooling_rate
                    continue

                old_diffs_to_remove = set()
                for i in range(j_idx):
                    old_diffs_to_remove.add(old_val - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    old_diffs_to_remove.add(row[i] - old_val)
                
                new_diffs_to_add = set()
                for i in range(j_idx):
                    new_diffs_to_add.add(v - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    new_diffs_to_add.add(row[i] - v)
                    
                if len(new_diffs_to_add) != k:
                    T *= cooling_rate
                    continue
                
                conflict = False
                for d in new_diffs_to_add:
                    if d in used_diffs:
                        if d not in old_diffs_to_remove:
                            conflict = True
                            break
                
                if conflict:
                    T *= cooling_rate
                    continue
                
                row[j_idx] = v
                used_diffs.difference_update(old_diffs_to_remove)
                used_diffs.update(new_diffs_to_add)
                row_diffs[r_idx] = (row_diffs[r_idx] - old_diffs_to_remove) | new_diffs_to_add
                
                new_score = get_score(rows)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    delta = new_score - current_score
                    # Use math.exp for accuracy and speed
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        no_improve_steps += 1
                    else:
                        row[j_idx] = old_val
                        used_diffs.difference_update(new_diffs_to_add)
                        used_diffs.update(old_diffs_to_remove)
                        row_diffs[r_idx] = (row_diffs[r_idx] - new_diffs_to_add) | old_diffs_to_remove
                        no_improve_steps += 1
                
                T *= cooling_rate
            
            # If we broke out of the inner loop due to no_improve_steps or T < min_T,
            # we restart the outer loop. 
            # To prevent infinite restarts if stuck, we can slightly increase T 
            # or just let it run. The time limit will stop it.
            # Resetting T to a moderate value helps.
            T = 10.0

        return best_rows

    # Main Logic
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Get valid solutions
    # Try a few seeds
    for seed in range(20):
        if time.time() - start_time > time_limit - 30:
            break
        rows = greedy_solution(seed)
        if rows and is_valid(rows):
            score = get_score(rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Phase 2: Simulated Annealing
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        # Reserve some time for local search
        sa_time = remaining * 0.85
        final_rows = simulated_annealing(best_rows, sa_time)
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            best_score = final_score

    # Phase 3: Local Search (Polish)
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2:
        ls_rows = local_search(best_rows, remaining - 1)
        ls_score = get_score(ls_rows)
        if ls_score < best_score:
            best_rows = [list(r) for r in ls_rows]
            best_score = ls_score

    return best_rows
