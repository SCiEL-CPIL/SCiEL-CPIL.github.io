import time
import random
import sys

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
        
        # Parameters
        T = 10.0
        cooling_rate = 0.9995
        min_T = 0.1
        end_time = time.time() + time_budget
        no_improve_steps = 0
        max_no_improve = 5000 # Restart logic if stuck
        
        while time.time() < end_time and T > min_T:
            # Pick a row and index to modify
            # Bias towards rows with large elements?
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k) # Index 1 to k
            
            row = rows[r_idx]
            old_val = row[j_idx]
            
            # Determine search range
            low_bound = row[j_idx - 1] + 1
            if j_idx < k:
                high_bound = row[j_idx + 1] - 1
            else:
                # If it's the last element, we can try to reduce it significantly
                # But we must stay > row[j_idx-1]
                # Let's search in [low_bound, old_val] primarily to reduce max
                # But SA allows increase. Let's allow a small window above old_val
                high_bound = old_val + 10
                
            if low_bound > high_bound:
                T *= cooling_rate
                continue
            
            # Propose new value
            # If old_val is part of the global max, bias towards lower values
            # Check if old_val is the current global max
            is_max_element = False
            # Quick check: is old_val == current_score? 
            # It's possible multiple elements have this value.
            if old_val == current_score:
                is_max_element = True
            
            if is_max_element:
                # Try to reduce
                # Pick v in [low_bound, old_val - 1]
                if low_bound < old_val:
                    v = random.randint(low_bound, old_val - 1)
                else:
                    T *= cooling_rate
                    continue
            else:
                # Pick v in a window around old_val
                # Window size proportional to T?
                window = max(1, int(T * 2))
                v_low = max(low_bound, old_val - window)
                v_high = min(high_bound, old_val + window)
                if v_low > v_high:
                    T *= cooling_rate
                    continue
                v = random.randint(v_low, v_high)
                
            if v == old_val:
                T *= cooling_rate
                continue

            # Check validity of v in row r_idx
            # Remove old diffs of old_val
            old_diffs_to_remove = set()
            # Diffs involving old_val:
            # 1. old_val - a for a in row[0..j_idx-1]
            # 2. a - old_val for a in row[j_idx+1..k]
            for i in range(j_idx):
                old_diffs_to_remove.add(old_val - row[i])
            for i in range(j_idx + 1, k_plus_1):
                old_diffs_to_remove.add(row[i] - old_val)
            
            # Calculate new diffs for v
            new_diffs_to_add = set()
            for i in range(j_idx):
                new_diffs_to_add.add(v - row[i])
            for i in range(j_idx + 1, k_plus_1):
                new_diffs_to_add.add(row[i] - v)
                
            # Check if new diffs are valid
            # 1. No internal conflict in new_diffs_to_add (handled by set creation? 
            #    Wait, if v - row[i] == v - row[m] for i!=m, that means row[i]==row[m], impossible.
            #    If v - row[i] == row[m] - v, that's a conflict within the new set?
            #    The set new_diffs_to_add contains all diffs involving v.
            #    We need to ensure len(new_diffs_to_add) == j_idx + (k - j_idx) = k.
            #    Since row is strictly increasing, v-row[i] are distinct and positive.
            #    row[m]-v are distinct and positive.
            #    Can v-row[i] == row[m]-v?
            #    v + v = row[i] + row[m]. Possible.
            #    So we must check len(new_diffs_to_add) == k.
            
            if len(new_diffs_to_add) != k:
                T *= cooling_rate
                continue
                
            # 2. Check against global used_diffs (excluding old_diffs_to_remove)
            #    We need to ensure new_diffs_to_add does not intersect 
            #    (used_diffs - old_diffs_to_remove)
            
            # Efficient check:
            # Intersect new_diffs_to_add with used_diffs
            # But we must ignore conflicts that are resolved by removing old_diffs_to_remove
            # i.e., if d in new_diffs_to_add and d in used_diffs:
            #    it's OK if d is in old_diffs_to_remove (because we are removing it)
            #    OR if d is in new_diffs_to_add itself? No, we already checked internal uniqueness.
            #    Wait, if d is in new_diffs_to_add, and d is in used_diffs, 
            #    and d is NOT in old_diffs_to_remove, then conflict.
            
            conflict = False
            for d in new_diffs_to_add:
                if d in used_diffs:
                    if d not in old_diffs_to_remove:
                        conflict = True
                        break
            
            if conflict:
                T *= cooling_rate
                continue
            
            # Valid move. Calculate new score.
            # We don't need to update the whole rows structure yet to calculate score
            # The new max of row r_idx is max(row[0..j-1], v, row[j+1..])
            # The global max is max(current_score, new_row_max) 
            # Wait, if old_val was the unique global max, and v < old_val, 
            # the new global max might be lower.
            
            # Let's just update the row and compute score properly if accepted?
            # Or compute new_score efficiently.
            
            # For now, let's just compute max of the modified row
            # row_max_new = max(row[:j_idx] + [v] + row[j_idx+1:])
            # But we need global max.
            # We can maintain a count of how many times current_score appears?
            # Or just compute get_score(rows) if we tentatively apply?
            # get_score is O(n*k) = O(32). Very fast.
            
            # Tentatively apply
            row[j_idx] = v
            
            # Update used_diffs
            used_diffs.difference_update(old_diffs_to_remove)
            used_diffs.update(new_diffs_to_add)
            
            # Update row_diffs for this row
            row_diffs[r_idx] = new_diffs_to_add | (row_diffs[r_idx] - old_diffs_to_remove)
            # Note: row_diffs[r_idx] should be the set of ALL diffs in the row.
            # old_diffs_to_remove are the ones involving old_val.
            # The other diffs in the row are unchanged.
            # So new_row_diffs = (old_row_diffs - old_diffs_to_remove) | new_diffs_to_add
            
            new_score = get_score(rows)
            
            # Metropolis
            if new_score < current_score:
                # Accept
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_rows = [list(r) for r in rows]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Probability of accepting worse move
                delta = new_score - current_score
                prob = 2.718281828 ** (-delta / T) # exp(-delta/T)
                if random.random() < prob:
                    current_score = new_score
                    no_improve_steps += 1
                else:
                    # Reject: Undo changes
                    row[j_idx] = old_val
                    used_diffs.difference_update(new_diffs_to_add)
                    used_diffs.update(old_diffs_to_remove)
                    row_diffs[r_idx] = old_diffs_to_remove | (row_diffs[r_idx] - new_diffs_to_add)
                    no_improve_steps += 1
            
            T *= cooling_rate
            
            # Restart logic
            if no_improve_steps > max_no_improve:
                # Reset to best and increase T slightly?
                # Or just break and let outer loop restart
                break
                
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
        # Run SA for a portion of remaining time
        # Maybe split into a few runs with different T?
        sa_time = remaining * 0.8
        final_rows = simulated_annealing(best_rows, sa_time)
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            best_score = final_score

    return best_rows
