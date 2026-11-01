import time
import random
import math

def construct(n_rows, k):
    """
    Finds a difference triangle set of n_rows rows, each with k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Helper functions
    
    def greedy_init():
        """
        Greedy initialization: each row takes the smallest next mark with no difference collision.
        """
        rows = []
        used = set()
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def calc_row_diffs(row):
        """Compute the set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def global_max(rows):
        return max(max(row) for row in rows)

    # 1. Generate a good initial solution
    best_rows = None
    best_val = float('inf')
    
    # Try a few greedy variants with different random seeds? 
    # Greedy is deterministic if we always pick smallest. 
    # To diversify, we can try "greedy with noise" or just use one good greedy.
    # Let's try greedy, and also a simple random initialization pruned by greedy?
    # Actually, greedy is quite good. Let's just use it.
    
    init_rows = greedy_init()
    best_rows = [r[:] for r in init_rows]
    best_val = global_max(best_rows)
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    
    # Precompute row diffs and all diffs
    row_diffs = [calc_row_diffs(r) for r in current_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    # Simulated Annealing
    T = 500.0
    T_min = 0.5
    cooling = 0.99995
    no_improve = 0
    max_no_improve = 50000
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            # Restart from best
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            row_diffs = [calc_row_diffs(r) for r in current_rows]
            all_diffs = set()
            for rd in row_diffs:
                all_diffs |= rd
            T = 300.0
            no_improve = 0
            # Perturb
            for _ in range(5):
                ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
                if low <= high:
                    new_val = random.randint(low, high)
                    current_rows[ri][pi] = new_val
            # Recalculate state after perturbation (might be invalid, SA will fix)
            # Actually, if perturbation makes it invalid, we need to handle it.
            # Simplest: just reset to best and let SA run. Perturbation is risky if it breaks validity.
            # Let's stick to valid moves only in SA.
            # So restart just means resetting to best and continuing.
            continue
        
        # Pick a random row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Determine valid range for new value
        min_allowed = current_rows[ri][pi - 1] + 1
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
        
        # Generate candidate values
        # We prefer moves that reduce the max, so bias towards smaller values
        # but allow exploration.
        candidates = []
        
        # Try values close to current
        for d in range(1, 20):
            if old_val - d >= min_allowed:
                candidates.append(old_val - d)
            if old_val + d <= max_allowed:
                candidates.append(old_val + d)
        
        # Random jump within range
        if random.random() < 0.5:
            if min_allowed <= max_allowed:
                candidates.append(random.randint(min_allowed, max_allowed))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cooling)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            # Create candidate row
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Calculate new diffs for this row
            # Optimization: We can update diffs incrementally, but for simplicity and safety,
            # let's compute the set. k is small (e.g., 6), so O(k^2) is fast.
            new_row_diffs = set()
            valid_internal = True
            n = len(cand_row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in new_row_diffs:
                        valid_internal = False
                        break
                    new_row_diffs.add(d)
                if not valid_internal:
                    break
            
            if not valid_internal:
                continue
                
            # Check cross-row conflicts
            # The new diffs must not conflict with diffs from OTHER rows.
            # all_diffs contains diffs from all rows.
            # old_row_diffs is the set of diffs from the current row BEFORE the move.
            # The "other" diffs are all_diffs - old_row_diffs.
            # We need new_row_diffs to be disjoint from (all_diffs - old_row_diffs).
            # Equivalently: (new_row_diffs & all_diffs) must be a subset of old_row_diffs.
            # Or simpler: (new_row_diffs & (all_diffs - old_row_diffs)) must be empty.
            
            # To avoid creating a huge temporary set (all_diffs - old_row_diffs),
            # we can iterate new_row_diffs and check if d is in all_diffs AND not in old_row_diffs.
            
            conflict = False
            for d in new_row_diffs:
                if d in all_diffs and d not in row_diffs[ri]:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Move is valid. Calculate cost change.
            new_row_max = max(cand_row)
            old_row_max = max(current_rows[ri])
            
            # Update global max
            # If this row was the unique max and we reduced it, global max might drop.
            # If we increased it, global max might rise.
            
            # Efficient global max update:
            # Keep a count of how many rows have max == current_val?
            # Or just recalculate? n_rows is small.
            # Let's just recalculate for simplicity, it's O(n_rows * k).
            
            # Actually, let's maintain row_maxes list for speed.
            # But for now, let's just compute delta based on potential global max change.
            
            # Let's compute the new global max properly.
            # We know current_val is the max of current_rows.
            # If new_row_max < old_row_max:
            #   The global max might decrease if old_row_max was the global max and no other row has it.
            #   Or it stays the same.
            # If new_row_max > old_row_max:
            #   The global max becomes max(current_val, new_row_max).
            
            # Let's just compute the new global max from the modified list.
            # To avoid copying the whole list, we can do:
            # If pi is the last element, and it's the max, then max(cand_row) is easy.
            # But pi is random.
            
            # For n_rows <= 20, computing max of max(row) is fast.
            # Let's assume we can afford O(n_rows) check.
            
            # Actually, let's just do the swap and calculate.
            # We need to be careful not to modify current_rows until we decide to accept.
            
            # Calculate new global max
            # We can do:
            # new_global_max = max([max(r) for i, r in enumerate(current_rows) if i != ri] + [new_row_max])
            # This is O(n_rows * k).
            
            # Optimization: Precompute row_maxes.
            # Let's maintain row_maxes.
            
            # I will add row_maxes maintenance below.
            
            # For now, let's just use a helper.
            # Since we are in a tight loop, let's optimize.
            
            # Let's assume row_maxes is a list of max(row) for each row.
            # We need to initialize it.
            
            # I'll refactor the main loop to use row_maxes.
            
            # Break out of candidate loop to accept/reject
            # We need to know the new global max.
            
            # Let's restructure:
            # We will calculate new_global_max.
            
            # To do this efficiently, we need row_maxes.
            # Let's add that to the state.
            
            # Since I'm writing the code in one go, I'll define row_maxes outside.
            
            # Let's restart the SA loop logic with row_maxes.
            pass # Placeholder, will implement below

    # --- Implementation with Row Maxes ---
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    row_diffs = [calc_row_diffs(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    T = 500.0
    T_min = 0.5
    cooling = 0.99995
    no_improve = 0
    max_no_improve = 50000
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            # Restart from best
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            row_diffs = [calc_row_diffs(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            all_diffs = set()
            for rd in row_diffs:
                all_diffs |= rd
            T = 300.0
            no_improve = 0
            continue
        
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
        
        candidates = []
        # Bias towards decreasing values if possible to lower the max
        if old_val > min_allowed:
            # Try decreasing
            for d in range(1, min(20, old_val - min_allowed + 1)):
                candidates.append(old_val - d)
        # Try increasing for exploration
        if old_val < max_allowed:
            for d in range(1, min(20, max_allowed - old_val + 1)):
                candidates.append(old_val + d)
        
        if random.random() < 0.3 and min_allowed <= max_allowed:
            candidates.append(random.randint(min_allowed, max_allowed))
            
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cooling)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Check internal validity
            new_row_diffs = set()
            valid_internal = True
            n = len(cand_row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in new_row_diffs:
                        valid_internal = False
                        break
                    new_row_diffs.add(d)
                if not valid_internal:
                    break
            
            if not valid_internal:
                continue
                
            # Check external validity
            conflict = False
            for d in new_row_diffs:
                if d in all_diffs and d not in row_diffs[ri]:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Calculate new global max
            new_row_max = max(cand_row)
            
            # Update row_maxes temporarily to find new global max
            # old_row_max = row_maxes[ri]
            # If new_row_max < old_row_max:
            #    The global max might decrease.
            #    New global max = max(new_row_max, max(row_maxes[i] for i != ri))
            # If new_row_max >= old_row_max:
            #    New global max = max(current_val, new_row_max)
            
            # Efficient check:
            # If new_row_max > current_val:
            #    new_global_max = new_row_max
            # Else:
            #    # new_row_max <= current_val
            #    # If old_row_max == current_val, we might have reduced the global max
            #    # if no other row has current_val.
            #    # Otherwise global max stays current_val.
            
            if new_row_max > current_val:
                new_global_max = new_row_max
            else:
                # Check if current_val is still present in other rows
                # Or if this row was the only one with current_val and we reduced it.
                # Simplest: just compute max of all row_maxes with this one updated.
                # Since n_rows is small, this is fine.
                # But we can be smarter.
                # If old_row_max < current_val:
                #    Global max is still current_val (from another row)
                #    new_global_max = current_val
                # Else (old_row_max == current_val):
                #    Global max is max(new_row_max, max(other_rows))
                #    We need to know max(other_rows).
                
                # Let's just do the O(n) scan for correctness and simplicity, it's fast enough.
                # Or even better, maintain a count of maxes?
                # Let's stick to O(n) scan for now.
                
                # To avoid O(n) scan in the inner loop, we can precompute
                # second_max and count_of_max.
                
                # For now, let's just do:
                # new_global_max = max([row_maxes[i] for i in range(n_rows) if i != ri] + [new_row_max])
                
                # This is O(n_rows).
                
                # Let's optimize this by maintaining `max_val` and `count_max`.
                # I'll implement that logic.
                
                # Actually, let's just use the O(n) scan. n_rows is likely < 20.
                # 20 operations is negligible compared to the diff checks.
                
                new_global_max = new_row_max
                for i in range(n_rows):
                    if i != ri:
                        if row_maxes[i] > new_global_max:
                            new_global_max = row_maxes[i]
            
            delta = new_global_max - current_val
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.1:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Commit the move
                # Update all_diffs
                all_diffs -= row_diffs[ri]
                all_diffs |= new_row_diffs
                
                current_rows[ri] = cand_row
                row_diffs[ri] = new_row_diffs
                row_maxes[ri] = new_row_max
                current_val = new_global_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cooling)

    return best_rows

# Run a quick self-test? No, just return the function.
# The function construct is defined above.

if __name__ == "__main__":
    # Example usage
    rows = construct(7, 5)
    print(rows)
