import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Problem parameters
    D = k * (k + 1) // 2  # Differences per row
    T = n_rows * D        # Total differences

    def get_max_entry(rows):
        return max(row[-1] for row in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(n_rows, k, randomize=False):
        """
        Constructs a valid set of rows using a greedy approach.
        If randomize is True, it shuffles the order of candidate values slightly
        to allow multi-start exploration.
        """
        used_diffs = set()
        rows = []
        
        # We want to pack differences tightly.
        # Greedy rule: For the next element, pick the smallest value such that
        # all new differences are unused.
        # To add "pressure", we can prioritize values that keep the max entry low.
        
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                # We want to find the smallest x > row[-1] such that
                # x - row[i] not in used_diffs for all i < j.
                
                # Optimization: Instead of checking from row[-1]+1 upwards,
                # we can estimate a lower bound.
                # But for simplicity and correctness in greedy, we scan.
                
                # To handle randomization, we might scan a window and pick randomly?
                # No, greedy is deterministic by nature. 
                # Multi-start usually means different initial conditions or random restarts 
                # with different random seeds affecting tie-breaking or order.
                # Here, we can randomize the order of rows construction? 
                # Rows are symmetric, so it doesn't help much unless we break symmetry.
                
                # Let's stick to standard greedy for the base solution.
                # The "multi-start" aspect will come from the local search restarts.
                
                x = row[-1] + 1
                while True:
                    conflict = False
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                    if not conflict:
                        break
                    x += 1
                
                row.append(x)
                for i in range(len(row) - 1):
                    used_diffs.add(x - row[i])
            
            rows.append(row)
        
        return rows

    # 1. Get initial solution
    best_rows = greedy_construct(n_rows, k)
    best_max = get_max_entry(best_rows)
    
    if T == best_max:
        return best_rows

    # 2. Local Search: Simulated Annealing
    # Objective: Minimize max_entry.
    # Move: Pick a random row r and a random position p (1..k).
    #       Change value at p to a new value v.
    #       Ensure row remains strictly increasing.
    #       Ensure no difference conflicts.
    #       If conflict, reject or move to a different value.
    #       Actually, a simpler move: Swap two values in the same row? No, must be increasing.
    #       Better move: Change a value to a nearby valid value.
    
    # Let's define a cost function:
    # cost = max_entry + penalty * (number_of_conflicts)
    # But we want valid solutions only for the "best" tracking.
    # For SA, we can allow invalid states to escape local minima, but that's complex.
    # Let's stick to valid moves only (hill climbing) with random restarts.
    
    # Move: 
    # 1. Pick random row r.
    # 2. Pick random index i in row r (1..k).
    # 3. Try to change row[r][i] to a new value.
    #    To keep it increasing, new_val must be > row[r][i-1] and < row[r][i+1] (if i<k).
    #    If i==k, new_val < infinity (but we want to minimize max, so maybe decrease?).
    #    Actually, to minimize max, we should try to DECREASE values.
    #    So, try new_val in [row[r][i-1]+1, row[r][i]-1].
    #    If valid (no diff conflicts), accept if it reduces max_entry or is equal.
    
    # This is a "local search" that tries to shrink the scope.
    
    def local_search(rows, time_limit):
        current_rows = [r[:] for r in rows]
        current_max = get_max_entry(current_rows)
        
        # Precompute differences for quick check? 
        # Or just check on the fly.
        
        # We will try to reduce the max entry.
        # Identify the row with the max entry.
        
        while time.time() < time_limit:
            # Find the row with the maximum entry
            max_row_idx = 0
            max_val = -1
            for i, r in enumerate(current_rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_row_idx = i
            
            # Try to reduce an element in this row
            # Pick a random element to modify (excluding 0)
            # To be effective, pick the last element first? Or random?
            # Let's pick the last element (index k) first, as it contributes most to max.
            # If we can reduce it, great.
            
            # Actually, reducing an earlier element might allow reducing later ones.
            # Let's try modifying the last element of the max row.
            
            r_idx = max_row_idx
            col_idx = k # Last element
            
            old_val = current_rows[r_idx][col_idx]
            
            # Try to find a smaller value for this element
            # Constraints:
            # 1. new_val > current_rows[r_idx][col_idx-1]
            # 2. new_val < old_val (to improve)
            # 3. All differences involving this element must be unique globally.
            
            # Differences involving (r_idx, col_idx):
            # For all i < col_idx: d = old_val - current_rows[r_idx][i]
            # These are currently in the global set.
            # If we change to new_val, the new differences are new_val - current_rows[r_idx][i].
            # We must ensure:
            # a) new_diffs are not in (global_used - old_diffs)
            # b) new_diffs are unique among themselves (they are, since row is increasing)
            
            # Let's collect old_diffs
            old_diffs = []
            for i in range(col_idx):
                d = old_val - current_rows[r_idx][i]
                old_diffs.append(d)
            
            # We need to check if new_val - current_rows[r_idx][i] is in used_diffs (excluding old_diffs)
            
            # To speed up, we can iterate new_val from old_val - 1 down to current_rows[r_idx][col_idx-1] + 1
            # and check validity.
            
            found_new = False
            for new_val in range(old_val - 1, current_rows[r_idx][col_idx-1], -1):
                # Check conflicts
                valid = True
                new_diffs = []
                for i in range(col_idx):
                    d = new_val - current_rows[r_idx][i]
                    new_diffs.append(d)
                    # Check if d is in used_diffs, but ignore if d is in old_diffs?
                    # No, we are replacing the element. The old_diffs are removed.
                    # So we check against (used_diffs - set(old_diffs))
                    # But wait, other rows might use these differences.
                    # So we check if d is in used_diffs.
                    # If d is in old_diffs, it's okay because it's being removed/replaced?
                    # No, the difference value itself is what matters.
                    # If another row has difference d, it's a conflict.
                    # If the SAME row had difference d (from an old pair), it's being removed.
                    # So we need to know if d is used by ANY OTHER pair.
                    
                    # Efficient check:
                    # Count occurrences? Too slow.
                    # Let's just assume used_diffs is a set.
                    # If d is in used_diffs, it might be from this row's old state or another row.
                    # We need to be careful.
                    
                    # Let's rebuild the check:
                    # 1. Remove old_diffs from a temporary set? No, used_diffs is global.
                    # 2. For each new_diff d:
                    #    If d is in used_diffs:
                    #       If d is also in old_diffs, it might be a false positive if no other row uses it.
                    #       But we don't have counts.
                    
                    # Better approach for local search:
                    # Maintain a dictionary of difference counts?
                    # Or, since we are only changing one element, we can check:
                    # For each new_diff d:
                    #   Is d in used_diffs?
                    #   If yes, is it ONLY from the old_diffs of this element?
                    #   This is hard without counts.
                    
                    # Let's use a simpler validity check:
                    # 1. Tentatively update the row.
                    # 2. Check global validity? O(N^2) or O(N*k^2). Too slow for SA steps.
                    
                    # Alternative:
                    # Keep a set of used_diffs.
                    # When checking new_val:
                    #   For each i < k:
                    #      d_new = new_val - row[i]
                    #      if d_new in used_diffs:
                    #          # Check if d_new is generated by the OLD value of this element
                    #          # i.e., is d_new in old_diffs?
                    #          # If yes, we need to know if it's used elsewhere.
                    #          # This is the bottleneck.
                    
                    # Let's approximate:
                    # If d_new in used_diffs and d_new not in old_diffs -> Conflict.
                    # If d_new in used_diffs and d_new in old_diffs -> Potential Conflict (need count).
                    # To be safe and fast, we can just reject if d_new in used_diffs.
                    # This might miss valid moves where the difference was only used by the old state.
                    # But it's a safe heuristic.
                    
                    if d in used_diffs:
                        # Check if it's "safe" to reuse
                        # If d is in old_diffs, it might be safe.
                        # But to keep it simple and fast, let's just reject if in used_diffs.
                        # This makes the search conservative but correct.
                        valid = False
                        break
                
                if valid:
                    # Update used_diffs
                    for d in old_diffs:
                        used_diffs.remove(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    current_rows[r_idx][col_idx] = new_val
                    current_max = min(current_max, new_val) # Actually recompute max to be safe
                    current_max = get_max_entry(current_rows)
                    found_new = True
                    break
            
            if not found_new:
                # Try other positions or other rows?
                # For now, if we can't reduce the last element of the max row,
                # we might be stuck.
                # Let's try a random move in any row to escape.
                # Pick random row r, random col c.
                r = random.randint(0, n_rows - 1)
                c = random.randint(1, k)
                
                old_val = current_rows[r][c]
                # Try to change to a nearby value
                # Range: [current_rows[r][c-1]+1, current_rows[r][c+1]-1] (if c<k)
                # Or if c==k, [current_rows[r][c-1]+1, infinity]
                
                # Let's try to decrease it first.
                lower_bound = current_rows[r][c-1] + 1
                upper_bound = old_val - 1 # Try to decrease
                
                if upper_bound >= lower_bound:
                    # Try a random value in this range
                    # Or just scan down?
                    # Scanning down might be slow if range is large.
                    # Let's try a few random candidates.
                    for _ in range(10):
                        new_val = random.randint(lower_bound, upper_bound)
                        old_diffs = [old_val - current_rows[r][i] for i in range(c)]
                        new_diffs = [new_val - current_rows[r][i] for i in range(c)]
                        
                        valid = True
                        for d in new_diffs:
                            if d in used_diffs:
                                # Check if d is in old_diffs
                                # If so, it's a "swap" within the same element's history.
                                # Still risky without counts.
                                # Let's reject.
                                valid = False
                                break
                        
                        if valid:
                            for d in old_diffs:
                                used_diffs.remove(d)
                            for d in new_diffs:
                                used_diffs.add(d)
                            current_rows[r][c] = new_val
                            break
                else:
                    # Try to increase? No, we want to minimize max.
                    # Maybe we can increase an earlier element to allow decreasing a later one?
                    # This is complex.
                    # Let's just restart if stuck.
                    pass

            # Check if we improved best
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
                if best_max == T:
                    break

        return current_rows, current_max

    # Run local search with time budget
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 1.0:
        # Run local search
        # We can run it multiple times with restarts?
        # The local search above is a single run.
        # Let's run it once.
        local_rows, local_max = local_search(best_rows, remaining_time)
        if local_max < best_max:
            best_rows = local_rows
            best_max = local_max

    return best_rows
