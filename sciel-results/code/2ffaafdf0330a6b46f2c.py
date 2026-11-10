import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110  # 10s buffer
    
    # 1. Generate initial valid solution using Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        last = x
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)
    
    # Pre-compute diffs for best_rows to maintain state
    def compute_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs

    # We will run Simulated Annealing
    # State: rows
    # Cost: max element
    
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    
    # Helper to check validity of a specific row change
    # This is expensive if done naively. 
    # Optimization: Keep a global set of used diffs.
    # When we change a row, we remove its old diffs, check new diffs against global set.
    
    global_used = compute_diffs(current_rows)
    
    # To speed up, we can store diffs per row
    row_diffs = []
    for r in current_rows:
        rd = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                rd.add(r[j] - r[i])
        row_diffs.append(rd)
        
    # SA Loop
    T = 1000.0
    T_min = 0.1
    alpha = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random row and index
        r_idx = random.randint(0, n_rows - 1)
        # Index 0 is fixed at 0, so pick from 1 to k
        j_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][j_idx]
        
        # Generate a candidate new value
        # To allow exploration, we can pick a value in a range around old_val
        # Or a random value in [1, current_max + 10]
        # Let's try to pick a value that might reduce the max, so bias towards lower values
        # But we must maintain strict increase.
        
        min_val = current_rows[r_idx][j_idx - 1] + 1
        max_val = current_rows[r_idx][j_idx + 1] - 1 if j_idx < k else current_max + 10
        
        if min_val > max_val:
            T *= alpha
            continue
            
        new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            T *= alpha
            continue
            
        # Check validity
        # Remove old diffs for row r_idx from global_used
        # Note: We must be careful with set operations.
        # Since diffs are unique across ALL rows, if we remove row r_idx's diffs, 
        # the remaining global_used contains diffs from all OTHER rows.
        
        # We need to check if new_diffs (for the modified row) intersect with 
        # (global_used - old_diffs).
        
        # Let's compute new row
        new_row = current_rows[r_idx][:]
        new_row[j_idx] = new_val
        
        new_row_diffs = set()
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                new_row_diffs.add(new_row[j] - new_row[i])
                
        # Check intersection with other rows' diffs
        # Other rows' diffs = global_used - row_diffs[r_idx]
        # But global_used is the union. 
        # Let's just check if any diff in new_row_diffs is in global_used AND not in old_row_diffs.
        # Actually, if a diff is in new_row_diffs and in global_used, it might be from the same row (old) or another row.
        # If it's from the same row (old), it's fine if it's still there? No, we replaced the row.
        # The condition is: All diffs in ALL rows must be distinct.
        # So, new_row_diffs must be disjoint from (global_used - old_row_diffs).
        
        other_diffs = global_used - row_diffs[r_idx]
        
        if not new_row_diffs & other_diffs:
            # Valid move
            # Calculate new cost
            new_max = max(new_row)
            # The global max is max(new_max, max of other rows)
            # We can precompute max of other rows? Or just compute max of all rows.
            # For speed, let's just compute max of current_rows with the change.
            # Actually, current_max is the global max.
            # If old_val was the unique global max, and new_val < old_val, we might reduce it.
            # If new_val > old_val, we increase it.
            
            # Let's just compute the new global max.
            # Since n_rows and k are small, max() is fast.
            new_global_max = max(max(r) for r in current_rows) # Wait, current_rows hasn't been updated yet
            
            # We need to update current_rows to calculate this correctly, or do it in place.
            # Let's simulate the update.
            
            # Check if this move improves or is neutral/worse
            # We need the actual new global max.
            # Let's just update a copy or calculate it carefully.
            
            # Efficient max calculation:
            # If new_val > old_val:
            #   new_global_max = max(new_val, current_max)
            # Else:
            #   If old_val == current_max, we need to check if there are other elements == current_max.
            #   This is getting complex. Let's just compute it.
            
            # Update state temporarily
            current_rows[r_idx] = new_row
            new_global_max = max(max(r) for r in current_rows)
            
            delta_cost = new_global_max - current_max
            
            if delta_cost <= 0:
                # Improvement or neutral
                current_rows[r_idx] = new_row
                current_max = new_global_max
                row_diffs[r_idx] = new_row_diffs
                # Update global_used
                # Remove old diffs, add new diffs
                # Since we know they are disjoint from others, we can just update the set
                # global_used = (global_used - row_diffs[r_idx]) | new_row_diffs
                # But we already checked disjointness.
                global_used = (global_used - row_diffs[r_idx]) | new_row_diffs
                
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
            else:
                # Worse, accept with probability
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current_rows[r_idx] = new_row
                    current_max = new_global_max
                    row_diffs[r_idx] = new_row_diffs
                    global_used = (global_used - row_diffs[r_idx]) | new_row_diffs
                else:
                    # Revert
                    current_rows[r_idx][j_idx] = old_val
                    # row_diffs and global_used remain unchanged
                    
        else:
            # Invalid
            pass
            
        T *= alpha

    return best_rows
