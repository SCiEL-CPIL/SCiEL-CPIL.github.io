import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 145  # 5s margin
    
    # 1. Generate initial greedy solution
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Optimization: check conflicts quickly
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used:
                        conflict = True
                        break
                    new_diffs.add(d)
                if conflict:
                    continue
                # Check internal conflicts? 
                # Since we pick increasing x, and previous elements are smaller,
                # new diffs are x - a for a in row.
                # Are these new diffs distinct from each other?
                # x - a1 and x - a2 are distinct if a1 != a2. Yes.
                # So only need to check against 'used'.
                row.append(x)
                used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_init()
    current_M = max(max(r) for r in rows)
    
    # Helper to calculate M
    def get_M(rows):
        return max(max(r) for r in rows)

    # Helper to check if a row is internally valid (distinct diffs)
    # And get its diffs
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return None # Internal conflict
                diffs.add(d)
        return diffs

    # SA Loop
    best_rows = [r[:] for r in rows]
    best_M = current_M
    
    T = 50.0
    cooling = 0.995
    
    # We will run for a fixed number of steps or time
    steps = 0
    max_steps = 2000000 # Arbitrary high, limited by time
    
    while time.time() < deadline and T > 0.5:
        steps += 1
        
        # Choose a move
        r_idx = random.randint(0, n_rows - 1)
        # Choose an element to change (indices 1 to k)
        pos = random.randint(1, k)
        old_val = rows[r_idx][pos]
        
        # Generate new value
        # Bias towards smaller values to reduce M
        # If we pick a value > current_M, M might increase.
        # Let's pick from a range [1, current_M + 10]
        # Or better, if we want to reduce M, pick [1, current_M].
        # But we need to escape local minima, so allow some larger values.
        upper_bound = current_M + 5
        if random.random() < 0.7:
            new_val = random.randint(1, current_M)
        else:
            new_val = random.randint(1, upper_bound)
            
        if new_val == old_val:
            continue
            
        # Construct new row
        new_row = rows[r_idx][:]
        new_row[pos] = new_val
        new_row.sort()
        
        # Check if 0 is still first (it is, since new_val >= 1)
        # Check internal validity
        new_diffs = get_row_diffs(new_row)
        if new_diffs is None:
            continue
            
        # Check external validity
        # Remove old diffs from used_diffs
        old_diffs = get_row_diffs(rows[r_idx])
        # Note: old_diffs is a set
        # We need to check if new_diffs intersects with (used_diffs - old_diffs)
        
        # Efficient check:
        # If any d in new_diffs is in used_diffs and d is not in old_diffs, it's a conflict.
        # Wait, if d is in old_diffs, it was "occupied" by the old row.
        # If the new row uses the same d, it's fine (it just keeps it).
        # But if the new row uses a d that was NOT in the old row, it must not be in used_diffs.
        
        conflict = False
        for d in new_diffs:
            if d in used_diffs and d not in old_diffs:
                conflict = True
                break
        
        if conflict:
            continue
            
        # Move is feasible!
        new_M = max(max(new_row), max(max(r) for i, r in enumerate(rows) if i != r_idx))
        # Optimization: pre-compute max of other rows?
        # For n=7, it's fast enough.
        
        delta_E = new_M - current_M
        
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            # Accept move
            # Update state
            # Remove old diffs, add new diffs
            used_diffs.difference_update(old_diffs)
            used_diffs.update(new_diffs)
            
            rows[r_idx] = new_row
            current_M = new_M
            
            if current_M < best_M:
                best_M = current_M
                best_rows = [r[:] for r in rows]
                
        T *= cooling
        
    return best_rows
