import random
import math
import time

def construct(n_rows, k):
    # n_rows is 4, k is 8 in this task, but we keep it general.
    # Each row has k+1 elements.
    row_len = k + 1
    
    def greedy_init():
        """
        Generate a valid initial configuration using a greedy approach.
        We fill rows one by one.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We want to keep numbers small, so we try small increments
            # But we must ensure uniqueness of differences.
            # A simple strategy: try to add the smallest possible next element.
            while len(row) < row_len:
                found = False
                # Start from last element + 1
                # To make it random/greedy, we can try a range
                # But for initialization, deterministic greedy is fine to get a valid start.
                # We'll try values from last+1 up to some limit.
                # Since we don't know the limit, we just keep going until valid.
                # To avoid infinite loops in bad cases, we can use a large bound.
                # But for k=8, n=4, the bound is small (~100-200).
                
                # Optimization: Instead of linear search from last+1, 
                # we can just find the next valid number.
                # Let's try from 1 upwards, skipping those in row.
                # Actually, simpler:
                # Try x = row[-1] + 1, row[-1] + 2, ...
                # If collision, try larger.
                
                # A better greedy: 
                # Try to place the next point as close as possible to the previous one?
                # No, that causes small differences which might collide.
                # Standard Golomb ruler construction:
                
                # Let's just do a simple search for the next valid point.
                # We maintain the set of used differences.
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        # Check internal uniqueness of new_diffs?
                        # Since x > a for all a in row, x-a are distinct if a are distinct.
                        # So new_diffs is a set of distinct values.
                        # We just need to ensure they don't collide with 'used'.
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    x += 1
            rows.append(row)
        return rows, used

    # Initial construction
    rows, used = greedy_init()
    best_rows = [r[:] for r in rows]
    best_cost = max(max(r) for r in rows)
    
    # Simulated Annealing
    start_time = time.time()
    max_time = 110  # 10s buffer
    
    def get_row_diffs(row):
        """Return set of all differences in the row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def remove_point_diffs(row, idx):
        """Return set of differences involving row[idx]."""
        val = row[idx]
        diffs = set()
        for j, v in enumerate(row):
            if j != idx:
                diffs.add(abs(val - v))
        return diffs

    def add_point_diffs(row, idx):
        """Return set of differences involving row[idx]."""
        val = row[idx]
        diffs = set()
        for j, v in enumerate(row):
            if j != idx:
                diffs.add(abs(val - v))
        return diffs

    # State
    current_rows = [r[:] for r in rows]
    current_used = set(used)
    current_cost = best_cost
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    # We will store the best solution found
    # Note: We must ensure 'used' is consistent with 'current_rows'.
    # We update 'current_used' explicitly during moves.
    
    while time.time() - start_time < max_time and T > T_min:
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick a random index to move (1 to k)
        # Index 0 is always 0
        p_idx = random.randint(1, row_len - 1)
        
        # Determine valid range for new value
        # Must be > row[p_idx-1] and < row[p_idx+1] (if exists)
        low = row[p_idx - 1] + 1
        high = row[p_idx + 1] - 1 if p_idx < row_len - 1 else current_cost + 10 # Allow expansion if needed? No, we want to minimize.
        # Actually, if we want to minimize max, we shouldn't expand unless necessary.
        # But to escape local minima, we might need to move up.
        # Let's allow moving up to current_max + 5 or so.
        # However, if we move up, cost increases. SA allows this.
        
        # To be efficient, let's pick a random value in a window around the current value.
        # Or a random value in [low, high].
        # If high < low, no valid move in range.
        
        if high < low:
            T *= cooling
            continue
            
        new_val = random.randint(low, high)
        
        if new_val == row[p_idx]:
            T *= cooling
            continue
            
        # Check validity of moving row[p_idx] to new_val
        # 1. Remove old diffs of this point from current_used
        old_diffs = remove_point_diffs(row, p_idx)
        
        # Temporarily remove from global used
        # We don't want to modify current_used yet until we confirm
        # Check if new_diffs intersect with (current_used - old_diffs)
        
        new_diffs = set()
        for j, v in enumerate(row):
            if j != p_idx:
                new_diffs.add(abs(new_val - v))
                
        # Check for internal duplicates in new_diffs?
        # abs(new_val - v) for distinct v are distinct unless new_val is midpoint.
        # If new_val is midpoint of v1, v2, then |new_val-v1| == |new_val-v2|.
        # This would create a duplicate difference within the row.
        # So we must check if len(new_diffs) == len(row) - 1.
        if len(new_diffs) != row_len - 1:
            T *= cooling
            continue
            
        # Check against global used
        # Conflict if any d in new_diffs is in current_used AND d is not in old_diffs
        # i.e., new_diffs & (current_used - old_diffs)
        
        # Optimization: 
        # We can just check intersection.
        # If intersection is not empty, invalid.
        
        # Note: old_diffs are currently in current_used.
        # So if d is in new_diffs and d is in old_diffs, it's fine (it's "reused" by this point).
        # If d is in new_diffs and d is in current_used but NOT in old_diffs, it's a conflict.
        
        # So: conflict = new_diffs & (current_used - old_diffs)
        
        # Calculating set difference every time is O(N).
        # Alternative: 
        # 1. Remove old_diffs from a temp copy? No, too slow.
        # 2. Check: for d in new_diffs: if d in current_used and d not in old_diffs: return invalid.
        
        valid = True
        for d in new_diffs:
            if d in current_used and d not in old_diffs:
                valid = False
                break
                
        if not valid:
            T *= cooling
            continue
            
        # Move is valid. Calculate new cost.
        # The max cost is max of all rows.
        # Only this row changed.
        # Old max of this row: max(row)
        # New max of this row: max(row with new_val)
        
        # We can just compute the global max after the move.
        # Since k is small, this is fast.
        
        # Perform move
        old_val = row[p_idx]
        row[p_idx] = new_val
        
        # Update current_used
        current_used -= old_diffs
        current_used |= new_diffs
        
        # Calculate new cost
        new_cost = max(max(r) for r in current_rows)
        
        # Acceptance
        delta_cost = new_cost - current_cost
        if delta_cost <= 0:
            # Better or equal
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [r[:] for r in current_rows]
        else:
            # Worse
            if random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
            else:
                # Reject
                row[p_idx] = old_val
                current_used -= new_diffs
                current_used |= old_diffs
        
        T *= cooling
        
    return best_rows
