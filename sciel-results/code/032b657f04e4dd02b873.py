import random
import math
import time

def construct(n_rows, k):
    start_time = time.time()
    timeout = 115.0

    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
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

    def is_valid_move(rows, i, j, new_val):
        """
        Check if setting rows[i][j] to new_val results in a valid configuration.
        """
        row = rows[i]
        # Check bounds
        if j > 0 and new_val <= row[j - 1]:
            return False
        if j < len(row) - 1 and new_val >= row[j + 1]:
            return False
        
        # Temporarily update
        old_val = row[j]
        row[j] = new_val
        
        # Calculate new diffs for this row
        new_row_diffs = get_row_diffs(row)
        
        # Check internal distinctness
        if len(new_row_diffs) != len(row) * (len(row) - 1) // 2:
            row[j] = old_val
            return False
            
        # Check against other rows
        for r_idx in range(n_rows):
            if r_idx == i:
                continue
            other_diffs = get_row_diffs(rows[r_idx])
            if new_row_diffs & other_diffs:
                row[j] = old_val
                return False
                
        row[j] = old_val
        return True

    # Initial solution
    rows = greedy_construct()
    current_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = current_max
    
    T = 10.0
    cooling_rate = 0.995
    
    # Precompute diffs for others to speed up? 
    # Since n_rows is small (7) and k is small (6), get_row_diffs is fast.
    # We can just recompute.
    
    while time.time() - start_time < timeout:
        if T < 0.01:
            T = 10.0 # Restart temperature
            
        i = random.randint(0, n_rows - 1)
        # Pick a random index to modify. Index 0 is fixed at 0.
        # We can modify indices 1 to k.
        j = random.randint(1, k)
        
        row = rows[i]
        old_val = row[j]
        
        # Determine range for new_val
        # Must be > row[j-1] and < row[j+1] (if exists)
        low = row[j - 1] + 1
        high = row[j + 1] - 1 if j < k else current_max + 10 # Allow some growth? 
        # Actually, to minimize max, we should prefer smaller values.
        # But if we are stuck, we might need to grow? No, we want to minimize max.
        # If we only move within (low, high), the max of the row might decrease if j==k and we move down.
        # If j < k, moving down might allow later elements to move down? No, we only move one element.
        # If we move an element down, the row's max might stay same or decrease (if it was the max).
        # If we move an element up, the row's max might increase.
        
        # Strategy: Try to find a valid new_val that reduces the row's max or keeps it same.
        # If j == k, moving down reduces max.
        # If j < k, moving down doesn't change max (unless it was max, which it isn't).
        # So, prefer moving down if possible.
        
        # Let's sample a few candidates.
        # Candidate 1: Try to move down by 1, 2, ...
        # Candidate 2: Random in (low, high)
        
        candidates = []
        # Try to shrink the row if possible (move last element down)
        if j == k:
            # Try values from old_val - 1 down to low
            # But we need to be efficient.
            # Let's try a few random steps down and some random in range.
            for _ in range(5):
                # Try to move down
                if old_val > low:
                    # Try a small decrement
                    delta = random.randint(1, min(5, old_val - low))
                    new_val = old_val - delta
                    if new_val > low:
                        candidates.append(new_val)
                        
        # Try random in range
        if high > low:
            for _ in range(5):
                new_val = random.randint(low, high)
                if new_val != old_val:
                    candidates.append(new_val)
                    
        # If no candidates, try expanding? 
        # If we are stuck, maybe we need to increase max to allow rearrangement?
        # But the goal is to minimize max. 
        # If we can't move down, maybe we can move up to allow a better global config later?
        # Simulated annealing allows bad moves.
        
        if not candidates:
            # If range is tight, try expanding high if j == k
            if j == k:
                candidates.append(old_val + random.randint(1, 5))
            else:
                continue

        random.shuffle(candidates)
        
        accepted = False
        for new_val in candidates:
            if is_valid_move(rows, i, j, new_val):
                # Calculate new global max
                # Only row i changed.
                new_row_max = max(rows[i]) # This is O(k)
                # We need to know the max of other rows to compute global max efficiently?
                # Or just compute max of all rows. O(n*k) is small.
                new_global_max = 0
                for r in rows:
                    if r[-1] > new_global_max:
                        new_global_max = r[-1]
                
                delta = new_global_max - current_max
                
                accept = False
                if delta < 0:
                    accept = True
                else:
                    if T > 0:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Commit the move
                    rows[i][j] = new_val
                    current_max = new_global_max
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in rows]
                    accepted = True
                    break
        
        if not accepted:
            # Cooldown
            T *= cooling_rate

    return best_rows
