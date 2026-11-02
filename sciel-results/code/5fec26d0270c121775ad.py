import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Parameters
    row_len = k + 1
    
    # Helper to calculate differences of a row
    def get_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]

    # Helper to check validity and get scope
    def evaluate(rows):
        diffs = []
        for r in rows:
            diffs.extend(get_diffs(r))
        counts = defaultdict(int)
        for d in diffs:
            counts[d] += 1
        collisions = sum(1 for d, c in counts.items() if c > 1)
        scope = max(max(r) for r in rows)
        return scope, collisions

    # Initial Greedy Construction
    # This ensures we have a valid starting point.
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_score, _ = evaluate(best_rows)
    
    # Local Search / Simulated Annealing
    # State: list of rows
    # Move: Pick a row, pick an element index (1..k), shift it by delta, 
    #        ensuring the row remains strictly increasing and non-negative.
    #        If the move creates collisions, it might still be accepted if it reduces scope significantly 
    #        or if we are in a "cooling" phase allowing invalid states to escape local minima.
    
    # A better move set for DTS:
    # 1. Shift an element in a row.
    # 2. Swap two elements in a row (if they don't break ordering).
    # 3. "Rotate" differences: hard to implement directly.
    
    # Let's use a simple hill climbing with restarts.
    # Objective: Minimize (Scope * 1000 + Collisions)
    
    def objective(rows):
        scope, coll = evaluate(rows)
        return scope * 1000 + coll

    current_rows = [r[:] for r in best_rows]
    current_obj = objective(current_rows)
    
    temp = 50.0
    cool_rate = 0.999
    
    # Pre-compute differences for current state to speed up evaluation?
    # With n_rows=4, k=7, total diffs = 4 * 28 = 112.
    # Evaluating is O(112). Very fast.
    
    while time.time() < deadline:
        # Generate a move
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick an index to shift (1 to k, index 0 is always 0)
        # We can also try to shift the whole row? No, row starts at 0.
        # We can try to change the value of row[i]
        
        # Strategy: Try to reduce the scope.
        # If the last element is the max, try to decrease it.
        # Or try to decrease any element that is contributing to the max.
        
        # Random perturbation
        # Pick a random index i > 0
        i_idx = random.randint(1, row_len - 1)
        old_val = row[i_idx]
        
        # Determine valid range for new_val
        # row[i_idx-1] < new_val < row[i_idx+1] (if i_idx < row_len-1)
        low = row[i_idx-1] + 1
        high = row[row_len-1] - 1 if i_idx < row_len-1 else old_val + 10 # Allow expansion
        
        # If we want to reduce scope, bias towards smaller values
        # If old_val is the max, bias towards smaller
        if old_val == max(row):
             high = old_val - 1
             if low > high:
                 continue
             # Try to decrease
             delta = random.randint(-10, 0)
             new_val = old_val + delta
             if new_val < low: new_val = low
             if new_val > high: new_val = high
        else:
             # Random walk
             delta = random.randint(-5, 5)
             new_val = old_val + delta
             if new_val < low: new_val = low
             if new_val > high: new_val = high
             
        if new_val == old_val:
            continue
            
        # Apply move
        row[i_idx] = new_val
        new_obj = objective(current_rows)
        
        # Acceptance criterion (Simulated Annealing)
        if new_obj < current_obj:
            current_obj = new_obj
            # Update best
            if current_obj < best_score * 1000: # Rough check, better to use actual scope
                scope, coll = evaluate(current_rows)
                if coll == 0 and scope < best_score:
                    best_score = scope
                    best_rows = [r[:] for r in current_rows]
        else:
            # Reject if worse, unless we want to allow invalid states temporarily?
            # For DTS, invalid states (collisions) make it hard to know the true scope.
            # Let's stick to valid-only search but with random restarts if stuck.
            # If we allow invalid states, we might escape local minima.
            # Let's try accepting worse states with probability exp(-deltaE/T)
            delta_e = new_obj - current_obj
            if delta_e > 0:
                prob = pow(2.718, -delta_e / temp)
                if random.random() > prob:
                    # Revert
                    row[i_idx] = old_val
            # If accepted, keep the move (even if worse)
        
        # Cool down
        temp *= cool_rate
        if temp < 0.1:
            # Restart with a new random perturbation of best_rows
            if random.random() < 0.05: # Small chance to restart even if temp is low
                current_rows = [r[:] for r in best_rows]
                # Perturb
                for _ in range(10):
                    r_idx = random.randint(0, n_rows - 1)
                    i_idx = random.randint(1, row_len - 1)
                    row = current_rows[r_idx]
                    old_val = row[i_idx]
                    low = row[i_idx-1] + 1
                    high = row[row_len-1] - 1 if i_idx < row_len-1 else old_val + 10
                    new_val = random.randint(low, high)
                    row[i_idx] = new_val
                current_obj = objective(current_rows)
                temp = 50.0

    # Final check: ensure best_rows is valid
    # The greedy start is valid. The search only accepts valid improvements or 
    # temporarily accepts invalid states but reverts if rejected.
    # However, if we accepted an invalid state, current_rows might be invalid.
    # We only update best_rows if coll == 0.
    # So best_rows should always be valid.
    
    return best_rows
# EVOLVE-BLOCK-END
