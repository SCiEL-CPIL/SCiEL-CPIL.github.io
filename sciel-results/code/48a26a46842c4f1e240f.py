import random
import math
import copy
import time

def construct(n_rows, k):
    # Target: n_rows rows, each with k+1 elements starting at 0.
    # Minimize max element.
    
    # 1. Initialization: Greedy construction
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

    current_rows = greedy_construct()
    
    # Precompute differences for current state
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def compute_all_diffs(rows):
        all_diffs = set()
        for r in rows:
            all_diffs.update(get_diffs(r))
        return all_diffs

    current_diffs = compute_all_diffs(current_rows)
    current_max = max(max(r) for r in current_rows)
    
    best_rows = copy.deepcopy(current_rows)
    best_max = current_max
    best_diffs = copy.deepcopy(current_diffs)
    
    # SA Parameters
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    T = 100.0
    cooling = 0.9995
    min_T = 0.01
    
    random.seed(42)
    
    while T > min_T and (time.time() - start_time) < timeout:
        # Pick a random row and element to move
        ri = random.randint(0, n_rows - 1)
        ei = random.randint(1, k) # Index 0 is fixed at 0
        
        row = current_rows[ri]
        old_val = row[ei]
        
        # Determine valid range for new value
        # Must be strictly between neighbors
        lower = row[ei-1] + 1
        upper = row[ei+1] - 1 if ei < k else current_max + 10 # Allow expanding slightly? No, we want to minimize max.
        # Actually, we should allow moving to any valid position, even if it increases max,
        # but we want to find a configuration with a smaller max eventually.
        # If we restrict to upper < current_max, we might get stuck.
        # Let's allow upper to be current_max + 50 to allow "kicks" out of local minima.
        upper = max(upper, current_max + 50)
        
        if lower > upper:
            T *= cooling
            continue
            
        # Pick a random new value in the range
        # To make it efficient, we don't iterate all, just pick one.
        # But picking a random value in a large range might be inefficient if most are invalid.
        # Better: Pick a few random candidates.
        accepted = False
        for _ in range(5): # Try 5 random candidates per step
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
                
            # Check validity
            # 1. Remove old diffs involving ei
            old_diffs = set()
            for j in range(len(row)):
                if j != ei:
                    d = abs(old_val - row[j])
                    old_diffs.add(d)
            
            # 2. Generate new diffs involving ei
            new_diffs = set()
            valid = True
            for j in range(len(row)):
                if j != ei:
                    d = abs(new_val - row[j])
                    new_diffs.add(d)
            
            # 3. Check for conflicts
            # The new diffs must not be in (current_diffs - old_diffs)
            # Also, new diffs must be distinct among themselves (they are, as they are from one row with unique elements? 
            # Wait, if new_val - row[j1] == new_val - row[j2], then row[j1] == row[j2], which is impossible.
            # So new_diffs are distinct internally.)
            
            # Check against global used diffs
            # available_diffs = current_diffs - old_diffs
            # if new_diffs & available_diffs: invalid
            
            # Optimization: Instead of creating a new set, iterate
            for d in new_diffs:
                if d in current_diffs and d not in old_diffs:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Valid move!
            # Calculate new max
            # We only changed one element in one row.
            # The max of that row might change.
            new_row_max = max(row[:ei] + [new_val] + row[ei+1:])
            # The global max is max of all other rows' maxes and new_row_max
            # We can precompute maxes of other rows? Or just calculate.
            # Since n_rows is small (4), calculating max is cheap.
            
            # Apply move temporarily to calculate new_max
            # Actually, we can just compute:
            # old_max_of_row = max(row)
            # if old_val was the max, new_max_of_row might be smaller or larger.
            # Let's just compute max of the modified row.
            
            # To avoid modifying and reverting lists repeatedly which is slow in Python,
            # we can just compute the new max value.
            
            # Let's just compute new_max
            # This is O(k)
            new_max_val = max(max(r) for r in current_rows if r is not row)
            if new_row_max > new_max_val:
                new_max_val = new_row_max
            
            # Energy difference
            delta_E = new_max_val - current_max
            
            # Acceptance probability
            if delta_E < 0:
                accept = True
            else:
                if T > 0:
                    accept = random.random() < math.exp(-delta_E / T)
                else:
                    accept = False
            
            if accept:
                # Apply move
                row[ei] = new_val
                current_diffs -= old_diffs
                current_diffs |= new_diffs
                current_max = new_max_val
                
                if current_max < best_max:
                    best_max = current_max
                    best_rows = copy.deepcopy(current_rows)
                    best_diffs = copy.deepcopy(current_diffs)
                
                accepted = True
                break # Move accepted, break candidate loop
        
        if not accepted:
            # No move accepted or no valid candidate found
            pass
            
        T *= cooling
        
    return best_rows
