import random
import math
import time

def construct(n_rows, k):
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_init():
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

    # Initial Solution
    rows = greedy_init()
    row_diffs = [get_diffs(r) for r in rows]
    all_used = set()
    for d in row_diffs:
        all_used |= d
        
    current_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # SA Parameters
    start_time = time.time()
    timeout = 110.0
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() - start_time < timeout:
        # Select Row
        # Bias to row with max
        max_val = max(max(r) for r in rows)
        max_rows = [i for i, r in enumerate(rows) if max(r) == max_val]
        
        if random.random() < 0.6:
            r_idx = random.choice(max_rows)
        else:
            r_idx = random.randint(0, n_rows - 1)
            
        # Select Position
        # Bias to last position
        if random.random() < 0.4:
            pos = k
        else:
            pos = random.randint(1, k)
            
        old_val = rows[r_idx][pos]
        old_diffs = row_diffs[r_idx]
        
        # Define Range
        low = rows[r_idx][pos-1] + 1
        high = rows[r_idx][pos+1] - 1 if pos < k else old_val
        
        if low > high:
            continue
            
        # Try to find a new value
        # Strategy: Try to reduce if possible, otherwise random
        # We try a few candidates
        
        accepted = False
        # Number of candidates to try
        num_candidates = 3
        
        for _ in range(num_candidates):
            # Generate candidate
            if random.random() < 0.5:
                # Try to reduce
                if old_val > low:
                    new_val = random.randint(low, old_val)
                else:
                    new_val = random.randint(low, high)
            else:
                new_val = random.randint(low, high)
                
            if new_val == old_val:
                continue
                
            # Check Validity
            all_used -= old_diffs
            rows[r_idx][pos] = new_val
            new_diffs = get_diffs(rows[r_idx])
            
            if new_diffs & all_used:
                # Conflict
                rows[r_idx][pos] = old_val
                all_used |= old_diffs
                continue
            else:
                # Valid Move Found
                new_max = max(max(r) for r in rows)
                delta = new_max - current_max
                
                # Acceptance
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < math.exp(-delta / T)
                
                if accept:
                    # Commit Move
                    row_diffs[r_idx] = new_diffs
                    all_used |= new_diffs
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in rows]
                    accepted = True
                    break
                else:
                    # Reject Move
                    rows[r_idx][pos] = old_val
                    all_used |= old_diffs
                    # Continue to next candidate? Or break?
                    # Breaking is faster, but trying another candidate might find a better one.
                    # Let's try another candidate.
                    continue
                    
        T *= cooling
        if T < min_T:
            break
            
    return best_rows
