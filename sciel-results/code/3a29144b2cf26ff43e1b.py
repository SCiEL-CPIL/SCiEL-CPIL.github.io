import time
import random
import math

def _greedy_construct(n_rows, k):
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

def _get_row_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _calculate_scope(rows):
    return max(row[-1] for row in rows)

def _sa_optimize(initial_rows, time_limit=110.0):
    rows = [row[:] for row in initial_rows]
    used_diffs = set()
    for row in rows:
        used_diffs |= _get_row_diffs(row)
        
    current_scope = _calculate_scope(rows)
    best_rows = [row[:] for row in rows]
    best_scope = current_scope
    
    row_diffs = [_get_row_diffs(row) for row in rows]
    
    start_time = time.time()
    T = 10.0
    cooling_rate = 0.99995
    min_T = 0.001
    
    n_rows = len(rows)
    k = len(rows[0]) - 1
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Select move type
        if random.random() < 0.7:
            # Compaction move: Try to reduce the last element of the row with max scope
            r = max(range(n_rows), key=lambda i: rows[i][-1])
            old_row = rows[r]
            old_diffs = row_diffs[r]
            
            # Try to reduce last mark
            for delta in range(1, 10):
                new_val = old_row[-1] - delta
                if new_val <= old_row[-2]:
                    break
                new_row = old_row[:]
                new_row[-1] = new_val
                
                # Check validity
                new_diffs = _get_row_diffs(new_row)
                other_diffs = used_diffs - old_diffs
                if new_diffs & other_diffs:
                    continue
                
                # Valid move
                new_scope = max(row[-1] for row in rows) # Need to update row first to calc scope?
                # Actually, only row r changed.
                other_max = max(rows[i][-1] for i in range(n_rows) if i != r)
                new_scope = max(other_max, new_val)
                
                # Acceptance
                if new_scope <= current_scope:
                    # Accept
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    rows[r] = new_row
                    row_diffs[r] = new_diffs
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [row[:] for row in rows]
                    break
                else:
                    # Reject bad compaction move (don't accept worsening scope in compaction usually, 
                    # but maybe with prob? No, compaction is for improvement. Let's stick to improvement only for this move type to be safe and fast)
                    pass
        else:
            # Perturbation move: Random jitter
            r = random.randint(0, n_rows - 1)
            old_row = rows[r]
            old_diffs = row_diffs[r]
            
            # Pick a mark to jitter (not the first one)
            j = random.randint(1, k)
            
            # Jitter
            delta = random.randint(-5, 5)
            new_val = old_row[j] + delta
            
            # Check bounds
            if new_val <= old_row[j-1]:
                continue
            if j < k and new_val >= old_row[j+1]:
                continue
                
            new_row = old_row[:]
            new_row[j] = new_val
            
            # Check validity
            new_diffs = _get_row_diffs(new_row)
            other_diffs = used_diffs - old_diffs
            if new_diffs & other_diffs:
                continue
                
            # Calculate new scope
            other_max = max(rows[i][-1] for i in range(n_rows) if i != r)
            new_scope = max(other_max, new_row[-1])
            
            # Acceptance
            if new_scope <= current_scope:
                # Accept
                used_diffs -= old_diffs
                used_diffs |= new_diffs
                rows[r] = new_row
                row_diffs[r] = new_diffs
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [row[:] for row in rows]
            else:
                delta_E = new_scope - current_scope
                prob = math.exp(-delta_E / T)
                if random.random() < prob:
                    # Accept bad move
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    rows[r] = new_row
                    row_diffs[r] = new_diffs
                    current_scope = new_scope
                    
        T *= cooling_rate

    return best_rows

def construct(n_rows, k):
    """
    Entry point.
    """
    rows = _greedy_construct(n_rows, k)
    rows = _sa_optimize(rows, time_limit=110.0)
    return rows
