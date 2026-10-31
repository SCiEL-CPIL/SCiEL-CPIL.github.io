import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Greedy Initialization
    used_diffs = set()
    rows = []
    row_maxes = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
        row_maxes.append(max(row))
    
    current_max = max(row_maxes)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Helper to get diffs for a specific mark in a row
    def get_mark_diffs(row, idx, val=None):
        if val is None:
            val = row[idx]
        diffs = set()
        for i in range(idx):
            diffs.add(val - row[i])
        for j in range(idx + 1, len(row)):
            diffs.add(row[j] - val)
        return diffs

    T = 5.0  # Start with lower temperature since max is small
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        # Pick random row and mark
        r_idx = random.randint(0, n_rows - 1)
        m_idx = random.randint(1, k)
        
        row = rows[r_idx]
        old_val = row[m_idx]
        prev_val = row[m_idx - 1]
        next_val = row[m_idx + 1] if m_idx < k else float('inf')
        
        low = prev_val + 1
        high = next_val - 1
        
        if low > high:
            continue
            
        # Remove old diffs
        old_diffs = get_mark_diffs(row, m_idx)
        used_diffs.difference_update(old_diffs)
        
        found_valid = False
        new_val = -1
        new_diffs = set()
        
        # Try to find a valid new_val
        # Bias towards smaller values
        # We'll try a few random candidates
        for _ in range(10):
            if random.random() < 0.7 and old_val > low:
                candidate = random.randint(low, old_val)
            else:
                candidate = random.randint(low, high)
                
            if candidate == old_val:
                continue
                
            cand_diffs = get_mark_diffs(row, m_idx, candidate)
            if not (cand_diffs & used_diffs):
                new_val = candidate
                new_diffs = cand_diffs
                found_valid = True
                break
        
        if not found_valid:
            # Revert
            used_diffs |= old_diffs
            continue
            
        # Calculate new max
        # Only row r_idx changes
        # New max of row r_idx
        # If new_val > old_val, max might increase.
        # If new_val < old_val, max might decrease.
        
        # We can compute new_row_max efficiently
        # The max of the row is either:
        # 1. The max of the parts that didn't change (left of m_idx, right of m_idx)
        # 2. new_val
        
        # Max of left part: row[m_idx-1] (since sorted)
        # Max of right part: row[-1] (since sorted)
        # So old_max was max(row[m_idx-1], old_val, row[-1])? No, max of whole row.
        # Since sorted, max(row) is row[-1].
        # Wait, if m_idx is the last index (k), then row[-1] is the mark.
        # If m_idx < k, then row[-1] is some other mark.
        
        # Let's just compute max of the new row.
        # new_row = row[:]
        # new_row[m_idx] = new_val
        # new_row_max = max(new_row)
        
        # Optimization:
        # If m_idx == k (last mark):
        #   new_row_max = new_val
        # Else:
        #   new_row_max = max(new_val, row[-1]) # Since row[-1] is the largest of the other marks
        
        if m_idx == k:
            new_row_max = new_val
        else:
            new_row_max = max(new_val, row[-1])
            
        # Update row_maxes
        old_row_max = row_maxes[r_idx]
        
        # Calculate new global max
        # We can update row_maxes[r_idx] temporarily
        row_maxes[r_idx] = new_row_max
        new_global_max = max(row_maxes)
        
        # Acceptance
        delta = new_global_max - current_max
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.5
        else:
            accept = random.random() < math.exp(-delta / T)
            
        if accept:
            # Commit
            row[m_idx] = new_val
            used_diffs |= new_diffs
            current_max = new_global_max
            # row_maxes[r_idx] is already updated
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            # row[m_idx] is still old_val, so no need to revert row
            used_diffs |= old_diffs
            row_maxes[r_idx] = old_row_max
            # current_max is still old value, no need to revert
            
        # Cooling
        T *= cooling_rate
        
    return best_rows
