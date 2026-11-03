import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    row_len = k + 1
    # n_rows=7, k=5 -> row_len=6
    
    # 1. Greedy Initialization
    rows = []
    used_diffs = set()
    for _ in range(n_rows):
        row = [0]
        for _ in range(row_len - 1):
            found = False
            x = row[-1] + 1
            while not found:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
                    found = True
                else:
                    x += 1
        rows.append(row)
        
    current_scope = max(max(row) for row in rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # Helper to get diffs of a row
    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(row_len) for j in range(i+1, row_len)}
        
    # Helper to try a move
    # Returns (new_scope, old_diffs, new_diffs, old_val) if valid, else None
    def try_move(r_idx, pos_idx, new_val):
        row = rows[r_idx]
        if pos_idx == 0: return None
        if pos_idx > 0 and new_val <= row[pos_idx-1]: return None
        if pos_idx < row_len-1 and new_val >= row[pos_idx+1]: return None
        
        old_val = row[pos_idx]
        if old_val == new_val: return None
        
        # Calculate old diffs involving pos_idx
        old_diffs = set()
        for j in range(row_len):
            if j != pos_idx:
                old_diffs.add(abs(old_val - row[j]))
                
        # Remove from global
        used_diffs -= old_diffs
        
        # Apply move
        row[pos_idx] = new_val
        
        # Calculate new diffs involving pos_idx
        new_diffs = set()
        for j in range(row_len):
            if j != pos_idx:
                new_diffs.add(abs(new_val - row[j]))
                
        # Check conflict
        if new_diffs & used_diffs:
            # Revert
            row[pos_idx] = old_val
            used_diffs |= old_diffs
            return None
            
        # Accept
        used_diffs |= new_diffs
        new_scope = max(max(r) for r in rows)
        
        return new_scope, old_diffs, new_diffs, old_val

    # 2. Simulated Annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    steps = 0
    while time.time() < deadline - 20:
        r_idx = random.randint(0, n_rows-1)
        pos_idx = random.randint(1, row_len-1)
        
        row = rows[r_idx]
        
        # Bias: If this row has the global max, try to reduce its max element
        if max(row) == current_scope and random.random() < 0.3:
            pos_idx = row_len - 1
            low = row[row_len-2] + 1
            high = row[row_len-1] - 1
            if low > high:
                new_val = row[row_len-1]
            else:
                new_val = random.randint(low, high)
        else:
            low = row[pos_idx-1] + 1
            # Allow some expansion to escape local minima
            high = row[pos_idx+1] - 1 if pos_idx < row_len-1 else current_scope + 5
            high = min(high, current_scope + 10)
            if low > high:
                new_val = row[pos_idx]
            else:
                new_val = random.randint(low, high)
                
        res = try_move(r_idx, pos_idx, new_val)
        
        if res is None:
            continue
            
        new_scope, old_diffs, new_diffs, old_val = res
        delta = new_scope - current_scope
        
        if delta <= 0:
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        else:
            if random.random() < math.exp(-delta / T):
                current_scope = new_scope
            else:
                # Revert
                rows[r_idx][pos_idx] = old_val
                used_diffs -= new_diffs
                used_diffs |= old_diffs
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        steps += 1
        if steps % 5000 == 0 and time.time() > start_time + 90:
            break

    # 3. Final Polish: Try to reduce scope further with low T
    T = 0.1
    while time.time() < deadline:
        r_idx = random.randint(0, n_rows-1)
        # Focus on rows with high max
        if max(rows[r_idx]) < best_scope - 5:
            continue
            
        pos_idx = random.randint(1, row_len-1)
        row = rows[r_idx]
        
        # Try to move to a smaller value
        low = row[pos_idx-1] + 1
        high = row[pos_idx] - 1 # Only try to reduce
        
        if low > high:
            continue
            
        new_val = random.randint(low, high)
        
        res = try_move(r_idx, pos_idx, new_val)
        if res is None:
            continue
            
        new_scope, old_diffs, new_diffs, old_val = res
        if new_scope < current_scope:
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            rows[r_idx][pos_idx] = old_val
            used_diffs -= new_diffs
            used_diffs |= old_diffs

    return best_rows
