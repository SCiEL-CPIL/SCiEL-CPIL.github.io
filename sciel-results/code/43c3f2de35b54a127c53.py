import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Initial Greedy Solution
    def generate_greedy():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used_diffs

    rows, used_diffs = generate_greedy()
    
    def get_max_scope(rows):
        return max(max(r) for r in rows)
    
    current_scope = get_max_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # Helper to get involved diffs
    def get_involved_diffs(row, pos, val):
        diffs = set()
        for i in range(pos):
            diffs.add(val - row[i])
        for j in range(pos + 1, len(row)):
            diffs.add(row[j] - val)
        return diffs

    # Phase 1: Hill Climbing (First Improvement)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for r_idx in range(n_rows):
            row = rows[r_idx]
            for pos in range(1, k + 1):
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < k else old_val
                
                # Try to reduce value
                for new_val in range(low, old_val):
                    old_involved = get_involved_diffs(row, pos, old_val)
                    used_diffs.difference_update(old_involved)
                    
                    new_involved = get_involved_diffs(row, pos, new_val)
                    
                    if len(new_involved) != len(old_involved):
                        used_diffs |= old_involved
                        continue
                    
                    if new_involved & used_diffs:
                        used_diffs |= old_involved
                        continue
                    
                    # Valid
                    used_diffs |= new_involved
                    row[pos] = new_val
                    
                    new_scope = get_max_scope(rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        improved = True
                        # Restart scan from beginning of this row? Or just break inner loop?
                        break
                    else:
                        # Revert
                        used_diffs.difference_update(new_involved)
                        used_diffs |= old_involved
                        row[pos] = old_val
                if improved:
                    break
            if improved:
                break

    # Phase 2: Simulated Annealing
    T = 5.0
    cooling = 0.99
    
    while time.time() < deadline - 1:
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick a random position
        pos = random.randint(1, k)
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < k else old_val + 50
        
        # Bias: if this is the max, try smaller. Else, try nearby.
        if old_val == max(row):
            if low < old_val:
                new_val = random.randint(low, old_val - 1)
            else:
                continue
        else:
            # Random neighborhood
            search_low = max(low, old_val - 10)
            search_high = min(high, old_val + 10)
            if search_low > search_high:
                continue
            new_val = random.randint(search_low, search_high)
            
        if new_val == old_val:
            continue
            
        old_involved = get_involved_diffs(row, pos, old_val)
        used_diffs.difference_update(old_involved)
        
        new_involved = get_involved_diffs(row, pos, new_val)
        
        if len(new_involved) != len(old_involved):
            used_diffs |= old_involved
            continue
        
        if new_involved & used_diffs:
            used_diffs |= old_involved
            continue
        
        # Valid move
        used_diffs |= new_involved
        row[pos] = new_val
        
        new_scope = get_max_scope(rows)
        
        if new_scope < current_scope:
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        else:
            delta = new_scope - current_scope
            if random.random() < math.exp(-delta / T):
                current_scope = new_scope
            else:
                # Revert
                used_diffs.difference_update(new_involved)
                used_diffs |= old_involved
                row[pos] = old_val
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_rows
