import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety
    
    # Helper to check if a move is valid
    def is_valid_move(row, idx, new_val, used_diffs):
        # Check strict increase
        if idx > 0 and new_val <= row[idx-1]:
            return False, None, None
        if idx < k and new_val >= row[idx+1]:
            return False, None, None
        
        old_val = row[idx]
        if old_val == new_val:
            return True, set(), set()
            
        # Calculate old diffs involving idx
        old_diffs = set()
        for i in range(k + 1):
            if i != idx:
                old_diffs.add(abs(old_val - row[i]))
                
        # Calculate new diffs involving idx
        new_diffs = set()
        for i in range(k + 1):
            if i != idx:
                new_diffs.add(abs(new_val - row[i]))
        
        # Check for conflicts with used_diffs excluding old_diffs
        conflict = new_diffs & (used_diffs - old_diffs)
        if conflict:
            return False, None, None
            
        return True, old_diffs, new_diffs

    def greedy_init(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                search_limit = 2000 
                for x in range(row[-1] + 1, search_limit):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    return None, None
            rows.append(row)
        return rows, used

    def get_max(rows):
        return max(max(row) for row in rows)

    # 1. Initialize
    best_rows = None
    best_used = None
    best_max = float('inf')
    
    # Try a few random greedy inits to find a good starting point
    for attempt in range(10):
        if time.time() > deadline: break
        rows, used = greedy_init(attempt)
        if rows:
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                best_used = set(used)
                
    if best_rows is None:
        # Fallback: deterministic greedy
        rows, used = greedy_init(0)
        if rows:
            best_rows = rows
            best_used = used
            best_max = get_max(rows)
        else:
            # Should not happen
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows

    current_rows = [r[:] for r in best_rows]
    current_used = set(best_used)
    current_max = best_max
    
    # 2. Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        r = random.randint(0, n_rows - 1)
        idx = random.randint(1, k)
        
        old_val = current_rows[r][idx]
        
        if random.random() < 0.7:
            min_val = current_rows[r][idx-1] + 1
            max_val_candidate = old_val - 1
            if min_val <= max_val_candidate:
                new_val = random.randint(min_val, max_val_candidate)
            else:
                new_val = old_val
        else:
            max_search = current_max + 10
            new_val = random.randint(0, max_search)
            if new_val <= current_rows[r][idx-1]:
                new_val = current_rows[r][idx-1] + 1
                
        valid, old_diffs, new_diffs = is_valid_move(current_rows[r], idx, new_val, current_used)
        
        if valid:
            new_max = current_max
            if old_val == current_max and new_val < old_val:
                max_other = 0
                for rr in range(n_rows):
                    if rr != r:
                        if max(current_rows[rr]) > max_other:
                            max_other = max(current_rows[rr])
                max_this_row = max(current_rows[r][:idx] + [new_val] + current_rows[r][idx+1:])
                new_max = max(max_other, max_this_row)
            elif new_val > current_max:
                new_max = new_val
            else:
                new_max = current_max

            delta = new_max - current_max
            
            if delta < 0:
                current_used -= old_diffs
                current_used |= new_diffs
                current_rows[r][idx] = new_val
                current_max = new_max
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [row[:] for row in current_rows]
                    best_used = set(current_used)
                    
            elif delta == 0:
                if random.random() < 0.5:
                    current_used -= old_diffs
                    current_used |= new_diffs
                    current_rows[r][idx] = new_val
            else:
                if T > 0:
                    prob = pow(2.718281828, -delta / T)
                    if random.random() < prob:
                        current_used -= old_diffs
                        current_used |= new_diffs
                        current_rows[r][idx] = new_val
                        current_max = new_max
        else:
            pass
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_rows
# EVOLVE-BLOCK-END
