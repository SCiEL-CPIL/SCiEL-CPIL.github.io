import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = 145.0
    
    num_elements = k + 1
    # n_rows=7, k=5 -> num_elements=6
    # Total diffs = 7 * 15 = 105
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = set()
        for r in rows:
            if len(r) != num_elements: return False
            if r[0] != 0: return False
            if any(r[i] >= r[i+1] for i in range(len(r)-1)): return False
            diffs = get_diffs(r)
            if len(diffs) != num_elements * (num_elements-1) // 2: return False
            if all_diffs & diffs: return False
            all_diffs.update(diffs)
        return True

    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    def simulated_annealing():
        nonlocal best_rows, best_scope
        rows = [list(r) for r in best_rows]
        used_diffs = set()
        for r in rows:
            used_diffs.update(get_diffs(r))
            
        current_scope = max(max(r) for r in rows)
        
        T = 10.0
        cooling = 0.995
        step = 0
        
        while time.time() < DEADLINE and T > 0.01:
            # Pick a random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick a random index to change (not the first one)
            # We can change any element except a[0]=0
            # If we change a[i], we need to re-sort
            i = random.randint(1, num_elements - 1)
            old_val = row[i]
            
            # Determine range for new value
            # It must be distinct from other row elements
            # To keep scope low, try values smaller than current max
            # But also allow some larger values to escape traps
            
            # Heuristic: Try values in [1, max_scope * 1.2]
            # But bias towards smaller
            max_try = max(current_scope + 10, 50)
            
            # Try a few random moves per T step? Or just one?
            # Let's do one move per loop iteration for simplicity, but fast.
            
            new_val = random.randint(1, max_try)
            
            # Check if new_val is already in row (ignoring index i)
            # We remove old_val, so if new_val == old_val, no change.
            # If new_val is in row, invalid.
            if new_val in row and new_val != old_val:
                continue
                
            # Construct candidate row
            new_row = row.copy()
            new_row.pop(i)
            new_row.append(new_val)
            new_row.sort()
            
            # Check if new_row[0] is 0
            if new_row[0] != 0:
                continue
                
            # Check distinctness
            if len(set(new_row)) != num_elements:
                continue
                
            # Compute diffs for new_row
            new_diffs = get_diffs(new_row)
            
            # Remove old diffs from used_diffs
            old_diffs = get_diffs(row)
            used_diffs.difference_update(old_diffs)
            
            # Check conflict
            if used_diffs & new_diffs:
                used_diffs.update(old_diffs)
                continue
            else:
                used_diffs.update(new_diffs)
                rows[r_idx] = new_row
                new_scope = max(max(r) for r in rows)
                
                # Acceptance
                delta = new_scope - current_scope
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                else:
                    # Reject
                    used_diffs.difference_update(new_diffs)
                    used_diffs.update(old_diffs)
                    rows[r_idx] = row # Restore
            
            T *= cooling
            step += 1
            
    # Run SA
    simulated_annealing()
    
    # If we have time, try to refine with local search (greedy reduction)
    # Or just return best
    if time.time() < DEADLINE - 5:
         # Try to fix best_rows to be even better?
         # The SA should have found the minimum.
         pass
         
    return best_rows
