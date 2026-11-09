import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Helper to get diffs of a row
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            ri = row[i]
            for j in range(i+1, len(row)):
                diffs.append(row[j] - ri)
        return diffs

    # 1. Greedy Initialization
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_init()
    
    # State for incremental cost
    # global_diffs: dict {diff: count}
    # total_conflicts: int
    # max_val: int
    
    def compute_state(rows):
        global_diffs = {}
        total_conflicts = 0
        max_val = 0
        for r in rows:
            max_val = max(max_val, r[-1])
            diffs = get_diffs(r)
            for d in diffs:
                if d in global_diffs:
                    global_diffs[d] += 1
                    if global_diffs[d] == 2:
                        total_conflicts += 1
                    elif global_diffs[d] > 2:
                        total_conflicts += 1
                else:
                    global_diffs[d] = 1
        return global_diffs, total_conflicts, max_val

    best_gd, best_conflicts, best_max = compute_state(best_rows)
    best_cost = best_max + 1000 * best_conflicts
    
    # Current state
    cur_rows = [r[:] for r in best_rows]
    cur_gd, cur_conflicts, cur_max = compute_state(cur_rows)
    cur_cost = cur_max + 1000 * cur_conflicts
    
    T = 1000.0
    T_min = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k)
        
        old_val = cur_rows[r_idx][j_idx]
        min_val = cur_rows[r_idx][j_idx-1] + 1
        max_val = cur_rows[r_idx][j_idx+1] - 1 if j_idx < k else old_val + 50
        
        if min_val > max_val:
            continue
            
        if random.random() < 0.5:
            new_val = min_val + random.randint(0, max(0, (old_val - min_val)//2))
        else:
            new_val = random.randint(min_val, max_val)
            
        if new_val == old_val:
            continue
            
        # Update row
        cur_rows[r_idx][j_idx] = new_val
        
        # Update diffs
        old_diffs = get_diffs(cur_rows[r_idx]) # This is wrong, we need old diffs BEFORE change
        # Let's backtrack:
        cur_rows[r_idx][j_idx] = old_val
        old_diffs = get_diffs(cur_rows[r_idx])
        
        cur_rows[r_idx][j_idx] = new_val
        new_diffs = get_diffs(cur_rows[r_idx])
        
        # Update global_diffs and conflicts
        # Remove old
        for d in old_diffs:
            cur_gd[d] -= 1
            if cur_gd[d] == 1:
                cur_conflicts -= 1
            # if it becomes 0, it's gone, no conflict change from 1->0
        # Add new
        for d in new_diffs:
            if d in cur_gd:
                cur_gd[d] += 1
                if cur_gd[d] == 2:
                    cur_conflicts += 1
                elif cur_gd[d] > 2:
                    cur_conflicts += 1
            else:
                cur_gd[d] = 1
                
        new_max = max(cur_rows[r][-1] for r in cur_rows)
        
        new_cost = new_max + 1000 * cur_conflicts
        
        delta = new_cost - cur_cost
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            cur_cost = new_cost
            cur_max = new_max
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [r[:] for r in cur_rows]
                best_gd, best_conflicts, best_max = compute_state(best_rows) # Re-compute for safety? Or copy?
                # Copying state is safer
                best_gd = dict(cur_gd)
                best_conflicts = cur_conflicts
                best_max = cur_max
        else:
            # Revert
            cur_rows[r_idx][j_idx] = old_val
            # Revert global state
            # Remove new
            for d in new_diffs:
                cur_gd[d] -= 1
                if cur_gd[d] == 1:
                    cur_conflicts -= 1
            # Add old
            for d in old_diffs:
                if d in cur_gd:
                    cur_gd[d] += 1
                    if cur_gd[d] == 2:
                        cur_conflicts += 1
                    elif cur_gd[d] > 2:
                        cur_conflicts += 1
                else:
                    cur_gd[d] = 1
            cur_max = max(cur_rows[r][-1] for r in cur_rows)
            cur_cost = cur_max + 1000 * cur_conflicts

        T *= cooling
        if T < T_min:
            T = 500.0
            # Restart from best
            cur_rows = [r[:] for r in best_rows]
            cur_gd, cur_conflicts, cur_max = compute_state(cur_rows)
            cur_cost = cur_max + 1000 * cur_conflicts
            # Perturb
            for _ in range(5):
                rr = random.randint(0, n_rows - 1)
                jj = random.randint(1, k)
                ov = cur_rows[rr][jj]
                mn = cur_rows[rr][jj-1] + 1
                mx = cur_rows[rr][jj+1] - 1 if jj < k else ov + 50
                if mn <= mx:
                    cur_rows[rr][jj] = random.randint(mn, mx)
            cur_gd, cur_conflicts, cur_max = compute_state(cur_rows)
            cur_cost = cur_max + 1000 * cur_conflicts

    return best_rows
