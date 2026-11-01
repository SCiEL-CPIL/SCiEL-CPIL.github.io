import time
import random
import math

def construct(n_rows, k):
    """
    Finds a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct, minimizing the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Helper to calculate differences for a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Initial Solution: Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_cost = max(max(row) for row in best_rows)
    
    # Simulated Annealing
    def sa_search(rows, max_time):
        cur_rows = [list(r) for r in rows]
        cur_cost = max(max(r) for r in cur_rows)
        best_local = [list(r) for r in cur_rows]
        best_local_cost = cur_cost
        
        # Precompute used diffs for current state
        used_diffs = set()
        for r in cur_rows:
            used_diffs |= get_diffs(r)
            
        T = 10.0  # Initial temperature
        cooling = 0.99995
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < max_time:
            # Pick a random row and index
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k) # Index 0 is always 0
            
            old_val = cur_rows[r_idx][j_idx]
            
            # Determine range for new value
            low = cur_rows[r_idx][j_idx - 1] + 1
            high = cur_rows[r_idx][j_idx + 1] - 1 if j_idx < k else old_val + 20 # Allow some increase
            
            # Random step
            # Bias towards reduction if old_val is large
            step = random.randint(-15, 15)
            new_val = old_val + step
            
            # Enforce bounds
            if new_val < low:
                new_val = low
            if new_val > high:
                new_val = high
                
            if new_val == old_val:
                continue
                
            # Check strict increase
            if j_idx > 0 and new_val <= cur_rows[r_idx][j_idx-1]:
                continue
            if j_idx < k and new_val >= cur_rows[r_idx][j_idx+1]:
                continue
                
            # Check validity
            row = cur_rows[r_idx]
            
            # Calculate old and new diffs
            old_diffs = set()
            new_diffs = set()
            for i in range(len(row)):
                if i == j_idx:
                    continue
                d_old = abs(row[i] - old_val)
                d_new = abs(row[i] - new_val)
                if d_old > 0:
                    old_diffs.add(d_old)
                if d_new > 0:
                    new_diffs.add(d_new)
            
            # Check conflicts
            # new_diffs must not intersect with (used_diffs - old_diffs)
            # Equivalent to: (new_diffs & used_diffs) must be a subset of old_diffs
            intersection = new_diffs & used_diffs
            if not intersection.issubset(old_diffs):
                continue
            
            # Valid move
            # Update state
            row[j_idx] = new_val
            used_diffs -= old_diffs
            used_diffs |= new_diffs
            
            new_cost = max(max(r) for r in cur_rows)
            
            # Acceptance
            delta = new_cost - cur_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                cur_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local = [list(r) for r in cur_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                row[j_idx] = old_val
                used_diffs -= new_diffs
                used_diffs |= old_diffs
                no_improve_count += 1
                
            T *= cooling
            if T < 0.01:
                T = 0.01
            
            if no_improve_count > max_no_improve:
                # Restart or perturb
                T = 5.0
                no_improve_count = 0
                # Perturb
                r_p = random.randint(0, n_rows - 1)
                j_p = random.randint(1, k)
                old_vp = cur_rows[r_p][j_p]
                low_p = cur_rows[r_p][j_p - 1] + 1
                high_p = cur_rows[r_p][j_p + 1] - 1 if j_p < k else old_vp + 30
                new_vp = random.randint(low_p, high_p)
                if new_vp != old_vp:
                    row_p = cur_rows[r_p]
                    if j_p > 0 and new_vp <= row_p[j_p-1]: continue
                    if j_p < k and new_vp >= row_p[j_p+1]: continue
                    
                    old_diffs_p = set(abs(row_p[i] - old_vp) for i in range(len(row_p)) if i != j_p and abs(row_p[i] - old_vp) > 0)
                    new_diffs_p = set(abs(row_p[i] - new_vp) for i in range(len(row_p)) if i != j_p and abs(row_p[i] - new_vp) > 0)
                    
                    inter_p = new_diffs_p & used_diffs
                    if inter_p.issubset(old_diffs_p):
                        row_p[j_p] = new_vp
                        used_diffs -= old_diffs_p
                        used_diffs |= new_diffs_p
                        cur_cost = max(max(r) for r in cur_rows)

        return best_local, best_local_cost

    # Run SA
    # We run SA in chunks to allow for global updates
    while time.time() < deadline:
        run_deadline = min(time.time() + 5, deadline)
        res_rows, res_cost = sa_search(best_rows, run_deadline)
        
        if res_cost < best_cost:
            best_cost = res_cost
            best_rows = res_rows
            
        # If we reach the known best 134, we can stop early if time is tight?
        # Or just keep trying to beat it.
        
    return best_rows
