import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    best_max = float('inf')
    best_rows = None

    # --- Helper Functions ---
    
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_max(rows):
        return max(max(r) for r in rows)

    # --- Greedy Baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = compute_max(best_rows)

    # --- Simulated Annealing ---
    
    def sa_optimize(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len
        
        # Precompute diffs per row and global counts
        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        # Calculate initial collision count
        collision_count = 0
        for d, c in diff_counts.items():
            if c > 1:
                collision_count += c - 1
                
        current_max = compute_max(rows)
        
        def get_cost():
            # Weight for collisions must be high enough to dominate max
            # Max is roughly < 200. Collisions can be up to 105.
            # If weight is 1000, collision cost is up to 105000. Max cost is < 200.
            # So 1000 is safe.
            return collision_count * 10000 + current_max

        current_cost = get_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max

        T = 100.0
        cooling = 0.9998
        min_T = 1.0
        start_time = time.time()
        
        # Weights for cost components
        W_COLL = 10000
        
        while time.time() - start_time < time_limit:
            # Select random row and index
            r_idx = random.randint(0, n - 1)
            # Allow moving indices 1 to L-1 (last element is L-1)
            # Moving index 0 is not allowed (must be 0)
            i_idx = random.randint(1, L - 1)
            
            row = rows[r_idx]
            old_val = row[i_idx]
            
            # Determine valid range for new_val
            lower = row[i_idx - 1] + 1
            # Upper bound: if not last, must be < next. If last, can be large.
            if i_idx < L - 1:
                upper = row[i_idx + 1] - 1
            else:
                # Allow expansion up to current_max + some margin
                upper = current_max + 20
            
            if lower > upper:
                continue
                
            # Random new value
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
            
            # --- Update State ---
            
            # 1. Remove old diffs
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                c = diff_counts[d]
                if c == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = c - 1
                    if c >= 2:
                        collision_count -= 1
            
            # 2. Update row
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs
            
            # 3. Add new diffs
            for d in new_diffs:
                c = diff_counts.get(d, 0)
                if c == 0:
                    diff_counts[d] = 1
                else:
                    diff_counts[d] = c + 1
                    if c >= 1:
                        collision_count += 1
            
            # 4. Update Max
            new_row_max = row[-1]
            # Optimization: only recompute global max if this row's max changed or was the global max
            # For simplicity and safety with small n, recompute global max
            # But we can be smarter:
            # If new_row_max < current_max, global max might stay same or decrease (if this was the max)
            # If new_row_max > current_max, global max is new_row_max
            
            # Let's just recompute for correctness, it's O(n*k) = 42 ops
            new_global_max = compute_max(rows)
            
            new_cost = collision_count * W_COLL + new_global_max
            
            # --- Acceptance ---
            delta = new_cost - current_cost
            accept = False
            
            if delta < 0:
                accept = True
            else:
                # If we are in a valid state (0 collisions), we should be very strict about increasing max
                # But SA handles this via temperature.
                # However, if delta is huge (e.g. introducing collisions), exp(-delta/T) will be 0.
                if T > min_T:
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        accept = True
                # If T is low, only accept if delta == 0 (neutral)
                elif delta == 0:
                    if random.random() < 0.5:
                        accept = True
            
            if accept:
                current_cost = new_cost
                current_max = new_global_max
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = new_global_max
            else:
                # Revert
                row[i_idx] = old_val
                row_diffs[r_idx] = old_diffs
                
                # Re-add old diffs
                for d in old_diffs:
                    c = diff_counts.get(d, 0)
                    if c == 0:
                        diff_counts[d] = 1
                    else:
                        diff_counts[d] = c + 1
                        if c >= 1:
                            collision_count += 1
                            
                # Remove new diffs
                for d in new_diffs:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                        if c >= 2:
                            collision_count -= 1
                
                current_max = compute_max(rows)
                # current_cost reverts implicitly by not updating it, but we need to be sure
                # Actually, we didn't update current_cost, so it's still the old value.
                # But we did update collision_count and rows.
                # We must ensure collision_count is restored correctly.
                # The revert logic above restores diff_counts and collision_count.
                
            T *= cooling
            if T < min_T:
                T = min_T

        return best_local_rows, best_local_max

    # --- Polish Phase ---
    def polish(rows, time_limit):
        # Assumes rows are valid (0 collisions)
        # Try to reduce the max element of each row
        rows = [r[:] for r in rows]
        current_max = compute_max(rows)
        start_time = time.time()
        
        # Rebuild diff counts
        diff_counts = {}
        for r in rows:
            for d in get_diffs(r):
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        # Verify validity
        for d, c in diff_counts.items():
            if c > 1:
                return rows, current_max # Not valid, return as is
                
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                # Try to reduce the last element
                # We can try reducing it by 1, 2, ...
                # But we must maintain strict increase.
                # So new_val must be > row[-2]
                
                # Try values from row[-1] - 1 down to row[-2] + 1
                for new_val in range(row[-1] - 1, row[-2], -1):
                    # Check if moving row[-1] to new_val is valid
                    # Remove old diffs involving row[-1]
                    old_last = row[-1]
                    old_diffs_to_remove = [old_last - x for x in row[:-1]]
                    
                    # Check if new diffs conflict
                    new_diffs_to_add = [new_val - x for x in row[:-1]]
                    
                    # Check internal conflicts (none, since new_val < old_last and row is increasing)
                    # Check external conflicts
                    
                    # Temporarily remove old diffs
                    valid = True
                    for d in old_diffs_to_remove:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                            
                    # Check new diffs
                    for d in new_diffs_to_add:
                        if d in diff_counts:
                            valid = False
                            break
                            
                    if valid:
                        # Accept change
                        row[-1] = new_val
                        for d in new_diffs_to_add:
                            diff_counts[d] = 1
                        improved = True
                        current_max = min(current_max, new_val) # Actually recompute
                        current_max = compute_max(rows)
                        break # Move to next row
                    else:
                        # Revert removal
                        for d in old_diffs_to_remove:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        
        return rows, current_max

    # Main Loop
    # Use most of the time for SA, last 5s for polish
    
    sa_time = min(105.0, deadline - time.time())
    if sa_time > 0:
        local_rows, local_max = sa_optimize(best_rows, sa_time)
        
        # Check if local solution is valid (0 collisions)
        # We need to check collisions in local_rows
        all_diffs = []
        for r in local_rows:
            all_diffs.extend(get_diffs(r))
        if len(all_diffs) == len(set(all_diffs)):
            if local_max < best_max:
                best_rows = local_rows
                best_max = local_max
    
    # Polish
    if time.time() < deadline - 1.0:
        polish_time = deadline - time.time() - 1.0
        polished_rows, polished_max = polish(best_rows, polish_time)
        if polished_max < best_max:
            best_rows = polished_rows
            best_max = polished_max

    return best_rows
