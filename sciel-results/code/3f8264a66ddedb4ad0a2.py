import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 118.0
    best_max = float('inf')
    best_rows = None

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
    best_max = max(max(r) for r in best_rows)

    # --- Helper: get all pairwise differences of a row ---
    def get_diffs(row):
        diffs = []
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                diffs.append(row[j] - row[i])
        return diffs

    # --- Row reconstruction hill climbing (from inspiration) ---
    def reconstruct_row(r_idx, rows, used_diffs, upper_bound):
        """Re-greedy row r_idx against the current set of used differences."""
        # Remove current row's differences from used
        current_row = rows[r_idx]
        current_diffs = set(get_diffs(current_row))
        for d in current_diffs:
            used_diffs.discard(d)
        
        # Re-greedy this row
        new_row = [0]
        x = 0
        while len(new_row) < row_len:
            x += 1
            if x > upper_bound:
                break
            new_diffs = {x - a for a in new_row}
            if not new_diffs & used_diffs:
                new_row.append(x)
                used_diffs |= new_diffs
        
        if len(new_row) == row_len:
            rows[r_idx] = new_row
            return True
        else:
            # Rollback
            for d in set(get_diffs(new_row)):
                used_diffs.discard(d)
            used_diffs |= current_diffs
            return False

    def hill_climb_rows(rows, max_val, time_limit):
        """Iteratively reconstruct rows to shrink the bounding box."""
        rows = [r[:] for r in rows]
        max_val = max(max(r) for r in rows)
        start_time = time.time()
        
        # Build used_diffs set
        used_diffs = set()
        for r in rows:
            for d in get_diffs(r):
                used_diffs.add(d)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for r_idx in range(n_rows):
                if time.time() - start_time >= time_limit:
                    break
                old_max = rows[r_idx][-1]
                if reconstruct_row(r_idx, rows, used_diffs, max_val):
                    new_max = rows[r_idx][-1]
                    if new_max < old_max:
                        improved = True
                        if new_max < max_val:
                            # Try to tighten global bound
                            max_val = max(max(r) for r in rows)
        
        return rows, max(max(r) for r in rows)

    # --- SA for final polishing ---
    def sa_optimize(initial_rows, time_limit, bound):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        row_diffs = [get_diffs(r) for r in rows]
        
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        def compute_cost():
            collisions = 0
            for d, c in diff_counts.items():
                if c > 1:
                    collisions += c - 1
            max_val = max(r[-1] for r in rows)
            return collisions * 10000 + max_val

        current_cost = compute_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        T = 50.0
        cooling = 0.9995
        start_time = time.time()

        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)
            
            row = rows[r_idx]
            old_val = row[i_idx]
            
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 10
            upper = min(upper, bound + 20)
            
            if lower > upper:
                continue
                
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
            
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs
            
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_cost = compute_cost()
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = max(r[-1] for r in best_local_rows)
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.5:
                        current_cost = new_cost
                    else:
                        row[i_idx] = old_val
                        for d in new_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
                else:
                    if delta < 1000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if random.random() < p:
                        current_cost = new_cost
                    else:
                        row[i_idx] = old_val
                        for d in new_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
            
            T *= cooling
            if T < 0.1:
                T = 0.1

        return best_local_rows, best_local_max, best_local_cost

    # --- Row-reconstruction based local search (key improvement) ---
    def row_reconstruct_search(rows, max_val, time_limit):
        """
        Use iterative row reconstruction to improve the solution.
        Each row is re-greedy'd against the current used differences.
        """
        rows = [r[:] for r in rows]
        max_val = max(max(r) for r in rows)
        start_time = time.time()
        
        # Build used_diffs set
        used_diffs = set()
        for r in rows:
            for d in get_diffs(r):
                used_diffs.add(d)
        
        # Iteratively try to reconstruct each row
        for iteration in range(50):
            if time.time() - start_time >= time_limit:
                break
            
            for r_idx in range(n_rows):
                if time.time() - start_time >= time_limit:
                    break
                
                # Try to reconstruct row with tighter bound
                old_row = rows[r_idx][:]
                old_max = old_row[-1]
                
                # Remove row's diffs from used
                row_diffs_set = set(get_diffs(old_row))
                for d in row_diffs_set:
                    used_diffs.discard(d)
                
                # Re-greedy with current max_val as upper bound
                new_row = [0]
                x = 0
                while len(new_row) < row_len:
                    x += 1
                    if x > max_val:
                        break
                    new_diffs = {x - a for a in new_row}
                    if not new_diffs & used_diffs:
                        new_row.append(x)
                        used_diffs |= new_diffs
                
                if len(new_row) == row_len:
                    rows[r_idx] = new_row
                    if new_row[-1] < max_val:
                        max_val = max(max(r) for r in rows)
                else:
                    # Rollback
                    for d in set(get_diffs(new_row)):
                        used_diffs.discard(d)
                    used_diffs |= row_diffs_set
        
        return rows, max(max(r) for r in rows)

    # Main Loop
    while time.time() < deadline - 3.0:
        # Try row reconstruction search first (more effective for packing)
        rc_time = min(10.0, deadline - time.time() - 2.0)
        if rc_time > 1.0:
            rc_rows, rc_max = row_reconstruct_search(best_rows, best_max, rc_time)
            
            # Verify
            all_diffs = []
            for r in rc_rows:
                for d in get_diffs(r):
                    all_diffs.append(d)
            if len(all_diffs) == len(set(all_diffs)):
                if rc_max < best_max:
                    best_rows = rc_rows
                    best_max = rc_max
        
        # Polish with SA
        sa_time = min(5.0, deadline - time.time() - 1.0)
        if sa_time > 0.5:
            init_rows = [r[:] for r in best_rows]
            # Small perturbation
            for _ in range(2):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                row = init_rows[r_idx]
                lower = row[i_idx - 1] + 1
                upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 5
                if lower <= upper:
                    init_rows[r_idx][i_idx] = random.randint(lower, upper)
            
            local_rows, local_max, local_cost = sa_optimize(init_rows, sa_time, best_max)
            
            if local_cost < 10000:
                if local_max < best_max:
                    best_rows = local_rows
                    best_max = local_max
    
    # Final verification
    all_diffs = []
    for r in best_rows:
        for d in get_diffs(r):
            all_diffs.append(d)
    if len(all_diffs) != len(set(all_diffs)):
        # Fallback to greedy if something went wrong
        best_rows = greedy_construct()
        best_max = max(max(r) for r in best_rows)
    
    return best_rows
