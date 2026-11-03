import time
import random
import math
import sys

def construct(n_rows, k):
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_all = n_rows * total_diffs_per_row

    deadline = time.time() + 118.0

    # Greedy construction for initial baseline
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
    best_max = max(r[-1] for r in best_rows)

    # Helper to compute all differences for a set of rows
    def get_all_diffs(rows):
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        return diffs

    def is_valid(rows):
        diffs = get_all_diffs(rows)
        return len(set(diffs)) == len(diffs)

    # Simulated Annealing with better move generation and cost function
    def simulate_annealing(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        current_max = max(r[-1] for r in rows)
        
        # Build diff count map
        def build_counts():
            counts = {}
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        counts[d] = counts.get(d, 0) + 1
            return counts

        counts = build_counts()
        
        def get_cost():
            collisions = sum(c - 1 for c in counts.values() if c > 1)
            return collisions * 10000 + current_max

        current_cost = get_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max
        
        T = 100.0
        cooling = 0.99995
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Random move: pick a random row and a random position to move
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            old_val = rows[r_idx][i_idx]
            
            # Determine valid range for new value
            lower = rows[r_idx][i_idx - 1] + 1
            upper = rows[r_idx][i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 50
            
            if lower > upper:
                continue
            
            # Try a random new value in range
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
            
            # Compute delta cost
            delta_coll = 0
            
            # Remove old differences involving position i_idx
            for j in range(row_len):
                if j == i_idx:
                    continue
                old_d = abs(old_val - rows[r_idx][j])
                new_d = abs(new_val - rows[r_idx][j])
                
                if old_d != new_d:
                    # Remove old_d
                    oc = counts.get(old_d, 0)
                    if oc > 0:
                        if oc == 1:
                            del counts[old_d]
                        else:
                            counts[old_d] = oc - 1
                            delta_coll -= 1
                    
                    # Add new_d
                    nc = counts.get(new_d, 0)
                    if nc == 0:
                        counts[new_d] = 1
                    else:
                        counts[new_d] = nc + 1
                        delta_coll += 1
            
            new_max = max(r[-1] for r in rows)
            new_collisions = sum(c - 1 for c in counts.values() if c > 1)
            new_cost = new_collisions * 10000 + new_max
            
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept the move
                rows[r_idx][i_idx] = new_val
                current_cost = new_cost
                current_max = new_max
                
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_max = new_max
                    best_local_rows = [r[:] for r in rows]
            else:
                # Reject the move: revert counts
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    old_d = abs(old_val - rows[r_idx][j])
                    new_d = abs(new_val - rows[r_idx][j])
                    
                    if old_d != new_d:
                        # Revert: remove new_d, add old_d
                        nc = counts.get(new_d, 0)
                        if nc == 1:
                            del counts[new_d]
                        else:
                            counts[new_d] = nc - 1
                        
                        oc = counts.get(old_d, 0)
                        if oc == 0:
                            counts[old_d] = 1
                        else:
                            counts[old_d] = oc + 1
            
            T *= cooling
            if T < 1.0:
                T = 1.0
        
        return best_local_rows, best_local_max, best_local_cost

    # Local search: try to reduce the maximum by moving the largest element
    def local_search_reduce_max(rows, time_limit):
        rows = [r[:] for r in rows]
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Find the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            max_val = rows[max_row_idx][-1]
            
            # Try to move the last element to a smaller value
            improved = False
            for new_val in range(max_val - 1, rows[max_row_idx][-2], -1):
                rows[max_row_idx][-1] = new_val
                if is_valid(rows):
                    improved = True
                    break
                rows[max_row_idx][-1] = max_val
            
            if not improved:
                # Try moving other elements in the max row
                for pos in range(1, row_len - 1):
                    old_val = rows[max_row_idx][pos]
                    lower = rows[max_row_idx][pos - 1] + 1
                    upper = rows[max_row_idx][pos + 1] - 1
                    
                    for new_val in range(lower, upper + 1):
                        if new_val == old_val:
                            continue
                        rows[max_row_idx][pos] = new_val
                        if is_valid(rows):
                            # Check if this reduces the max
                            if rows[max_row_idx][-1] < max_val:
                                improved = True
                                break
                        rows[max_row_idx][pos] = old_val
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        return rows

    # Main search loop
    while time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining <= 0:
            break
        
        # Use best valid rows as starting point
        if is_valid(best_rows):
            init_rows = best_rows
        else:
            init_rows = baseline_rows
        
        # Run SA with a portion of remaining time
        sa_time = min(remaining * 0.6, 10.0)
        if sa_time < 1.0:
            break
        
        local_rows, local_max, local_cost = simulate_annealing(init_rows, sa_time)
        
        # Check if we found a valid solution
        if local_cost < 10000:  # No collisions
            if is_valid(local_rows):
                if local_max < best_max:
                    best_max = local_max
                    best_rows = local_rows
                    # Try local search to further reduce
                    if time.time() < deadline - 10.0:
                        ls_rows = local_search_reduce_max(best_rows, min(5.0, deadline - time.time() - 10.0))
                        if is_valid(ls_rows):
                            ls_max = max(r[-1] for r in ls_rows)
                            if ls_max < best_max:
                                best_max = ls_max
                                best_rows = ls_rows
        
        # Also try a direct greedy approach with different randomization
        if time.time() < deadline - 10.0:
            # Randomized greedy
            used = set()
            rand_rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < row_len:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                    else:
                        # Try skipping ahead
                        x += random.randint(1, 5)
                if len(row) < row_len:
                    success = False
                    break
                rand_rows.append(row)
            
            if success:
                rand_max = max(r[-1] for r in rand_rows)
                if rand_max < best_max:
                    best_max = rand_max
                    best_rows = rand_rows

    return best_rows
