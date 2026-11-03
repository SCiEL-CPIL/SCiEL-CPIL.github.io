import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 118.0
    best_max = float('inf')
    best_rows = None

    # --- Helper: get all pairwise differences of a row ---
    def get_diffs(row):
        diffs = []
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                diffs.append(row[j] - row[i])
        return diffs

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

    # --- Row reconstruction with randomization ---
    def reconstruct_row_random(r_idx, rows, used_diffs, upper_bound, randomize=False):
        """Re-greedy row r_idx with optional randomization."""
        current_row = rows[r_idx]
        current_diffs = set(get_diffs(current_row))
        for d in current_diffs:
            used_diffs.discard(d)
        
        new_row = [0]
        x = 0
        while len(new_row) < row_len:
            x += 1
            if x > upper_bound:
                break
            
            if randomize and len(new_row) < row_len - 1 and random.random() < 0.15:
                # Skip some candidates to explore different structures
                x += random.randint(1, 3)
                continue
            
            new_diffs = {x - a for a in new_row}
            if not new_diffs & used_diffs:
                new_row.append(x)
                used_diffs |= new_diffs
        
        if len(new_row) == row_len:
            rows[r_idx] = new_row
            return True
        else:
            for d in set(get_diffs(new_row)):
                used_diffs.discard(d)
            used_diffs |= current_diffs
            return False

    # --- Improved row reconstruction search ---
    def row_reconstruct_search(rows, max_val, time_limit):
        rows = [r[:] for r in rows]
        max_val = max(max(r) for r in rows)
        start_time = time.time()
        
        used_diffs = set()
        for r in rows:
            for d in get_diffs(r):
                used_diffs.add(d)
        
        # Try multiple passes with different randomization levels
        for pass_num in range(100):
            if time.time() - start_time >= time_limit:
                break
            
            # Alternate between deterministic and randomized reconstruction
            randomize = (pass_num % 3 == 2)
            
            for r_idx in range(n_rows):
                if time.time() - start_time >= time_limit:
                    break
                
                old_row = rows[r_idx][:]
                old_max = old_row[-1]
                
                row_diffs_set = set(get_diffs(old_row))
                for d in row_diffs_set:
                    used_diffs.discard(d)
                
                # Use slightly tighter bound on later passes
                effective_bound = max_val if pass_num < 20 else max_val - 1
                
                if reconstruct_row_random(r_idx, rows, used_diffs, effective_bound, randomize):
                    new_max = rows[r_idx][-1]
                    if new_max < max_val:
                        max_val = max(max(r) for r in rows)
                else:
                    for d in set(get_diffs(rows[r_idx])):
                        used_diffs.discard(d)
                    used_diffs |= row_diffs_set
        
        return rows, max(max(r) for r in rows)

    # --- SA with improved cost function and move diversity ---
    def sa_optimize(initial_rows, time_limit, bound):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        row_diffs = [get_diffs(r) for r in rows]
        
        diff_counts = defaultdict(int)
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] += 1
        
        def compute_cost():
            collisions = sum(c - 1 for c in diff_counts.values() if c > 1)
            max_val = max(r[-1] for r in rows)
            # Use a smoother objective: penalize collisions heavily, but also consider spread
            return collisions * 5000 + max_val + sum(r[-1] for r in rows) / 10

        current_cost = compute_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        T = 100.0
        cooling = 0.9998
        start_time = time.time()
        step = 0

        while time.time() - start_time < time_limit:
            step += 1
            r_idx = random.randint(0, n - 1)
            
            # Diversify moves: sometimes change multiple entries or use different strategies
            move_type = random.random()
            
            if move_type < 0.7:
                # Single entry move
                i_idx = random.randint(1, L - 1)
                row = rows[r_idx]
                old_val = row[i_idx]
                
                lower = row[i_idx - 1] + 1
                upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 15
                upper = min(upper, bound + 25)
                
                if lower > upper:
                    continue
                    
                # Biased sampling: prefer values near current or boundary
                if random.random() < 0.3:
                    new_val = random.choice([lower, upper, (lower + upper) // 2])
                else:
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
                        
            elif move_type < 0.9:
                # Swap two entries in the same row
                if L >= 3:
                    i1, i2 = random.sample(range(1, L), 2)
                    row = rows[r_idx]
                    if i1 > i2:
                        i1, i2 = i2, i1
                    # Ensure we don't violate ordering
                    if row[i1 - 1] + 1 < row[i2] and row[i1] < row[i2 + 1] - 1 if i2 < L - 1 else True:
                        old_val1, old_val2 = row[i1], row[i2]
                        # Try swapping if valid
                        if old_val1 + 1 < old_val2 and old_val2 - 1 > old_val1:
                            row[i1], row[i2] = old_val2, old_val1
                            # Check ordering
                            valid = True
                            for j in range(L - 1):
                                if row[j] >= row[j + 1]:
                                    valid = False
                                    break
                            if valid:
                                old_diffs = row_diffs[r_idx]
                                for d in old_diffs:
                                    diff_counts[d] -= 1
                                    if diff_counts[d] == 0:
                                        del diff_counts[d]
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
                                    if delta < 1000 and random.random() < math.exp(-delta / T):
                                        current_cost = new_cost
                                    else:
                                        row[i1], row[i2] = old_val1, old_val2
                                        for d in new_diffs:
                                            diff_counts[d] -= 1
                                            if diff_counts[d] == 0:
                                                del diff_counts[d]
                                        for d in old_diffs:
                                            diff_counts[d] = diff_counts.get(d, 0) + 1
                                        row_diffs[r_idx] = old_diffs
            else:
                # Reconstruct entire row using greedy
                old_row = rows[r_idx][:]
                old_diffs = row_diffs[r_idx]
                for d in old_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                new_row = [0]
                x = 0
                while len(new_row) < row_len:
                    x += 1
                    if x > bound + 10:
                        break
                    new_diffs = {x - a for a in new_row}
                    # Check against all other rows' diffs
                    conflict = False
                    for d in new_diffs:
                        if diff_counts.get(d, 0) > 0:
                            conflict = True
                            break
                    if not conflict:
                        new_row.append(x)
                
                if len(new_row) == row_len:
                    rows[r_idx] = new_row
                    row_diffs[r_idx] = get_diffs(new_row)
                    for d in row_diffs[r_idx]:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    new_cost = compute_cost()
                    if new_cost < current_cost:
                        current_cost = new_cost
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_rows = [r[:] for r in rows]
                            best_local_max = max(r[-1] for r in best_local_rows)
                    else:
                        # Rollback
                        for d in row_diffs[r_idx]:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        rows[r_idx] = old_row
                        row_diffs[r_idx] = old_diffs
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                else:
                    # Rollback
                    for d in set(get_diffs(new_row)):
                        if d in diff_counts:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                    rows[r_idx] = old_row
                    row_diffs[r_idx] = old_diffs
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
            
            T *= cooling
            if T < 0.5:
                T = 0.5

        return best_local_rows, best_local_max, best_local_cost

    # Main Loop
    while time.time() < deadline - 3.0:
        # Try row reconstruction search first
        rc_time = min(8.0, deadline - time.time() - 2.0)
        if rc_time > 1.0:
            rc_rows, rc_max = row_reconstruct_search(best_rows, best_max, rc_time)
            
            all_diffs = []
            for r in rc_rows:
                for d in get_diffs(r):
                    all_diffs.append(d)
            if len(all_diffs) == len(set(all_diffs)):
                if rc_max < best_max:
                    best_rows = rc_rows
                    best_max = rc_max
        
        # Polish with SA
        sa_time = min(6.0, deadline - time.time() - 1.0)
        if sa_time > 0.5:
            init_rows = [r[:] for r in best_rows]
            # More diverse perturbation
            for _ in range(random.randint(1, 3)):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                row = init_rows[r_idx]
                lower = row[i_idx - 1] + 1
                upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 10
                if lower <= upper:
                    init_rows[r_idx][i_idx] = random.randint(lower, upper)
            
            local_rows, local_max, local_cost = sa_optimize(init_rows, sa_time, best_max)
            
            if local_cost < 5000:  # No collisions
                if local_max < best_max:
                    best_rows = local_rows
                    best_max = local_max
    
    # Final verification
    all_diffs = []
    for r in best_rows:
        for d in get_diffs(r):
            all_diffs.append(d)
    if len(all_diffs) != len(set(all_diffs)):
        best_rows = greedy_construct()
        best_max = max(max(r) for r in best_rows)
    
    return best_rows
