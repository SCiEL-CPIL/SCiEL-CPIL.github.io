import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    
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

    def build_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_collisions(counts):
        coll = 0
        for d, c in counts.items():
            if c > 1:
                coll += c - 1
        return coll

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.append(r[j] - r[i])
        return len(set(all_diffs)) == len(all_diffs)

    def scope_of(rows):
        return max(r[-1] for r in rows)

    # Start with greedy
    best_rows = greedy_construct()
    best_max = scope_of(best_rows)
    best_cost = best_max  # no collisions assumed for greedy
    
    def simulate_annealing(init_rows, time_limit):
        nonlocal best_rows, best_max, best_cost
        
        rows = [r[:] for r in init_rows]
        current_max = scope_of(rows)
        counts = build_diff_counts(rows)
        current_collisions = get_collisions(counts)
        current_cost = current_collisions * 100000 + current_max
        
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max
        
        T = 100.0
        cooling = 0.99995
        start_time = time.time()
        steps = 0
        rng = random.Random(int(start_time * 1000))
        
        while time.time() - start_time < time_limit:
            # Pick a random row and position to move
            r_idx = rng.randint(0, n_rows - 1)
            i_idx = rng.randint(1, row_len - 1)
            
            lower = rows[r_idx][i_idx - 1] + 1
            # Upper bound: allow some expansion but not too much
            upper_limit = best_max + 20
            if i_idx < row_len - 1:
                upper = rows[r_idx][i_idx + 1] - 1
                upper = min(upper, upper_limit)
            else:
                upper = upper_limit
            
            if lower > upper:
                continue
            
            new_val = rng.randint(lower, upper)
            if new_val == rows[r_idx][i_idx]:
                continue
            
            old_val = rows[r_idx][i_idx]
            
            # Check strict increasing
            if new_val <= rows[r_idx][i_idx - 1]:
                continue
            if i_idx < row_len - 1 and new_val >= rows[r_idx][i_idx + 1]:
                continue
            
            # Compute delta in collisions
            delta_coll = 0
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
            
            # Apply move
            rows[r_idx][i_idx] = new_val
            new_max = scope_of(rows)
            new_collisions = current_collisions + delta_coll
            new_cost = new_collisions * 100000 + new_max
            
            if new_cost < current_cost:
                current_cost = new_cost
                current_collisions = new_collisions
                current_max = new_max
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_max = new_max
                    best_local_rows = [r[:] for r in rows]
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if rng.random() < 0.3:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        # Revert
                        rows[r_idx][i_idx] = old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            old_d = abs(old_val - rows[r_idx][j])
                            new_d = abs(new_val - rows[r_idx][j])
                            if old_d != new_d:
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
                else:
                    if delta < 10000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if rng.random() < p:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        # Revert
                        rows[r_idx][i_idx] = old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            old_d = abs(old_val - rows[r_idx][j])
                            new_d = abs(new_val - rows[r_idx][j])
                            if old_d != new_d:
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
            if T < 0.1:
                T = 0.1
            steps += 1
            
            # Periodic consistency check
            if steps % 2000 == 0:
                actual_coll = get_collisions(counts)
                if actual_coll != current_collisions:
                    counts = build_diff_counts(rows)
                    current_collisions = get_collisions(counts)
                    current_max = scope_of(rows)
                    current_cost = current_collisions * 100000 + current_max
        
        # Update global best if improved
        if best_local_cost < best_cost:
            best_cost = best_local_cost
            best_max = best_local_max
            best_rows = best_local_rows
        
        return best_local_rows, best_local_max, best_local_cost

    # Main SA loop with restarts
    sa_start = time.time()
    while time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining <= 0:
            break
        
        # Use best_rows as init if valid, else random
        if is_valid(best_rows):
            init_rows = best_rows
        else:
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, best_max + 1), k)
                pts.sort()
                init_rows.append([0] + pts)
        
        sa_time = min(remaining, 6.0)
        if sa_time <= 0:
            break
        
        local_rows, local_max, local_cost = simulate_annealing(init_rows, sa_time)
        
        # If we found a valid solution, update
        if local_cost < 100000:
            if local_max < best_max:
                best_max = local_max
                best_rows = local_rows
                best_cost = local_max

    return best_rows
