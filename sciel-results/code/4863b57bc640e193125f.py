import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 112.0  # Safety margin for 120s limit
    
    n = k + 1
    diffs_per_row = n * (n - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def row_diffs(row):
        return [row[j] - row[i] for i in range(n) for j in range(i + 1, n)]
    
    def row_diff_set(row):
        return set(row_diffs(row))
    
    def validate(rows):
        all_diffs = set()
        for r in rows:
            if len(r) != n or r[0] != 0:
                return False
            for i in range(1, n):
                if r[i] <= r[i-1]:
                    return False
            dset = row_diff_set(r)
            if len(dset) != diffs_per_row:
                return False
            if dset & all_diffs:
                return False
            all_diffs.update(dset)
        return True
    
    def scope(rows):
        return max(max(r) for r in rows)
    
    # Phase 1: Greedy construction with randomized row ordering
    def greedy_construct(seed, max_scope=150):
        rng = random.Random(seed)
        used = set()
        rows = []
        # Randomize the order in which we build rows
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        row_slots = [None] * n_rows
        
        for row_idx in row_order:
            row = [0]
            while len(row) < n:
                found = False
                # Try values with some randomization
                candidates = list(range(row[-1] + 1, max_scope + 1))
                rng.shuffle(candidates)
                # But prefer smaller values - sort by value with some randomness
                candidates.sort(key=lambda x: x + rng.random() * 5)
                
                for x in candidates:
                    new_diffs = set(x - a for a in row)
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            row_slots[row_idx] = row
        
        return row_slots
    
    # Get initial solution
    initial_rows = None
    for s in range(200):
        initial_rows = greedy_construct(s, max_scope=140)
        if initial_rows:
            break
    
    if not initial_rows:
        # Guaranteed fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n:
                x = row[-1] + 1
                while True:
                    new_diffs = set(x - a for a in row)
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        break
                    x += 1
                row.append(x)
                used.update(new_diffs)
            rows.append(row)
        initial_rows = rows
    
    best_rows = [list(r) for r in initial_rows]
    best_scope = scope(best_rows)
    current_rows = [list(r) for r in initial_rows]
    current_scope = best_scope
    
    # Precompute difference sets
    def compute_diff_sets(rows):
        return [row_diff_set(r) for r in rows]
    
    current_diff_sets = compute_diff_sets(current_rows)
    used_diffs = set()
    for ds in current_diff_sets:
        used_diffs.update(ds)
    
    # Phase 2: Local search - try to reduce the maximum element
    # Strategy: For each row, try to find a smaller valid configuration
    # by reducing elements from the end
    
    def try_reduce_row(row, row_idx, used_others, rng):
        """Try to reduce the max element of a row while keeping validity."""
        if len(row) != n:
            return None
        
        best_local = list(row)
        best_local_scope = row[-1]
        
        # Try reducing the last element
        for new_last in range(row[-1] - 1, row[-2], -1):
            test_row = list(row)
            test_row[-1] = new_last
            # Check strictly increasing
            if test_row[-1] <= test_row[-2]:
                break
            dset = row_diff_set(test_row)
            if len(dset) != diffs_per_row:
                continue
            if dset & used_others:
                continue
            # Valid!
            best_local = test_row
            best_local_scope = new_last
            break
        
        # Try reducing second-to-last, then last
        for new_s2 in range(row[-2] - 1, row[-3] if n >= 3 else 0, -1):
            for new_last in range(best_local[-1] - 1, new_s2, -1):
                test_row = list(row)
                test_row[-2] = new_s2
                test_row[-1] = new_last
                if test_row[-2] <= test_row[-3] if n >= 3 else test_row[-2] <= 0:
                    continue
                if test_row[-1] <= test_row[-2]:
                    continue
                dset = row_diff_set(test_row)
                if len(dset) != diffs_per_row:
                    continue
                if dset & used_others:
                    continue
                if new_last < best_local_scope:
                    best_local = test_row
                    best_local_scope = new_last
                break
        
        return best_local
    
    def full_local_search(rows, diff_sets, used, rng, time_budget=5.0):
        """Iteratively try to reduce each row."""
        rows = [list(r) for r in rows]
        diff_sets = [row_diff_set(r) for r in rows]
        used = set()
        for ds in diff_sets:
            used.update(ds)
        
        improved = True
        while improved and time.time() - start_time < time_limit - time_budget:
            improved = False
            # Try rows in order of their max element (largest first)
            order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
            for idx in order:
                used_others = used - diff_sets[idx]
                new_row = try_reduce_row(rows[idx], idx, used_others, rng)
                if new_row and new_row[-1] < rows[idx][-1]:
                    # Update
                    used.difference_update(diff_sets[idx])
                    rows[idx] = new_row
                    diff_sets[idx] = row_diff_set(new_row)
                    used.update(diff_sets[idx])
                    improved = True
        
        return rows, diff_sets, used
    
    # Phase 3: Simulated annealing with multi-move steps
    def sa_search(rows, diff_sets, used, rng, time_budget=10.0):
        """Simulated annealing to reduce scope."""
        rows = [list(r) for r in rows]
        diff_sets = [row_diff_set(r) for r in rows]
        used = set()
        for ds in diff_sets:
            used.update(ds)
        
        T = 5.0
        cooling = 0.997
        min_T = 0.05
        current_scope_val = scope(rows)
        best_local = [list(r) for r in rows]
        best_local_scope = current_scope_val
        
        deadline = time.time() + time_budget
        
        while time.time() < deadline and T > min_T:
            # Pick a row to modify
            # Bias toward rows with larger max
            weights = [rows[i][-1] + 1 for i in range(n_rows)]
            total_w = sum(weights)
            r = rng.random() * total_w
            acc = 0
            chosen = 0
            for i, w in enumerate(weights):
                acc += w
                if r <= acc:
                    chosen = i
                    break
            
            # Pick an index to change (1 to n-1)
            idx = rng.randint(1, n - 1)
            old_val = rows[chosen][idx]
            
            # Generate candidates
            lo = rows[chosen][idx - 1] + 1
            hi = old_val + 8 if idx < n - 1 else old_val + 3
            candidates = list(range(max(lo, old_val - 10), hi + 1))
            rng.shuffle(candidates)
            
            move_made = False
            for nv in candidates:
                if nv <= rows[chosen][idx - 1]:
                    continue
                if idx < n - 1 and nv >= rows[chosen][idx + 1]:
                    continue
                
                new_row = list(rows[chosen])
                new_row[idx] = nv
                dset = row_diff_set(new_row)
                if len(dset) != diffs_per_row:
                    continue
                
                other_diffs = used - diff_sets[chosen]
                if dset & other_diffs:
                    continue
                
                # Valid move
                new_scope_val = max(max(r) for r in rows[:chosen] + [new_row] + rows[chosen+1:])
                delta = new_scope_val - current_scope_val
                
                if delta < 0 or rng.random() < pow(2.71828, -delta / T):
                    # Accept
                    used.difference_update(diff_sets[chosen])
                    rows[chosen] = new_row
                    diff_sets[chosen] = dset
                    used.update(dset)
                    current_scope_val = new_scope_val
                    
                    if current_scope_val < best_local_scope:
                        best_local_scope = current_scope_val
                        best_local = [list(r) for r in rows]
                    
                    move_made = True
                    break
            
            T *= cooling
        
        return best_local, best_local_scope
    
    # Run local search
    best_rows, _, _ = full_local_search(current_rows, current_diff_sets, used_diffs, random.Random(42), time_budget=3.0)
    best_scope = scope(best_rows)
    
    # Run SA with multiple restarts
    deadline = start_time + time_limit - 2.0
    restart_count = 0
    while time.time() < deadline and best_scope > 105:
        restart_count += 1
        seed = 1000 + restart_count * 7
        rng = random.Random(seed)
        
        # Start from best_rows
        sa_rows, sa_scope = sa_search(best_rows, compute_diff_sets(best_rows), 
                                       set().union(*[row_diff_set(r) for r in best_rows]),
                                       rng, time_budget=min(8.0, max(2.0, deadline - time.time())))
        
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
        
        # Also try from a fresh greedy solution
        fresh = greedy_construct(seed, max_scope=max(best_scope + 10, 120))
        if fresh:
            fresh_scope = scope(fresh)
            if fresh_scope < best_scope:
                best_rows = fresh
                best_scope = fresh_scope
                # Polish it
                best_rows, _, _ = full_local_search(best_rows, compute_diff_sets(best_rows),
                                                     set().union(*[row_diff_set(r) for r in best_rows]),
                                                     rng, time_budget=2.0)
                best_scope = scope(best_rows)
        
        if best_scope <= 105:
            break
    
    # Final validation
    if not validate(best_rows):
        # Fallback to initial
        best_rows = [list(r) for r in initial_rows]
    
    return best_rows


if __name__ == "__main__":
    rows = construct(7, 5)
    print(f"Scope: {max(max(r) for r in rows)}")
    print(f"Valid: {validate(rows)}")
    for r in rows:
        print(r)
