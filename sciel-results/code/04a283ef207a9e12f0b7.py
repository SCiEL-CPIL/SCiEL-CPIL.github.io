"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Build all differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Check if a row is valid (increasing, starts at 0)
    def valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                return False
        return True
    
    # Full validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if not valid_row(row):
                return False
            rd = row_diffs(row)
            if rd & all_diffs:
                return False
            all_diffs |= rd
        return True
    
    # Greedy baseline
    def greedy_solution(seed=None):
        if seed is not None:
            random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    # Simulated Annealing approach
    def simulated_annealing(initial_rows, deadline_t, max_iters=50000):
        """
        SA over row configurations. State: list of rows.
        Move: pick a random row, perturb one of its elements (except first=0).
        Cost: max entry + penalty for conflicts.
        """
        rows = [list(r) for r in initial_rows]
        current_max = get_max_entry(rows)
        current_cost = current_max
        
        # Track used diffs as a set for quick conflict checking
        def compute_used_diffs(rows):
            used = set()
            for row in rows:
                used |= row_diffs(row)
            return used
        
        used_diffs = compute_used_diffs(rows)
        
        # Count conflicts
        def count_conflicts(used):
            # A diff is conflicting if it appears in multiple rows
            # We need to track per-row diffs
            diff_count = {}
            for row in rows:
                for d in row_diffs(row):
                    diff_count[d] = diff_count.get(d, 0) + 1
            conflicts = sum(1 for d, c in diff_count.items() if c > 1)
            return conflicts
        
        # Better: track cost as max_entry * (1 + 100*conflicts)
        def compute_cost(rows):
            max_e = get_max_entry(rows)
            # Count duplicate diffs
            all_d = []
            for row in rows:
                all_d.extend(row_diffs(row))
            from collections import Counter
            cnt = Counter(all_d)
            dup = sum(1 for d, c in cnt.items() if c > 1)
            return max_e + 10000 * dup, max_e, dup
        
        current_cost, current_max, current_dup = compute_cost(rows)
        best_rows_sa = [list(r) for r in rows]
        best_cost_sa = current_cost
        best_max_sa = current_max
        
        T = max(10.0, current_max * 0.1)
        T_min = 0.1
        cooling = 0.9999
        
        it = 0
        while time.time() < deadline_t and it < max_iters:
            it += 1
            
            # Pick random row to perturb
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            
            # Pick a random position (1 to k) to change
            pos = random.randint(1, k)
            old_val = row[pos]
            
            # Generate new value: try to reduce max or explore
            if random.random() < 0.5:
                # Try to reduce
                new_val = max(1, old_val - random.randint(1, max(1, old_val // 3)))
            else:
                # Explore
                new_val = old_val + random.randint(-old_val // 2, old_val // 2)
                new_val = max(1, new_val)
            
            # Ensure row stays increasing
            valid = True
            if pos > 0 and new_val <= row[pos - 1]:
                valid = False
            if pos < k and new_val >= row[pos + 1]:
                valid = False
            
            if not valid:
                continue
            
            # Compute old row diffs
            old_row_diffs = row_diffs(row)
            
            # Try the new value
            row[pos] = new_val
            new_row_diffs = row_diffs(row)
            
            # Check if row is still valid (increasing)
            if not valid_row(row):
                row[pos] = old_val
                continue
            
            # Compute new cost
            new_max = get_max_entry(rows)
            
            # Count duplicates efficiently
            # Remove old diffs, add new diffs
            # This is expensive; let's just compute full cost
            new_cost, new_max_val, new_dup = compute_cost(rows)
            
            delta_cost = new_cost - current_cost
            
            # Acceptance
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                current_max = new_max_val
                if new_max_val < best_max_sa:
                    best_max_sa = new_max_val
                    best_rows_sa = [list(r) for r in rows]
            else:
                row[pos] = old_val
            
            T = max(T_min, T * cooling)
        
        return best_rows_sa, best_max_sa
    
    # Phase 1: Multiple SA runs with different initial conditions
    search_deadline = deadline - 5.0
    
    # Try SA from greedy solution
    if time.time() < search_deadline:
        sa_rows, sa_max = simulated_annealing(best_rows, search_deadline, max_iters=30000)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Phase 2: Try SA from different greedy seeds
    if time.time() < search_deadline - 10:
        for seed in [42, 123, 999, 7777, 31415]:
            if time.time() > search_deadline - 5:
                break
            g_rows = greedy_solution(seed=seed)
            g_max = get_max_entry(g_rows)
            
            if g_max < best_max:
                best_max = g_max
                best_rows = g_rows
            
            # Run SA on this
            remaining = search_deadline - 5 - time.time()
            if remaining > 5:
                sa_rows, sa_max = simulated_annealing(g_rows, time.time() + min(remaining, 20), max_iters=20000)
                if sa_max < best_max:
                    best_max = sa_max
                    best_rows = sa_rows
    
    # Phase 3: Backtracking for small M values if time allows
    def find_solution_backtrack(M, deadline_t):
        """Backtracking to find a solution with max entry <= M"""
        
        def solve_row(idx, global_used):
            if time.time() > deadline_t:
                raise TimeoutError()
            
            if idx == n_rows:
                return []
            
            def build(pos, vals, row_used):
                if time.time() > deadline_t:
                    raise TimeoutError()
                
                if pos == k + 1:
                    if row_used & global_used:
                        return None
                    res = solve_row(idx + 1, global_used | row_used)
                    if res is None:
                        return None
                    return [vals] + res
                
                last = vals[-1]
                candidates = list(range(last + 1, M + 1))
                random.shuffle(candidates)
                
                for v in candidates:
                    new_diffs = 0
                    conflict = False
                    for u in vals:
                        d = v - u
                        if (row_used >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    
                    if conflict:
                        continue
                    if new_diffs & global_used:
                        continue
                    
                    vals.append(v)
                    res = build(pos + 1, vals, row_used | new_diffs)
                    if res is not None:
                        return res
                    vals.pop()
                
                return None
            
            return build(1, [0], 0)
        
        return solve_row(0, 0)
    
    if time.time() < search_deadline - 15:
        # Try backtracking with M = best_max - 1, best_max - 2, etc.
        for M in range(max(lower_bound, best_max - 1), max(lower_bound - 1, best_max - 6), -1):
            if time.time() > search_deadline - 10:
                break
            time_slice = min(8.0, search_deadline - 10 - time.time())
            if time_slice < 1:
                break
            try:
                sol = find_solution_backtrack(M, time.time() + time_slice)
                if sol:
                    curr_max = get_max_entry(sol)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_rows = sol
                        break
            except TimeoutError:
                pass
            except Exception:
                pass
    
    # Final validation
    if not validate(best_rows):
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
