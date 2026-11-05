"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2

    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    # Helper to compute all differences in a row as a set
    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Validate a complete solution
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # Greedy baseline with multiple seeds
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

    # Try multiple greedy seeds and keep best
    best_rows = None
    best_max = float('inf')
    for seed in range(20):
        rows = greedy_solution(seed)
        mx = get_max_entry(rows)
        if mx < best_max:
            best_max = mx
            best_rows = rows

    # Constraint-based backtracking: try to build rows with smaller max
    def try_build_rows(max_allowed, seed=None):
        """Try to build n_rows rows with max entry <= max_allowed using backtracking."""
        if seed is not None:
            random.seed(seed)
        used = 0  # bitset for used differences
        rows = []
        
        def backtrack(row_idx, current_row):
            if row_idx == n_rows:
                return True
            if time.time() > deadline:
                return False
            
            # Try to build row row_idx
            # We need to pick k more values after 0, all <= max_allowed
            # Use a different strategy: try to place values greedily with backtracking
            
            # For simplicity, use a recursive approach with pruning
            def build_row(pos, current, used_diffs):
                if pos == k + 1:
                    return current
                if time.time() > deadline:
                    return None
                
                last = current[-1]
                start_val = last + 1
                end_val = max_allowed
                
                # Try values in increasing order
                for val in range(start_val, end_val + 1):
                    # Check if val creates conflicts
                    new_diffs = 0
                    conflict = False
                    for a in current:
                        d = val - a
                        if (used_diffs >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    
                    if conflict:
                        continue
                    
                    # Check internal consistency (differences within current row)
                    # Actually, we need to check that new differences don't conflict with each other
                    # But since we're adding one at a time and checking against used_diffs,
                    # we need to also check against the differences created by this row so far
                    # This is complex, so let's use a different approach
                    
                    new_current = current + [val]
                    # Check all differences in new_current
                    valid = True
                    seen_diffs = set()
                    for i in range(len(new_current)):
                        for j in range(i + 1, len(new_current)):
                            d = new_current[j] - new_current[i]
                            if d in seen_diffs:
                                valid = False
                                break
                            seen_diffs.add(d)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check against global used_diffs
                    for d in seen_diffs:
                        if (used_diffs >> d) & 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Recurse
                    result = build_row(pos + 1, new_current, used_diffs | new_diffs)
                    if result is not None:
                        return result
                return None
            
            result = build_row(1, [0], 0)
            if result is not None:
                # Update used_diffs with this row's differences
                for i in range(len(result)):
                    for j in range(i + 1, len(result)):
                        d = result[j] - result[i]
                        used |= (1 << d)
                rows.append(result)
                return True
            return False
        
        return backtrack(0, [])

    # Try to find a better solution using backtracking with decreasing max
    current_max = best_max
    for target_max in range(current_max - 1, max(current_max - 50, 0), -1):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            # We need a more efficient approach - the above backtracking is too slow
            # Instead, let's use a different strategy: local search with restarts
            found = False
            break
        
        if found:
            current_max = target_max
            break

    # Use a smarter local search: hill climbing with restarts
    def local_search_with_restarts(initial_rows, deadline_t, n_restarts=50):
        best_found = [list(r) for r in initial_rows]
        best_max_found = get_max_entry(best_found)
        
        for restart in range(n_restarts):
            if time.time() > deadline_t:
                break
            
            # Start from initial or perturbed version
            if restart == 0:
                rows = [list(r) for r in initial_rows]
            else:
                # Perturb: randomly modify one row
                rows = [list(r) for r in initial_rows]
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, len(rows[row_idx]) - 1)
                delta = random.randint(-5, 5)
                old_val = rows[row_idx][pos]
                new_val = max(1, old_val + delta)
                if pos > 0 and new_val <= rows[row_idx][pos - 1]:
                    new_val = rows[row_idx][pos - 1] + 1
                if pos < len(rows[row_idx]) - 1 and new_val >= rows[row_idx][pos + 1]:
                    new_val = rows[row_idx][pos + 1] - 1
                if new_val != old_val:
                    rows[row_idx][pos] = new_val
            
            # Hill climbing: try to reduce max by modifying individual elements
            improved = True
            local_best = [list(r) for r in rows]
            local_max = get_max_entry(local_best)
            
            while improved and time.time() < deadline_t:
                improved = False
                # Try each row and position
                for row_idx in range(n_rows):
                    if time.time() > deadline_t:
                        break
                    row = local_best[row_idx]
                    for pos in range(1, len(row)):
                        if time.time() > deadline_t:
                            break
                        old_val = row[pos]
                        # Try smaller values
                        for new_val in range(max(1, old_val - 3), old_val):
                            if pos > 0 and new_val <= row[pos - 1]:
                                continue
                            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                                continue
                            
                            # Try the change
                            row[pos] = new_val
                            
                            # Validate
                            valid = True
                            all_diffs = set()
                            for r in local_best:
                                if not all(r[i] < r[i+1] for i in range(len(r)-1)):
                                    valid = False
                                    break
                                for i in range(len(r)):
                                    for j in range(i + 1, len(r)):
                                        d = r[j] - r[i]
                                        if d in all_diffs:
                                            valid = False
                                            break
                                        all_diffs.add(d)
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                new_max = get_max_entry(local_best)
                                if new_max < local_max:
                                    local_max = new_max
                                    improved = True
                                else:
                                    row[pos] = old_val
                            else:
                                row[pos] = old_val
                        if improved:
                            break
                    if improved:
                        break
            
            # Update global best
            if local_max < best_max_found:
                best_max_found = local_max
                best_found = [list(r) for r in local_best]
        
        return best_found, best_max_found

    # Run local search
    ls_deadline = deadline - 5.0
    try:
        ls_rows, ls_max = local_search_with_restarts(best_rows, ls_deadline, n_restarts=100)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    except Exception:
        pass

    # Final validation
    if not validate(best_rows):
        best_rows = greedy_solution(0)
        best_max = get_max_entry(best_rows)

    return best_rows

# EVOLVE-BLOCK-END
