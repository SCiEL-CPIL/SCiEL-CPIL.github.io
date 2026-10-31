import time
import math
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Total number of distinct differences needed
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Theoretical lower bound on max entry:
    # If max entry is M, the set of possible differences is {1, 2, ..., M}
    # We need total_diffs_needed distinct differences, so M >= total_diffs_needed
    lower_bound_M = total_diffs_needed
    
    # Greedy initialization to get a valid starting point
    used_diffs = set()
    greedy_rows = []
    
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    used_diffs.update(x - a for a in row[:-1])
                    current_max = x
                    break
                x += 1
        greedy_rows.append(row)
    
    best_rows = [r[:] for r in greedy_rows]
    best_scope = max(max(r) for r in greedy_rows)
    
    if best_scope == lower_bound_M:
        return best_rows
    
    # Backtracking solver with forward checking
    # We search for a solution with max entry <= M
    # For each row, we need to choose k+1 elements: 0 = e0 < e1 < ... < ek <= M
    # All pairwise differences must be globally unique.
    
    def solve_backtracking(M, time_limit=10.0):
        """
        Try to find a valid solution with max entry <= M.
        Returns list of rows if found, None otherwise.
        """
        t_start = time.time()
        
        used_diffs = set()
        rows = []
        n_needed = k + 1  # elements per row
        
        # We'll build rows one by one
        # For each row, we backtrack on positions 1..k (position 0 is fixed at 0)
        
        def build_row(row_idx, current_row, used_diffs):
            """
            Recursively build row row_idx.
            current_row: list of elements chosen so far (starts with [0])
            Returns True if row completed successfully, False otherwise.
            """
            if time.time() - t_start > time_limit:
                return False
            
            if len(current_row) == n_needed:
                # Row complete
                rows.append(current_row[:])
                return True
            
            pos = len(current_row)
            lower = current_row[-1] + 1
            upper = M
            
            # Dynamic variable ordering: pick the position with fewest options first?
            # For now, use natural order (left to right)
            
            # Try values from lower to upper
            # Optimization: if we've tried many values and failed, we can prune
            for x in range(lower, upper + 1):
                # Check if x creates any new difference conflicts
                new_diffs = []
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Forward checking: for subsequent positions, check if there's
                # still at least one valid value available
                # This is expensive, so only do it for early positions
                if pos < 3:
                    # Light forward check
                    next_lower = x + 1
                    if next_lower > upper:
                        continue
                    # Just verify that next_lower could potentially work
                    # (this is a very weak check, skip for speed)
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if build_row(row_idx, current_row, used_diffs):
                    return True
                
                # Backtrack
                used_diffs.difference_update(new_diffs)
                current_row.pop()
            
            return False
        
        # Build all rows
        for i in range(n_rows):
            if not build_row(i, [0], used_diffs):
                return None
        
        return rows
    
    # First, try to find a solution at the lower bound
    # If that fails, try progressively larger M
    
    # But pure backtracking might be too slow. Let's combine with a smarter approach.
    
    # Alternative: Use a hybrid approach
    # 1. Start with greedy solution
    # 2. Try to reduce the scope using local search + backtracking repair
    
    # For now, let's try the backtracking approach for a few M values
    
    current_M = max(lower_bound_M, best_scope - 1)
    
    found = False
    while current_M >= lower_bound_M and time.time() < deadline - 10:
        # Try to solve with max entry = current_M
        # Use a time limit for each attempt
        time_limit = min(8.0, deadline - 5 - time.time())
        if time_limit < 1:
            break
        
        sol = solve_backtracking(current_M, time_limit)
        if sol is not None:
            # Verify
            used = set()
            valid = True
            for r in sol:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d in used:
                            valid = False
                            break
                        used.add(d)
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                scope = max(max(r) for r in sol)
                if scope < best_scope:
                    best_rows = sol
                    best_scope = scope
                found = True
                break
        
        current_M -= 1
    
    if found or best_scope == lower_bound_M:
        return best_rows
    
    # If backtracking didn't find a better solution, fall back to local search
    # to try to improve the greedy solution
    
    # Local search setup
    current_rows = [r[:] for r in greedy_rows]
    
    # Precompute difference counts
    M_max = max(max(r) for r in current_rows) + 20
    diff_counts = [0] * (M_max + 1)
    
    def update_diffs(row, sign):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= M_max:
                    diff_counts[d] += sign
    
    for r in current_rows:
        update_diffs(r, 1)
    
    num_dupes = sum(1 for c in diff_counts if c > 1)
    
    def apply_move(ri, ei, new_val):
        nonlocal num_dupes
        row = current_rows[ri]
        old_val = row[ei]
        
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= M_max:
                old_c = diff_counts[d]
                diff_counts[d] -= 1
                if old_c == 2 and diff_counts[d] == 1:
                    num_dupes -= 1
        
        current_rows[ri][ei] = new_val
        
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= M_max:
                old_c = diff_counts[d]
                diff_counts[d] += 1
                if old_c == 1 and diff_counts[d] == 2:
                    num_dupes += 1
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    def local_search(max_steps=100000):
        nonlocal num_dupes, best_scope, best_rows
        
        steps = 0
        T = 10.0
        alpha = 0.995
        
        while steps < max_steps and time.time() < deadline - 5:
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            
            row = current_rows[ri]
            old_val = row[ei]
            
            lower = row[ei - 1] + 1
            upper = row[ei + 1] - 1 if ei < k else best_scope + 10
            
            if num_dupes == 0:
                search_upper = min(upper, best_scope)
            else:
                search_upper = min(upper, best_scope + 20)
            
            if lower > search_upper:
                continue
            
            new_val = random.randint(lower, search_upper)
            
            if new_val == old_val:
                continue
            
            # Calculate delta dupes
            delta = 0
            for j in range(len(row)):
                if j == ei:
                    continue
                d_old = abs(row[j] - old_val)
                if d_old <= M_max:
                    c = diff_counts[d_old]
                    if c == 2:
                        delta -= 1
            
            for j in range(len(row)):
                if j == ei:
                    continue
                d_new = abs(row[j] - new_val)
                if d_new <= M_max:
                    c = diff_counts[d_new]
                    if c == 1:
                        delta += 1
            
            accepted = False
            if num_dupes > 0:
                if delta < 0:
                    accepted = True
                elif delta == 0:
                    if random.random() < 0.05:
                        accepted = True
            else:
                if new_val < old_val:
                    accepted = True
                elif new_val > old_val:
                    delta_scope = new_val - old_val
                    if random.random() < math.exp(-delta_scope / T):
                        accepted = True
            
            if accepted:
                apply_move(ri, ei, new_val)
                
                if num_dupes == 0:
                    cur_scope = get_scope()
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best_rows = [r[:] for r in current_rows]
            
            T *= alpha
            if T < 0.1:
                T = 0.1
            steps += 1
    
    local_search(max_steps=200000)
    
    return best_rows
