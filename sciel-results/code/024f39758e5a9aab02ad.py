import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a branch-and-bound search on a conflict graph of precomputed valid rows.
    """
    start_time = time.time()
    deadline = start_time + 116.0  # Safety margin before 120s limit

    # For this specific task (n_rows=7, k=5), we can use specialized logic.
    # But we must keep it general enough or at least handle the specific case well.
    
    # If n_rows and k are small, precomputation is viable.
    # If they are large, we fall back to SA.
    
    if n_rows <= 10 and k <= 6:
        return construct_mwis(n_rows, k, deadline)
    else:
        return construct_sa_fallback(n_rows, k, deadline)

def construct_mwis(n_rows, k, deadline):
    """
    Solves the DTS problem by finding an independent set of size n_rows
    in the conflict graph of valid rows, minimizing the maximum entry.
    """
    # 1. Generate all valid rows up to a certain scope
    # Lower bound: n_rows * k * (k+1) / 2
    lb = n_rows * k * (k + 1) // 2
    # Upper bound: Start with a greedy solution to get a decent upper bound
    greedy_rows = greedy_construct(n_rows, k)
    greedy_scope = max(max(row) for row in greedy_rows)
    
    # We will search scopes between lb and greedy_scope
    # Let's generate rows up to greedy_scope + 10
    
    max_scope_gen = greedy_scope + 20
    
    rows_pool = []
    # Generate rows
    # A row is [0, r1, r2, ..., rk]
    # We can generate them by iterating r1, r2, etc.
    # Or simply iterate all combinations? No, too many.
    # Iterative generation:
    # r[0] = 0
    # for r[1] in 1..max_scope_gen:
    #   for r[2] in r[1]+1..max_scope_gen:
    #     ...
    # Check validity: all differences distinct.
    
    # To speed up generation, we can use backtracking during generation.
    def gen_rows(k, max_scope):
        valid_rows = []
        def backtrack(current_row, used_diffs):
            if len(current_row) == k + 1:
                valid_rows.append(current_row[:])
                return
            
            start = current_row[-1] + 1 if current_row else 1
            # If current_row is empty, start is 1? No, current_row[0] is 0.
            # So current_row always has 0.
            
            for v in range(start, max_scope + 1):
                new_diffs = {v - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                # Also check internal validity of new_diffs? 
                # Since current_row was valid, and v is new, 
                # we just need to ensure new_diffs don't conflict with each other?
                # new_diffs are {v - current_row[i]}. 
                # Are they distinct? 
                # v - a_i == v - a_j => a_i == a_j, which is false since current_row is strictly increasing.
                # So new_diffs are always distinct internally.
                # We just need to check against used_diffs.
                
                current_row.append(v)
                used_diffs |= new_diffs
                backtrack(current_row, used_diffs)
                used_diffs -= new_diffs
                current_row.pop()
        
        backtrack([0], set())
        return valid_rows

    # Generate pool
    rows_pool = gen_rows(k, max_scope_gen)
    
    # Sort by max entry
    rows_pool.sort(key=lambda r: max(r))
    
    # Precompute difference sets for all rows
    row_diffs = []
    for r in rows_pool:
        diffs = set()
        n = len(r)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(r[j] - r[i])
        row_diffs.append(diffs)
        
    # Binary search on scope
    low = lb
    high = greedy_scope
    
    # Function to check if a solution exists with max scope <= M
    def exists_solution(M):
        # Filter rows with max <= M
        candidate_indices = [i for i, r in enumerate(rows_pool) if max(r) <= M]
        
        if len(candidate_indices) < n_rows:
            return None
        
        # Backtracking search
        # State: selected_indices, forbidden_diffs
        # We want to select n_rows indices from candidate_indices such that no two share a diff.
        
        # Heuristic: Order candidates by "degree" (number of conflicts with other candidates)
        # Or just by index (which is sorted by scope).
        # Let's try ordering by scope first, then by "flexibility" (fewer conflicts).
        
        # To make it faster, we can use a greedy pre-filter or a better ordering.
        # Let's use a recursive function with pruning.
        
        # Optimization: If the number of candidates is small, brute force is fine.
        # If large, we need good pruning.
        
        # Let's try a simple backtracking with a greedy bound.
        # Bound: The maximum number of non-conflicting rows we can still pick from the remaining candidates.
        # A simple bound is just the count, but we can do better with greedy coloring.
        
        # For now, let's try a direct backtracking. 
        # Since n_rows is small (7), the depth is small.
        # The branching factor is the number of candidates.
        # To reduce branching, we can order candidates such that those with fewer conflicts are tried first?
        # Or those with smaller scope?
        
        # Let's sort candidate_indices by scope (already done in rows_pool)
        # And then maybe by "conflict count" within the candidate set?
        
        # Precompute conflict sets for candidates? 
        # conflict_set[i] = set of indices j in candidate_indices that conflict with i.
        # This might be expensive if candidate_indices is large.
        # But max_scope_gen is ~130. Number of rows might be ~5000.
        # 5000^2 is 25M checks. Each check is set intersection. Might be slow in Python.
        
        # Alternative: On-the-fly conflict check.
        
        def backtrack(selected, forbidden_diffs, start_idx):
            if len(selected) == n_rows:
                return selected[:]
            
            # Pruning: 
            # If we can't fill the remaining slots, return None.
            # Simple bound: len(candidate_indices) - start_idx < n_rows - len(selected)
            if len(candidate_indices) - start_idx < n_rows - len(selected):
                return None
            
            # Better bound: Greedy Independent Set size on remaining candidates
            # We can approximate this.
            
            for idx in range(start_idx, len(candidate_indices)):
                r_idx = candidate_indices[idx]
                diffs = row_diffs[r_idx]
                
                if diffs & forbidden_diffs:
                    continue
                
                # Try adding this row
                new_selected = selected + [r_idx]
                new_forbidden = forbidden_diffs | diffs
                
                # Optimization: If we pick a row, we can skip candidates that conflict with it?
                # No, we just continue from idx+1.
                
                res = backtrack(new_selected, new_forbidden, idx + 1)
                if res is not None:
                    return res
            
            return None
        
        # To speed up, we can sort candidate_indices by a heuristic.
        # Heuristic: Rows with smaller max entry are preferred (already sorted).
        # But maybe rows with "rarer" differences are better?
        
        # Let's just try the simple backtracking. 
        # If it's too slow, we add a timeout and return None.
        
        import threading
        # We can't easily use threads for this small logic, so we just rely on the outer deadline.
        
        # One optimization: If n_rows is small, we can use a "meet in the middle" or just brute force if candidate count is small.
        # If candidate count > 1000, backtracking might be slow.
        
        # Let's add a simple greedy check first.
        # Greedy: Pick the row with smallest scope that doesn't conflict with selected.
        # If greedy succeeds, we found a solution.
        # If greedy fails, backtracking might still succeed, but it's less likely.
        
        # Actually, let's just run the backtracking.
        # To prevent timeout, we can limit the number of nodes visited.
        
        nodes_visited = 0
        max_nodes = 100000 # Tune this
        
        def backtrack_limited(selected, forbidden_diffs, start_idx):
            nonlocal nodes_visited
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return None
            
            if len(selected) == n_rows:
                return selected[:]
            
            if len(candidate_indices) - start_idx < n_rows - len(selected):
                return None
            
            for idx in range(start_idx, len(candidate_indices)):
                r_idx = candidate_indices[idx]
                diffs = row_diffs[r_idx]
                
                if diffs & forbidden_diffs:
                    continue
                
                new_selected = selected + [r_idx]
                new_forbidden = forbidden_diffs | diffs
                
                res = backtrack_limited(new_selected, new_forbidden, idx + 1)
                if res is not None:
                    return res
            
            return None
        
        # Run backtrack
        res = backtrack_limited([], set(), 0)
        return res

    # Binary Search
    best_solution = None
    while low <= high and time.time() < deadline - 2:
        mid = (low + high) // 2
        sol_indices = exists_solution(mid)
        
        if sol_indices is not None:
            # Found a solution with scope <= mid
            best_solution = [rows_pool[i][:] for i in sol_indices]
            high = mid - 1
        else:
            low = mid + 1
            
    if best_solution is None:
        # Fallback to greedy if search failed or timed out
        return greedy_rows
        
    return best_solution

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def construct_sa_fallback(n_rows, k, deadline):
    # Fallback to the original SA logic if MWIS is not applicable or fails
    # (Simplified version of the original SA)
    
    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_used(rows):
        all_used = set()
        for row in rows:
            all_used |= _row_diffs(row)
        return all_used

    def _global_max(rows):
        return max(max(row) for row in rows)

    def local_search_init(n_rows, k, seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 300.0
        T_min = 0.1
        cool = 0.99995
        
        no_improve = 0
        max_no_improve = 50000
        
        while time.time() < deadline and no_improve < max_no_improve:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
            
            # Decreasing moves
            for d in range(1, min(20, old_val - min_allowed) + 1):
                candidates.append(old_val - d)
            # Increasing moves
            for d in range(1, min(10, max_allowed - old_val) + 1):
                candidates.append(old_val + d)
                
            if random.random() < 0.2:
                low = current_rows[ri][pi - 1] + 1
                high = max_allowed
                if low <= high:
                    candidates.append(random.randint(low, high))

            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            candidates.sort(key=lambda v: (v > old_val, v))

            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                if new_val == old_val:
                    continue

                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                new_diffs = _row_diffs(cand_row)
                
                # Check validity
                valid = True
                if len(new_diffs) != k * (k + 1) // 2:
                    valid = False
                else:
                    for d in new_diffs:
                        if d in all_used and d not in row_diffs[ri]:
                            valid = False
                            break
                
                if not valid:
                    continue

                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.3:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.05))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                    moved = True
                    break

            if not moved:
                no_improve += 1
                
            T = max(T_min, T * cool)

        return best_rows, best_val

    best_solution = greedy_construct(n_rows, k)
    best_scope = _global_max(best_solution)
    
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        init_rows = local_search_init(n_rows, k, seed=trial)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, time.time(), deadline - 2)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows
            
    return best_solution

# EVOLVE-BLOCK-END
