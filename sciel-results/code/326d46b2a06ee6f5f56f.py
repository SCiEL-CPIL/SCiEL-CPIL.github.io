import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def get_row_diffs(row):
        diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs

    def is_valid_row(row, global_used_except_self):
        # Check internal distinctness and external distinctness
        row_diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if (row_diffs >> d) & 1:
                    return False, 0 # Internal conflict
                row_diffs |= (1 << d)
        if row_diffs & global_used_except_self:
            return False, 0 # External conflict
        return True, row_diffs

    def greedy_row(max_val, global_used):
        """
        Greedily construct a row of length k+1 starting with 0, 
        with all values <= max_val, avoiding conflicts with global_used.
        """
        row = [0]
        current_diffs = 0
        
        while len(row) < k + 1:
            last = row[-1]
            found = False
            # Try smallest possible next value
            for v in range(last + 1, max_val + 1):
                # Check conflicts with current row
                conflict = False
                new_diffs = 0
                for u in row:
                    d = v - u
                    if (current_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                
                if conflict:
                    continue
                
                # Check conflicts with global
                if new_diffs & global_used:
                    continue
                
                # Accept
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            
            if not found:
                return None
        return row

    def random_row(max_val, global_used, attempts=1):
        """
        Try to find a valid row with max <= max_val using randomized greedy.
        """
        for _ in range(attempts):
            row = [0]
            current_diffs = 0
            success = True
            for _ in range(k):
                last = row[-1]
                # Limit candidate range to avoid excessive shuffling time if max_val is large
                # But we need to ensure we can find a valid one.
                candidates = list(range(last + 1, max_val + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    conflict = False
                    new_diffs = 0
                    for u in row:
                        d = v - u
                        if (current_diffs >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if conflict:
                        continue
                    if new_diffs & global_used:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                # Final check
                valid, _ = is_valid_row(row, global_used)
                if valid:
                    return row
        return None

    def backtrack_solve(rows, row_idx, max_val, time_limit):
        """
        Recursively try to fill rows[row_idx:] such that all rows are valid and max entry <= max_val.
        Returns True if solution found, updating rows in place.
        """
        if time.time() > time_limit:
            return False
        if row_idx == n_rows:
            return True
        
        # Calculate used diffs from rows before row_idx
        used_before = 0
        for i in range(row_idx):
            used_before |= get_row_diffs(rows[i])
        
        candidates = []
        
        # Try greedy first as it's fast and often good
        g_row = greedy_row(max_val, used_before)
        if g_row is not None:
            candidates.append(g_row)
            
        # Try a few random constructions
        for _ in range(2):
            if time.time() > time_limit:
                return False
            row = random_row(max_val, used_before, attempts=1)
            if row is not None:
                candidates.append(row)
        
        if not candidates:
            return False
            
        random.shuffle(candidates)
        
        for c_row in candidates:
            rows[row_idx] = c_row
            if backtrack_solve(rows, row_idx + 1, max_val, time_limit):
                return True
        return False

    def find_solution_for_scope(M, time_limit):
        for attempt in range(5):
            if time.time() > time_limit:
                return None
            rows = [None] * n_rows
            if backtrack_solve(rows, 0, M, time_limit):
                return rows
        return None

    # 1. Initial Solution: Greedy with large scope
    init_scope = 250
    rows = [None] * n_rows
    success = False
    
    # Try to find a solution with backtracking within a reasonable time
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            rows = [None] * n_rows
            if backtrack_solve(rows, 0, init_scope, deadline - 20):
                success = True
                break
    
    if not success:
        # Fallback: Very large scope with pure greedy (should always work)
        rows = []
        global_used = 0
        for i in range(n_rows):
            row = greedy_row(500, global_used)
            if row is None:
                row = [0] + list(range(1, k+1))
            rows.append(row)
            global_used |= get_row_diffs(row)
            
    best_rows = [list(r) for r in rows]
    best_max = get_max_entry(best_rows)
    current_rows = [list(r) for r in rows]
    current_max = best_max

    # 2. Optimization Loop: Binary Search for Minimum Max
    # We know best_max is valid. We want to find the smallest M such that a solution exists with max <= M.
    
    low = lower_bound
    high = best_max
    
    search_start_time = time.time()
    remaining_time = deadline - search_start_time - 5.0 # 5s margin
    
    if remaining_time > 10:
        while low < high and (time.time() - search_start_time) < remaining_time - 5:
            mid = (low + high) // 2
            # Check if mid is feasible
            # Allocate half the remaining time to this check
            check_time_limit = time.time() + (remaining_time - (time.time() - search_start_time)) * 0.5
            sol = find_solution_for_scope(mid, check_time_limit)
            if sol is not None:
                # Mid is feasible
                best_rows = [list(r) for r in sol]
                best_max = get_max_entry(best_rows)
                high = mid
            else:
                # Mid is not feasible (or timed out)
                low = mid + 1
        
        # After binary search, try to polish the best solution found
        if best_rows:
            current_rows = best_rows
            current_max = best_max
    
    # 3. Local Search to Polish
    # Try to shrink individual rows to reduce the global max further.
    step = 0
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        step += 1
        
        # Pick a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        
        # Calculate global used for other rows
        global_used_except = 0
        for i in range(n_rows):
            if i != row_idx:
                global_used_except |= get_row_diffs(current_rows[i])
                
        curr_row_max = current_rows[row_idx][-1]
        
        # Target: reduce curr_row_max
        target = curr_row_max - 1
        
        if target < k: # Minimum possible max for a row of length k+1 starting at 0 is k
            continue
            
        new_row = None
        # Try a few times to find a smaller row
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            new_row = random_row(target, global_used_except, attempts=1)
            if new_row is not None:
                break
        
        if new_row is not None:
            # Validate new row against global
            valid, _ = is_valid_row(new_row, global_used_except)
            if valid:
                # Update state
                current_rows[row_idx] = new_row
                new_global_max = get_max_entry(current_rows)
                if new_global_max < best_max:
                    best_max = new_global_max
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1
            
    # Final Validation
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

    if not validate(best_rows):
        # Fallback to a known valid structure if something went wrong
        rows_fallback = []
        gu = 0
        for i in range(n_rows):
            r = greedy_row(1000, gu)
            if r is None:
                r = [0] + list(range(1, k+1))
            rows_fallback.append(r)
            gu |= get_row_diffs(r)
        best_rows = rows_fallback

    return best_rows
