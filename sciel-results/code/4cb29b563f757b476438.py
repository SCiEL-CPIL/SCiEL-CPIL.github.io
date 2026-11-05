import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def solve_for_scope(M, time_limit):
        used_mask = 0
        rows = []
        
        def backtrack(row_idx, current_row, current_mask):
            if time.time() > time_limit:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                new_used_mask = used_mask | current_mask
                result = backtrack(row_idx + 1, [0], 0)
                if result is not None:
                    return result
                rows.pop()
                return None
            
            last_val = current_row[-1]
            
            # Optimization: If we are at the last few elements, we can be more aggressive
            for v in range(last_val + 1, M + 1):
                new_diffs_mask = 0
                valid = True
                for u in current_row:
                    d = v - u
                    bit = 1 << d
                    if (current_mask & bit) or (used_mask & bit):
                        valid = False
                        break
                    new_diffs_mask |= bit
                
                if not valid:
                    continue
                
                current_row.append(v)
                new_current_mask = current_mask | new_diffs_mask
                
                result = backtrack(row_idx, current_row, new_current_mask)
                if result is not None:
                    return result
                
                current_row.pop()
            
            return None

        return backtrack(0, [0], 0)

    def greedy_construct(M):
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            row_mask = 0
            for _ in range(k):
                last = row[-1]
                found = False
                for v in range(last + 1, M + 1):
                    new_diffs_mask = 0
                    valid = True
                    for u in row:
                        d = v - u
                        bit = 1 << d
                        if (row_mask & bit) or (used_mask & bit):
                            valid = False
                            break
                        new_diffs_mask |= bit
                    if valid:
                        row.append(v)
                        row_mask |= new_diffs_mask
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
            used_mask |= row_mask
        return rows

    # 1. Baseline
    baseline = greedy_construct(500)
    if baseline is None:
        # Fallback: Construct with very large gaps
        baseline = []
        used_mask = 0
        for i in range(n_rows):
            row = [0]
            row_mask = 0
            for j in range(k):
                # Pick a value that is definitely valid
                # Since M is large, we can just increment by a large prime or similar
                # But we need to check.
                # For fallback, just use greedy with M=1000
                pass
            
            # Simpler fallback:
            row = [0]
            row_mask = 0
            next_v = 0
            for j in range(k):
                while True:
                    next_v += 1
                    new_diffs_mask = 0
                    valid = True
                    for u in row:
                        d = next_v - u
                        bit = 1 << d
                        if (row_mask & bit) or (used_mask & bit):
                            valid = False
                            break
                        new_diffs_mask |= bit
                    if valid:
                        row.append(next_v)
                        row_mask |= new_diffs_mask
                        break
            baseline.append(row)
            used_mask |= row_mask
            
    best_rows = baseline
    best_max = max(max(r) for r in best_rows)
    
    # 2. Binary Search
    low = 112
    high = best_max
    
    # If best_max is small, we might not need to search
    while low < high and time.time() < deadline - 10:
        mid = (low + high) // 2
        # Limit solver time to 2 seconds per check to allow more iterations
        check_deadline = time.time() + 2.0
        if check_deadline > deadline - 5:
            check_deadline = deadline - 5
            
        sol = solve_for_scope(mid, check_deadline)
        if sol is not None:
            best_rows = sol
            best_max = max(max(r) for r in best_rows)
            high = mid
        else:
            low = mid + 1
            
    # 3. Final Polish: Try to reduce best_max by 1 repeatedly if time permits
    # This is essentially a linear search downwards from best_max
    # But binary search should have found the minimum feasible M.
    
    # Validation
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
        best_rows = baseline
        
    return best_rows
