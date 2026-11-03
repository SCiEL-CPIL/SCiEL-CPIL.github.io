import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    m = k + 1  # entries per row
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    lb = total_diffs_needed  # Counting bound: smallest M such that M >= total_diffs
    
    # 1. Fast greedy baseline
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < m:
                # Check if x - a is in used for all a in row
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
                x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    if best_scope <= lb:
        return best_rows

    # 2. Local search / Polish to reduce scope
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for d in get_diffs(row):
                if d in used:
                    return False
                used.add(d)
        return True

    def polish(rows, time_limit):
        rows = [list(r) for r in rows]
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for pos in range(m - 1, 0, -1):
                    orig = row[pos]
                    # Try to move row[pos] down
                    for new_val in range(row[pos - 1] + 1, orig):
                        # Check consistency
                        # New diffs involving new_val
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos + 1, m):
                            new_diffs.add(row[j] - new_val)
                        
                        # Old diffs involving orig
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(orig - row[i])
                        for j in range(pos + 1, m):
                            old_diffs.add(row[j] - orig)
                        
                        # Diffs not involving pos (unchanged)
                        # We need to check if new_diffs \ old_diffs conflicts with used_diffs - old_diffs
                        # Actually, simpler:
                        # 1. Compute all diffs of the row with new_val
                        test_row = row[:pos] + [new_val] + row[pos+1:]
                        # Check strict increase
                        for i in range(1, m):
                            if test_row[i] <= test_row[i-1]:
                                break
                        else:
                            # Check uniqueness within row
                            if len(get_diffs(test_row)) != total_diffs_per_row:
                                continue
                            
                            # Check global uniqueness
                            # This is O(N^2) per step, might be slow. 
                            # For small N it's fine.
                            # To speed up, we can maintain a global set, but modifying it in place is tricky.
                            # Let's just do the full check for simplicity in polish, as it's called on a small number of rows.
                            all_diffs = []
                            for r in rows:
                                if r is rows[r_idx]:
                                    all_diffs.extend(get_diffs(test_row))
                                else:
                                    all_diffs.extend(get_diffs(r))
                            
                            if len(all_diffs) == len(set(all_diffs)):
                                rows[r_idx] = test_row
                                improved = True
                                break # Move to next pos? Or break out of loop?
                                # Breaking inner loop, continue with next pos
            if not improved:
                break
        return rows

    # Polish the greedy solution
    best_rows = polish(best_rows, deadline - 10.0)
    best_scope = max(max(r) for r in best_rows)

    # 3. Backtracking search for smaller scopes
    # We try to find a solution with scope M < best_scope.
    # We iterate M from best_scope - 1 down to lb.
    
    def solve_with_scope(M, seed):
        rng = random.Random(seed)
        used_diffs = [False] * (M + 1)
        rows = []
        
        # Backtracking for one row
        def build_row(pos, current_val, row_so_far):
            if time.time() > deadline:
                return None
            if pos == m:
                return list(row_so_far)
            
            # Pruning: need space for remaining elements
            max_x = M - (m - pos - 1)
            if max_x <= current_val:
                return None
            
            valid_candidates = []
            for x in range(current_val + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if d > M:
                        continue # Should not happen if x <= M
                    if used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None
            
            # Heuristic: try smaller values first to keep scope tight, but add randomness
            # Sort candidates, then shuffle a subset?
            # Let's sort and then pick randomly from the first few to bias towards small
            valid_candidates.sort()
            # Take up to 20 smallest
            candidates_to_try = valid_candidates[:20]
            rng.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                # Mark diffs
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True
                
                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                
                # Unmark
                for d in new_diffs:
                    used_diffs[d] = False
            
            return None

        # Build rows one by one
        # To increase diversity, we can shuffle the order in which we try to build rows?
        # No, rows are indistinguishable in terms of constraints, but the "used_diffs" set grows.
        # Building "harder" rows first might be better? Or easier?
        # Usually, building rows that leave fewer options for others is good.
        # Greedy tends to make rows that use small numbers.
        # Let's just build in order.
        
        for _ in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            # Mark all diffs in this row as used
            for i in range(m):
                for j in range(i + 1, m):
                    used_diffs[row[j] - row[i]] = True
        
        return rows

    # Try to improve scope
    # We start from best_scope - 1 down to lb
    # If we fail at M, we can stop? 
    # Strictly speaking, if no solution exists for M, no solution exists for M-1.
    # However, our search is randomized and might fail to find a solution even if one exists.
    # So we should try a few restarts for each M.
    # If we exhaust restarts and fail, we assume M is too small and break.
    
    attempts_per_scope = 5
    seed_counter = 0
    
    # We can also try to jump M down if we have time
    # But linear descent is safer.
    
    for M in range(best_scope - 1, lb - 1, -1):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        for _ in range(attempts_per_scope):
            if time.time() > deadline - 5.0:
                break
            
            rows = solve_with_scope(M, seed_counter)
            seed_counter += 1
            
            if rows is not None:
                if is_valid(rows):
                    best_rows = rows
                    best_scope = M
                    found = True
                    break
        
        if not found:
            # If we didn't find a solution for M, it's likely infeasible or we got unlucky.
            # Given the small gap, breaking is reasonable to save time for potential polish?
            # But if we break, we return the current best.
            # Let's break to be safe and return the best found.
            break
            
        # If found, continue to M-1

    # Final polish
    if time.time() < deadline - 2.0:
        best_rows = polish(best_rows, deadline - 2.0)
        best_scope = max(max(r) for r in best_rows)

    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
