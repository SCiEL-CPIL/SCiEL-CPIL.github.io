import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    row_len = k + 1
    # Total differences per row is k*(k+1)/2
    # Total differences overall is n_rows * k*(k+1)/2
    
    # We want to minimize M (the largest entry).
    # Lower bound: The counting argument suggests M >= 112.
    # Let's try values from 112 upwards.
    
    best_solution = None
    best_M = float('inf')
    
    def solve_with_scope(M):
        """
        Tries to find a valid configuration with max element <= M.
        Uses backtracking with constraint propagation.
        Returns the configuration if found, else None.
        """
        used_diffs = set()
        rows = []
        
        # We build rows one by one.
        # To reduce symmetry, we enforce that rows are lexicographically non-decreasing.
        # Since all rows start with 0, this means row[i+1] >= row[i] lexicographically.
        
        def is_lex_geq(row_a, row_b, pos):
            """
            Checks if row_a (current partial) is lexicographically >= row_b (fully formed)
            up to index pos.
            If row_a is a prefix of row_b, it must be <= at the next step?
            Actually, we just need to ensure the final row is >= previous row.
            We can check this incrementally.
            """
            for i in range(pos):
                if row_a[i] > row_b[i]:
                    return True
                if row_a[i] < row_b[i]:
                    return False
            # If equal so far, the next element will determine it, or if pos == len, it's equal.
            # If pos < len, we haven't decided yet.
            # For the purpose of pruning candidates for the current position `pos`,
            # if the prefix is equal, the current value must be >= row_b[pos].
            # If the prefix is already larger, any value is fine (regarding lex order).
            # If the prefix is smaller, it's invalid (but we should have pruned earlier).
            return True # If equal so far, we allow >= at current pos.
        
        def backtrack(row_idx, pos_idx):
            if row_idx == n_rows:
                return True
            
            if pos_idx == 0:
                # Start a new row
                rows.append([0])
                if backtrack(row_idx, 1):
                    return True
                rows.pop()
                return False
            
            current_row = rows[row_idx]
            prev_val = current_row[pos_idx - 1]
            
            # Determine lower bound
            lb = prev_val + 1
            
            # Lexicographic constraint with previous row
            if row_idx > 0:
                prev_row = rows[row_idx - 1]
                # Check prefix
                lex_greater = False
                for i in range(pos_idx):
                    if current_row[i] > prev_row[i]:
                        lex_greater = True
                        break
                    if current_row[i] < prev_row[i]:
                        # This should not happen if we enforce constraints correctly at each step
                        # But if it does, this branch is invalid
                        return False
                
                if not lex_greater:
                    # Prefix is equal, so current val must be >= prev_row[pos_idx]
                    lb = max(lb, prev_row[pos_idx])
            
            # Upper bound is M
            ub = M
            
            # Iterate candidates
            # Heuristic: Try smaller values first to keep scope low
            for val in range(lb, ub + 1):
                # Check conflicts with existing elements in current_row
                valid = True
                new_diffs = []
                for i in range(pos_idx):
                    d = val - current_row[i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Accept this value
                current_row.append(val)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx, pos_idx + 1):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        if backtrack(0, 0):
            return rows
        return None

    # Search for the minimum M
    # Start from a reasonable lower bound.
    # Counting bound: total diffs = 4 * 28 = 112.
    # Max diff is M. Number of possible diffs is M.
    # So M >= 112.
    
    M_start = 112
    M_end = 150 # Safe upper bound
    
    # We can try to be smarter and jump, but linear is safe for this small range.
    # To speed up, we can parallelize or use better heuristics, but 112-150 is small.
    
    # Optimization: Instead of trying every M, we can just run the backtracker
    # with a fixed M and see if it finds a solution.
    # If we want the *minimum* M, we must check from low to high.
    
    # However, backtracking with M=112 might be very slow if no solution exists.
    # We should add a node limit or time limit to the backtracker.
    
    def solve_with_scope_limited(M, time_limit):
        used_diffs = set()
        rows = []
        node_count = [0]
        start_t = time.time()
        
        def is_lex_geq_prefix(current_row, prev_row, pos):
            """
            Returns:
            1 if current_row[0:pos] > prev_row[0:pos]
            0 if current_row[0:pos] == prev_row[0:pos]
            -1 if current_row[0:pos] < prev_row[0:pos]
            """
            for i in range(pos):
                if current_row[i] > prev_row[i]:
                    return 1
                if current_row[i] < prev_row[i]:
                    return -1
            return 0

        def backtrack(row_idx, pos_idx):
            if time.time() - start_t > time_limit:
                return False
            
            node_count[0] += 1
            
            if row_idx == n_rows:
                return True
            
            if pos_idx == 0:
                rows.append([0])
                if backtrack(row_idx, 1):
                    return True
                rows.pop()
                return False
            
            current_row = rows[row_idx]
            prev_val = current_row[pos_idx - 1]
            
            lb = prev_val + 1
            
            if row_idx > 0:
                prev_row = rows[row_idx - 1]
                cmp = is_lex_geq_prefix(current_row, prev_row, pos_idx)
                if cmp < 0:
                    return False # Invalid branch
                if cmp == 0:
                    lb = max(lb, prev_row[pos_idx])
                # if cmp > 0, lb is just prev_val + 1
            
            ub = M
            
            for val in range(lb, ub + 1):
                valid = True
                new_diffs = []
                for i in range(pos_idx):
                    d = val - current_row[i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                current_row.append(val)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx, pos_idx + 1):
                    return True
                
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        if backtrack(0, 0):
            return rows
        return None

    # We will try M values from 112 to 140.
    # We give each M a small time budget to see if a solution exists.
    # If a solution is found, we return it.
    # If time runs out, we return the best partial solution or a known good one.
    
    # Actually, if we just want *a* solution with low scope, we can just pick M=134 (known good)
    # and try to find a solution. If we find one, score is 134.
    # But we want to *minimize* it.
    # Let's try M=112, 113, ...
    
    # To be safe against timeouts, we limit the search time per M.
    # If we don't find a solution for M in X seconds, we assume it's hard and try M+1?
    # Or just give up and use a heuristic for that M?
    # Better: Just find *any* valid solution. The lowest M we find is the score.
    
    # Strategy:
    # 1. Try M=112 with a tight time limit (e.g., 2s). If found, return.
    # 2. Try M=113 with a tight time limit.
    # ...
    # 3. If we run out of time, return the best valid solution found so far.
    #    If none found, use a greedy fallback to ensure validity, even if scope is higher.
    
    best_found = None
    best_M_found = float('inf')
    
    # We know 134 is possible. Let's ensure we have a fallback.
    # Greedy fallback:
    def greedy_fallback():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        break
                    x += 1
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
            rows.append(row)
        return rows

    fallback_rows = greedy_fallback()
    fallback_scope = max(max(r) for r in fallback_rows)
    
    # Search loop
    # We'll allocate time dynamically.
    # Total time available ~ 140s.
    # Let's try M from 112 to 135.
    # For each M, allow up to 5 seconds of backtracking.
    # If we find a solution, we stop (since we're going up from min).
    # Wait, if we find a solution for M=115, is it possible that M=112 has a solution?
    # Yes, but we tried it and failed (or timed out).
    # If we timed out, we don't know.
    # So we should keep trying lower M's if we have time?
    # No, if we go 112, 113, 114... and 112 times out, 113 might be easier or harder.
    # Usually, smaller M is harder.
    # So if 112 fails, 113 might also fail.
    # But if 112 succeeds, we are done.
    
    # Let's just iterate M from 112 to 140.
    # For each M, run backtracker with a time limit.
    # If found, return immediately.
    
    time_per_M = 2.0 # seconds
    for M in range(112, 141):
        if time.time() > deadline - 10: # Leave 10s for final checks
            break
            
        # Run solver
        res = solve_with_scope_limited(M, time_per_M)
        
        if res:
            # Verify validity
            all_diffs = []
            valid = True
            for r in res:
                if r[0] != 0:
                    valid = False
                    break
                for i in range(row_len):
                    for j in range(i+1, row_len):
                        d = r[j] - r[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.append(d)
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                scope = max(max(r) for r in res)
                # It's possible the solver found a solution with max < M?
                # No, we constrained val <= M.
                # But maybe the actual max is smaller.
                # We return the rows. The evaluator calculates the scope.
                return res
        
    # If we didn't find a solution in the loop, return the fallback
    # But the fallback might have a very high scope.
    # We can try to improve the fallback with local search if we have time.
    
    if time.time() < deadline - 5:
        # Run a quick local search on the fallback to reduce scope
        # This is the same as the original program's logic but on the fallback.
        # For simplicity and safety, we just return the fallback.
        # The fallback is guaranteed valid.
        pass
        
    return fallback_rows

# Test locally
if __name__ == "__main__":
    rows = construct(4, 7)
    # Verify
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
            break
        for i in range(8):
            for j in range(i+1, 8):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    scope = max(max(r) for r in rows)
    print(f"Valid: {valid}, Scope: {scope}")
    print(rows)
