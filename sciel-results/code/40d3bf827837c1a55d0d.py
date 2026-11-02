import time
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Greedy baseline
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Backtracking with binary search on Scope M
    # Lower bound: We need at least n * (k+1) numbers? No, differences must be distinct.
    # Counting bound: Number of distinct differences is n * k(k+1)/2.
    # Max difference is M. So M(M+1)/2 >= n * k(k+1)/2 => M >= sqrt(n*k(k+1)) roughly.
    # Actually, differences are from 1..M. There are M possible differences.
    # We need N_diff = n * k(k+1)/2 distinct differences.
    # So M >= N_diff.
    # For n=7, k=5: N_diff = 7 * 15 = 105. So M >= 105.
    
    lb = n_rows * k * (k + 1) // 2
    if lb < 1: lb = 1
    
    # We know best_scope is a valid upper bound.
    # We want to find if there is a solution with M < best_scope.
    # We binary search M in [lb, best_scope - 1].
    
    def can_construct(M):
        # Returns True if a solution exists with max entry <= M
        used_diffs = 0
        rows = []
        
        # Bitmask check helper
        # Diffs are 1..M. We use bits 1..M.
        
        def backtrack(row_idx, curr_row, last_val):
            nonlocal used_diffs, rows
            
            if time.time() > deadline:
                raise TimeoutError()
                
            if row_idx == n_rows:
                return True
            
            if len(curr_row) == k + 1:
                # Row complete
                rows.append(list(curr_row))
                # Start next row
                res = backtrack(row_idx + 1, [0], 0)
                if res:
                    return True
                rows.pop()
                return False
            
            # Determine next value x
            # x must be > last_val and <= M
            # Also, the new differences must not conflict with used_diffs
            
            # Optimization: 
            # The smallest possible next value is last_val + 1.
            # The largest is M.
            
            # We can prune if the number of remaining slots in the row 
            # cannot be filled with values <= M given the current last_val.
            # Need (k+1 - len(curr_row)) more values.
            # Max value is M.
            # Min value is last_val + 1.
            # So we need M - last_val >= (k+1 - len(curr_row)).
            if M - last_val < (k + 1 - len(curr_row)):
                return False
            
            for x in range(last_val + 1, M + 1):
                # Calculate new differences
                # new_diffs = {x - a for a in curr_row}
                # Check if any diff is in used_diffs
                valid = True
                new_mask = 0
                for a in curr_row:
                    d = x - a
                    # d is between 1 and M
                    bit = 1 << d
                    if used_diffs & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if not valid:
                    continue
                
                # Accept x
                curr_row.append(x)
                used_diffs |= new_mask
                
                if backtrack(row_idx, curr_row, x):
                    return True
                
                # Backtrack
                curr_row.pop()
                used_diffs ^= new_mask
                
            return False
        
        # Start with row 0, curr_row [0], last_val 0
        # Note: 0 is not a positive difference, so it doesn't affect used_diffs mask initially.
        # But wait, the first element is 0. The differences are calculated from 0.
        # So for the first element 0, there are no differences.
        # The loop above handles this: curr_row starts as [0].
        # When we pick x, d = x - 0 = x.
        
        return backtrack(0, [0], 0)

    # Binary search
    lo = lb
    hi = best_scope - 1
    found_better = False
    
    # If lb >= best_scope, no need to search
    if lo < hi:
        while lo <= hi:
            mid = (lo + hi) // 2
            if time.time() > deadline - 5: # Reserve time for finalization
                break
            
            # Try to solve for scope mid
            # We need to capture the rows if it succeeds
            # Modify can_construct to return rows
            
            # Let's refactor can_construct to return rows or None
            def solve_for_scope(M):
                used_diffs = 0
                rows = []
                
                def backtrack(row_idx, curr_row, last_val):
                    nonlocal used_diffs, rows
                    if time.time() > deadline:
                        raise TimeoutError()
                    
                    if row_idx == n_rows:
                        return True
                    
                    if len(curr_row) == k + 1:
                        rows.append(list(curr_row))
                        res = backtrack(row_idx + 1, [0], 0)
                        if res:
                            return True
                        rows.pop()
                        return False
                    
                    if M - last_val < (k + 1 - len(curr_row)):
                        return False
                    
                    for x in range(last_val + 1, M + 1):
                        valid = True
                        new_mask = 0
                        for a in curr_row:
                            d = x - a
                            bit = 1 << d
                            if used_diffs & bit:
                                valid = False
                                break
                            new_mask |= bit
                        
                        if not valid:
                            continue
                        
                        curr_row.append(x)
                        used_diffs |= new_mask
                        
                        if backtrack(row_idx, curr_row, x):
                            return True
                        
                        curr_row.pop()
                        used_diffs ^= new_mask
                        
                    return False
                
                try:
                    if backtrack(0, [0], 0):
                        return rows
                except TimeoutError:
                    return None
                return None

            res = solve_for_scope(mid)
            if res:
                # Found a valid solution with scope mid
                best_rows = res
                best_scope = mid
                hi = mid - 1
                found_better = True
            else:
                lo = mid + 1

    return best_rows
