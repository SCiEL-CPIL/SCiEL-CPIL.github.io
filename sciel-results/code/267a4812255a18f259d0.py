import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    row_len = k + 1
    # Lower bound estimate: simple counting bound is k(k+1)/2 * n_rows
    # For n=4, k=7, diffs per row = 28. Total diffs = 112.
    # Max diff is scope. So scope >= 112 is a very rough lower bound if diffs are 1..scope.
    # Actually, the counting bound is that the number of distinct diffs must be <= scope.
    # So scope >= total_diffs.
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    def get_scope(rows):
        return max(max(r) for r in rows)
        
    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def solve_with_scope(target_scope):
        # rows[r][i] = value, -1 if unassigned
        rows = [[-1] * row_len for _ in range(n_rows)]
        for r in range(n_rows):
            rows[r][0] = 0
            
        used_diffs = set()
        
        # List of unassigned variables (r, i)
        # We will dynamically find the MRV variable
        
        def backtrack():
            # Check time
            if time.time() > deadline:
                return False
            
            # Find MRV variable
            best_r = -1
            best_i = -1
            min_candidates = float('inf')
            best_cands = []
            
            # We iterate through all unassigned variables to find the one with fewest options
            # Optimization: We can keep a list of unassigned vars, but with N=32, O(N^2) check is fine
            
            for r in range(n_rows):
                for i in range(1, row_len):
                    if rows[r][i] != -1:
                        continue
                    
                    # Determine lower bound
                    prev_val = rows[r][i-1]
                    lower = prev_val + 1
                    upper = target_scope
                    
                    if lower > upper:
                        return False # No space left
                    
                    # Count candidates
                    # A candidate v is valid if v - rows[r][j] not in used_diffs for all j < i where rows[r][j] is assigned
                    # Note: Since we fill in order or via MRV, we need to be careful.
                    # If we use MRV, we might assign (0,2) before (0,1)?
                    # No, constraint x[r][i] > x[r][i-1] implies we must know x[r][i-1].
                    # So we can only assign i if i-1 is assigned.
                    # Thus, the "frontier" of unassigned variables for each row is just the next one.
                    
                    # Wait, if we strictly enforce order i-1 assigned before i, then for each row, there is at most ONE unassigned variable that is "available" to be assigned (the first unassigned one).
                    # Let's verify.
                    # If rows[r][i] is -1, can we assign it? Only if rows[r][i-1] is not -1.
                    # So for each row r, let next_unassigned[r] be the smallest i such that rows[r][i] == -1.
                    # We only consider these variables for MRV.
                    
                    # Let's restructure:
                    # Maintain next_unassigned[r]
                    # The set of candidate variables is { (r, next_unassigned[r]) for r in 0..n_rows-1 if next_unassigned[r] < row_len }
                    
                    # This simplifies MRV significantly.
                    pass
                    
            # Let's implement the simplified MRV with frontier tracking
            
            # We need to track next_unassigned
            # It's easier to just scan.
            
            # Actually, let's just scan all r, and find the first unassigned i.
            # That is the only assignable i for that row.
            
            for r in range(n_rows):
                # Find next unassigned
                i = 1
                while i < row_len and rows[r][i] != -1:
                    i += 1
                if i == row_len:
                    continue # Row complete
                
                # This (r, i) is the only candidate for row r
                prev_val = rows[r][i-1]
                lower = prev_val + 1
                upper = target_scope
                
                if lower > upper:
                    return False
                
                cands = []
                # We can limit the search for candidates to a window or all
                # To be safe and fast, we check all in [lower, upper]
                # But we can prune: if v - rows[r][j] is in used_diffs, skip.
                
                for v in range(lower, upper + 1):
                    valid = True
                    for j in range(i):
                        d = v - rows[r][j]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        cands.append(v)
                        
                if len(cands) == 0:
                    return False # Dead end for this row
                
                if len(cands) < min_candidates:
                    min_candidates = len(cands)
                    best_r = r
                    best_i = i
                    best_cands = cands
                    
                    # Early termination if only 1 candidate
                    if min_candidates == 1:
                        break
            
            if best_r == -1:
                # All rows complete
                return True
            
            # Try candidates for best_r, best_i
            # Heuristic: try smaller values first to keep scope tight? Or random?
            # Smaller values usually leave more room for future variables.
            # Let's try sorted order.
            
            for v in best_cands:
                rows[best_r][best_i] = v
                
                # Add new diffs
                new_diffs = []
                for j in range(best_i):
                    d = v - rows[best_r][j]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                if backtrack():
                    return True
                
                # Undo
                for d in new_diffs:
                    used_diffs.remove(d)
                rows[best_r][best_i] = -1
                
            return False
            
        return backtrack()

    # Main loop: try scopes starting from lower bound
    # The problem states best known is 134, counting bound 112.
    # We will try from 112 upwards.
    
    # To handle potential timeouts on a specific scope, we can set a time limit per scope attempt
    # But the global deadline is the hard limit.
    # If we get stuck on scope 112 for 10s, we should move to 113?
    # Or just let it run until deadline?
    # Given the structure, if 112 is impossible, it might take a long time to prove.
    # However, for n=4, k=7, the search space is small enough that a good solver finds a solution quickly if it exists.
    # If it doesn't exist, it might take longer.
    # Let's add a time check inside the loop to switch scopes if we've spent too much time on one.
    
    current_scope = total_diffs # 112
    max_scope_to_try = 200 # Safety upper bound
    
    while current_scope <= max_scope_to_try:
        if time.time() > deadline - 5:
            break
            
        # Reset random seed? No, deterministic search is better for reproducibility and pruning.
        # But randomized restarts can help escape local minima in the search tree?
        # Backtracking is complete, so it will find a solution if one exists, eventually.
        # The order of candidates matters for speed.
        
        # We can randomize the candidate order to avoid bad paths.
        # Let's modify solve_with_scope to accept a random seed or use global random.
        
        # For now, let's just run it.
        # If it takes too long, we break.
        
        # To prevent a single scope from eating all time, we can't easily interrupt the recursive backtrack 
        # cleanly without global flags.
        # The `if time.time() > deadline: return False` inside backtrack handles the hard kill.
        # But we want to try the next scope if the current one is "slow".
        # Since we can't easily interrupt, we rely on the fact that if a solution exists at scope S, 
        # it will likely be found quickly. If not, we might time out.
        # To be safe, we can wrap the call in a way that checks time.
        
        # Let's just call it. If it returns True, we are done.
        # If it returns False, it either proved no solution (unlikely for small N without complete proof) 
        # or timed out (returned False due to time check).
        
        # Wait, if it times out, it returns False. Then we increment scope.
        # This is a good strategy: "Try scope S for up to X seconds. If fail, try S+1."
        # How to implement "up to X seconds"?
        # We can pass a time_limit to solve_with_scope.
        
        pass

    # Let's refactor solve_with_scope to accept a time_limit
    
    def solve_with_scope_limited(target_scope, time_limit):
        start_t = time.time()
        # ... same as above but check (time.time() - start_t > time_limit) ...
        
    # Actually, the global deadline is 140s.
    # If we spend 130s on scope 112 and fail, we have 10s left.
    # Better to allocate small chunks.
    
    scope = total_diffs
    while scope <= 200:
        if time.time() > deadline - 10:
            break
            
        # Give each scope attempt 5 seconds max
        # If it finds a solution, great.
        # If not, move to next scope.
        
        # We need a version of backtrack that respects a local time limit.
        # Let's just use the global deadline for the first few scopes, 
        # but if we are deep in the search, we might want to restart with a higher scope.
        
        # Simple approach:
        # Try scope. If it returns True, return the rows.
        # If it returns False, increment scope.
        # The `time.time() > deadline` check in backtrack ensures we stop before 150s.
        # The risk is spending all 140s on scope 112 and returning nothing.
        # To mitigate, we can add a "soft" time limit per scope.
        
        # Let's modify the backtrack to check a local start time.
        
        def solve_limited(target_scope, limit_seconds):
            start_t = time.time()
            rows = [[-1] * row_len for _ in range(n_rows)]
            for r in range(n_rows):
                rows[r][0] = 0
            used_diffs = set()
            
            def backtrack():
                if time.time() - start_t > limit_seconds:
                    return False
                if time.time() > deadline:
                    return False
                
                # MRV logic
                best_r = -1
                best_i = -1
                min_candidates = float('inf')
                best_cands = []
                
                for r in range(n_rows):
                    i = 1
                    while i < row_len and rows[r][i] != -1:
                        i += 1
                    if i == row_len:
                        continue
                        
                    prev_val = rows[r][i-1]
                    lower = prev_val + 1
                    upper = target_scope
                    
                    if lower > upper:
                        return False
                    
                    cands = []
                    for v in range(lower, upper + 1):
                        valid = True
                        for j in range(i):
                            d = v - rows[r][j]
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            cands.append(v)
                            
                    if len(cands) == 0:
                        return False
                        
                    if len(cands) < min_candidates:
                        min_candidates = len(cands)
                        best_r = r
                        best_i = i
                        best_cands = cands
                        if min_candidates == 1:
                            break
                
                if best_r == -1:
                    return True
                
                # Randomize candidate order to avoid deterministic bad paths
                random.shuffle(best_cands)
                
                for v in best_cands:
                    rows[best_r][best_i] = v
                    new_diffs = []
                    for j in range(best_i):
                        d = v - rows[best_r][j]
                        new_diffs.append(d)
                        used_diffs.add(d)
                        
                    if backtrack():
                        return True
                        
                    for d in new_diffs:
                        used_diffs.remove(d)
                    rows[best_r][best_i] = -1
                return False
                
            if backtrack():
                return rows
            return None

        # Try scopes
        # We'll try 112, 113, ... 
        # Give each 5 seconds? Or less?
        # If we find a solution at 112, we are done.
        # If not, we try 113.
        
        # To be efficient, we can try 112 with a short time limit, then 113, etc.
        # If we reach a scope where we find a solution, we return it.
        
        # Let's try a few scopes with short limits, then one with long limit.
        # Or just iterate.
        
        # Given the "best known 134", it's likely 112-133 are hard or impossible.
        # We should prioritize finding *a* valid solution quickly to ensure a score.
        # A valid solution with scope 140 is better than no solution.
        # So, we should start with a higher scope if we are worried about time?
        # No, the score is 1/scope (roughly, via seed/value). Lower scope is better.
        # So we MUST try low scopes first.
        
        # Strategy:
        # 1. Try scope 112 with 2s limit.
        # 2. Try scope 113 with 2s limit.
        # ...
        # 3. If we have time left, try the next scope with remaining time.
        
        # Let's implement this loop.
        
        current_scope = total_diffs
        found_rows = None
        
        while current_scope <= 200:
            if time.time() > deadline - 5:
                break
                
            # Determine time limit for this scope
            # If this is the first few scopes, use short time.
            # If we are near the end of time, use remaining.
            
            elapsed = time.time() - start_time
            remaining = deadline - 5 - elapsed
            
            if remaining <= 0:
                break
            
            # Heuristic: allocate 2s for scopes < 130, and the rest for the final attempt
            # Or just iterate.
            # Let's allocate min(2.0, remaining) seconds per scope attempt.
            
            limit = min(2.0, remaining)
            
            res = solve_limited(current_scope, limit)
            
            if res:
                found_rows = res
                break
                
            current_scope += 1
            
        if found_rows:
            return found_rows
            
        # If we didn't find anything in the loop (should not happen if we tried high enough scopes),
        # fall back to a greedy construction to ensure we return *something* valid.
        
        def greedy_init():
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < row_len:
                    last = row[-1]
                    x = last + 1
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

        return greedy_init()

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
