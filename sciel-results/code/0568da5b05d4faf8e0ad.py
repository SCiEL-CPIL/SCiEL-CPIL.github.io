import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Helper to check if a row is a valid Golomb ruler (distinct internal diffs)
    # Note: The row must start at 0 and be increasing.
    def is_valid_golomb(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Helper to generate a random Golomb ruler with max value <= M
    # Uses randomized greedy: at each step, pick a random valid next number from a small window of the smallest valid options
    def generate_random_row(M, used_diffs_global=None, seed=None):
        if seed is not None:
            random.seed(seed)
        
        row = [0]
        used_local = set()
        
        while len(row) < k + 1:
            last = row[-1]
            # Candidate next values must be > last and <= M
            # We look at a window of candidates to allow randomness but keep them small
            # The window size can be dynamic or fixed. Let's try a fixed small window or all valid up to some limit.
            # To ensure we find a row, we should consider the smallest valid candidates first.
            
            candidates = []
            # We can't iterate up to M if M is large and k is small, but k=6 is small.
            # However, M can be ~100. Iterating 100 candidates is fine.
            # Let's collect the first few valid candidates.
            
            found_any = False
            # We want to pick from the smallest valid options to keep the ruler compact
            # But to have variety, we pick randomly from the first N valid options
            N_OPTIONS = 10 
            
            for x in range(last + 1, M + 1):
                # Check if adding x creates duplicate diffs within this row
                valid_local = True
                for a in row:
                    d = x - a
                    if d in used_local:
                        valid_local = False
                        break
                    if used_diffs_global is not None and d in used_diffs_global:
                        valid_local = False
                        break
                if valid_local:
                    candidates.append(x)
                    found_any = True
                    if len(candidates) >= N_OPTIONS:
                        break
            
            if not found_any:
                return None # Failed to extend row
            
            # Pick a random candidate from the top N
            next_val = random.choice(candidates)
            row.append(next_val)
            # Update used_local
            for a in row[:-1]:
                used_local.add(next_val - a)
                
        return row

    # Initial Greedy to get an upper bound
    def greedy_construct(M_limit):
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            found = False
            while len(row) < k + 1:
                x += 1
                if x > M_limit:
                    break
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                    found = True
            if not found:
                return None
            rows.append(row)
        return rows

    # Get initial upper bound
    # Start with a reasonable guess, e.g., k*(k+1)/2 * n_rows is too high.
    # Simple greedy without limit
    current_best = greedy_construct(1000) 
    if current_best is None:
        # Fallback to very large
        current_best = greedy_construct(10000)
        
    initial_max = max(max(r) for r in current_best)
    
    # Now try to improve by searching for smaller M
    best_result = current_best
    best_max = initial_max
    
    # Search downwards
    # We will try M from initial_max - 1 down to a lower bound
    # Lower bound: counting bound is n_rows * k * (k+1) / 2 ? No, differences are distinct.
    # Total differences = n_rows * k(k+1)/2.
    # Max difference is M. So M(M+1)/2 >= n_rows * k(k+1)/2.
    # M ~ sqrt(n_rows * k(k+1))
    # For n=7, k=6: 7 * 21 = 147. sqrt(147) ~ 12. This is a lower bound on M? No, M is the max value.
    # The number of distinct differences available in [1, M] is M(M+1)/2.
    # We need n_rows * k(k+1)/2 distinct differences.
    # So M(M+1)/2 >= n_rows * k(k+1)/2.
    
    import math
    lower_bound = int(math.ceil((math.sqrt(8 * n_rows * k * (k+1) + 1) - 1) / 2))
    
    # We iterate M downwards from initial_max - 1 to lower_bound
    # But doing this for every M might be slow.
    # We can use binary search? No, the property "exists solution for M" is monotonic.
    # If a solution exists for M, it exists for M+1.
    # So we can binary search the minimum M!
    
    def exists_solution(M, time_limit):
        # Randomized backtracking search for a solution with max value <= M
        # We use a timeout guard
        start = time.time()
        
        used_global = set()
        rows = []
        
        # We use a recursive solver with randomness and backtracking
        # To avoid deep recursion overhead, we can use a stack or just recursion with sys.setrecursionlimit
        sys.setrecursionlimit(10000)
        
        def dfs(row_idx):
            if time.time() > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Generate candidates for this row
            # We want to try candidates that are "small" first
            # Generate a batch of random candidates
            # To ensure coverage, we generate multiple random rulers
            
            # Strategy: Generate 50 random rows with max <= M
            # Filter those that don't conflict with used_global
            # Sort by max value
            # Try them in order
            
            candidates = []
            for _ in range(100): # Number of random attempts per node
                r = generate_random_row(M, used_global)
                if r:
                    # Check if it really doesn't conflict (generate_random_row already checks against used_global passed in? 
                    # Yes, I passed used_diffs_global)
                    # But wait, generate_random_row uses used_diffs_global to prune.
                    # So r is valid against current used_global.
                    candidates.append(r)
            
            if not candidates:
                return False
            
            # Sort by max element to prefer compact rows
            candidates.sort(key=lambda x: x[-1])
            
            # Deduplicate candidates to avoid trying the same row twice
            unique_candidates = []
            seen = set()
            for r in candidates:
                if tuple(r) not in seen:
                    seen.add(tuple(r))
                    unique_candidates.append(r)
            
            for r in unique_candidates:
                # Add r to used_global
                new_diffs = set()
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        new_diffs.add(r[j] - r[i])
                
                # Check conflict again just in case (though generate_random_row should have handled it)
                if new_diffs & used_global:
                    continue
                
                used_global.update(new_diffs)
                rows.append(r)
                
                if dfs(row_idx + 1):
                    return True
                
                # Backtrack
                rows.pop()
                used_global.difference_update(new_diffs)
                
                if time.time() > time_limit:
                    return False
            
            return False
        
        if dfs(0):
            return True, rows[:]
        else:
            return False, None

    # Binary Search for the smallest M
    lo = lower_bound
    hi = initial_max
    
    # We have ~100s left.
    # Binary search takes log2(hi-lo) steps.
    # Each step runs a search with a time limit.
    # Let's allocate time dynamically.
    
    # First, verify that hi is valid (it is, from greedy).
    # We want to find the smallest M in [lo, hi].
    
    # To be safe and simple, let's do a linear search downwards from hi-1, but with a time budget per M.
    # Or better, binary search.
    
    # Let's use Binary Search.
    # Time budget for each check: proportional to remaining time / remaining steps?
    # Simpler: Give each check a fixed fraction of the remaining time.
    
    while lo < hi:
        if time.time() > deadline - 5:
            break
            
        mid = (lo + hi) // 2
        remaining = deadline - time.time()
        # Allocate 50% of remaining time to this check, keeping some for the next
        # Or just a fixed time like 5s? 
        # If hi-lo is large, binary search is fast.
        # Let's use a time limit of max(1, remaining / 10) seconds for the search?
        # Actually, the search might be very fast if M is too small (fails quickly) or too large (succeeds quickly).
        # The hardest cases are near the boundary.
        
        check_time = 5.0 # 5 seconds per check
        if remaining < check_time:
            check_time = remaining * 0.8
            
        success, res = exists_solution(mid, time.time() + check_time)
        
        if success:
            hi = mid
            if res:
                # Update best result if this M is better
                # Note: res is a valid solution for mid
                # We want the one with smallest max entry.
                # Since we are searching for the smallest M, any valid solution for M=mid is better than any for M>mid.
                # But wait, the "score" is the max entry of the rows.
                # The rows in `res` have max entry <= mid.
                # It might be strictly less than mid.
                # We should track the actual max entry of the found solution.
                actual_max = max(max(r) for r in res)
                if actual_max < best_max:
                    best_max = actual_max
                    best_result = res
        else:
            lo = mid + 1
        
        # If we found a solution, we can update best_result
        # But we need to ensure we return the best one found.
        
    # Final check: if best_result is not updated, use the initial greedy one
    # Also, try to polish the best_result?
    # The binary search finds a valid set for M=hi.
    # The set returned by exists_solution is valid.
    
    # One more polish: try to reduce the max value of the found rows?
    # The rows are already constrained to <= M.
    # We can try to replace rows with smaller ones.
    # But the binary search already minimized M.
    # The actual max value in the returned rows might be less than M.
    
    return best_result
