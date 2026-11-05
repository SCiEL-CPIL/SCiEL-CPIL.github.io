import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: 
    1. Determine a tight lower bound for the scope M.
    2. Use a randomized backtracking search (DFS) with MRV (Minimum Remaining Values) 
       heuristic to construct valid rows sequentially.
    3. If a valid set is found, attempt to lower M and repeat.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs_per_row = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs_per_row
    
    # Theoretical lower bound: We need total_diffs_needed distinct differences in [1, M]
    # So M >= total_diffs_needed.
    lower_bound = total_diffs_needed
    
    # Helper to calculate all differences in a row
    def get_row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    # Helper to check if a new value x added to row conflicts with existing used_diffs
    # and returns the new set of diffs to add
    def get_new_diffs(row, x):
        return [x - y for y in row]

    # Backtracking search to construct n_rows valid rows given a max scope M
    # used_diffs: set of differences already used
    # current_rows: list of completed rows
    def backtrack(rows_completed, used_diffs, M, rng):
        if len(rows_completed) == n_rows:
            return rows_completed
        
        # Heuristic: MRV (Minimum Remaining Values)
        # We are constructing rows one by one. 
        # To make it smarter, we could pick which row to construct next, 
        # but since rows are indistinguishable in validity, we just construct the next one.
        # However, we can randomize the order of elements within the row construction.
        
        row = [0]
        
        # We need to pick k more numbers from [1, M]
        # To ensure efficiency, we pick candidates that don't immediately cause conflicts.
        
        # Strategy for row construction:
        # Pick the next element. To keep the row "tight" (small scope), we should prefer smaller values,
        # but we must ensure we don't get stuck.
        # A good heuristic is to try values in a randomized order biased towards smaller values,
        # but also ensuring that the remaining space is sufficient to complete the row.
        
        # Let's use a recursive helper for the single row construction
        def build_row(current_row, used_diffs_local, depth):
            if depth == num_elements:
                return current_row
            
            last_val = current_row[-1]
            # Candidates must be > last_val and <= M
            # Also, the difference (candidate - last_val) and (candidate - any prev) must be in used_diffs_local
            
            # Estimate minimum required value to finish the row?
            # If we have d elements left, we need at least d * (min_gap) space.
            # min_gap is 1. So we need M - last_val >= d.
            if M - last_val < (num_elements - depth):
                return None
            
            # Generate candidates
            # To improve MRV-like behavior, we can check how many future options a candidate leaves.
            # But that's expensive. Let's just shuffle candidates but bias towards smaller ones 
            # to keep the scope small, while allowing some randomness to escape local minima.
            
            # Range of candidates: last_val + 1 to M
            # If we pick too large a number early, we might not have enough room for subsequent elements?
            # Actually, if we pick x, the next must be > x.
            
            # Let's try candidates in a randomized order, but with a slight preference for smaller values
            # to encourage compact rows.
            candidates = list(range(last_val + 1, M + 1))
            
            # Shuffle but keep small values somewhat prioritized? 
            # Or just random shuffle. Random shuffle is safer for global search.
            rng.shuffle(candidates)
            
            # Optimization: Filter candidates that immediately conflict
            valid_candidates = []
            for x in candidates:
                new_diffs = [x - y for y in current_row]
                if not any(d in used_diffs_local for d in new_diffs):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None
            
            # Try each valid candidate
            for x in valid_candidates:
                new_diffs = [x - y for y in current_row]
                new_used = used_diffs_local.union(new_diffs)
                
                # Check if we can still complete the row with remaining elements
                # Simple check: M - x >= (num_elements - depth - 1)
                if M - x < (num_elements - depth - 1):
                    continue
                
                res = build_row(current_row + [x], new_used, depth + 1)
                if res is not None:
                    return res
            return None
        
        new_row = build_row(row, used_diffs, 1)
        if new_row is None:
            return None
        
        new_used_diffs = used_diffs.union(get_row_diffs(new_row))
        new_rows = rows_completed + [new_row]
        
        return backtrack(new_rows, new_used_diffs, M, rng)

    # Main search loop
    # Start from the lower bound and increase M until a solution is found or time runs out.
    # Since the problem is hard, we might need to restart with different random seeds for a fixed M.
    
    best_solution = None
    best_M = float('inf')
    
    # We will iterate M from lower_bound upwards.
    # For each M, we try multiple random restarts.
    
    current_M = lower_bound
    
    # To avoid getting stuck in a bad local structure for a specific M, 
    # we limit the number of restarts per M and then increment M.
    
    # However, if M is close to the lower bound, the space is very tight.
    # If M is larger, the space is looser.
    
    # Let's define a budget for restarts.
    max_restarts_per_M = 20
    
    while current_M <= 200:  # Cap at 200 to be safe, though 112 is expected
        if time.time() > DEADLINE - 5:
            break
            
        found = False
        for restart in range(max_restarts_per_M):
            if time.time() > DEADLINE - 2:
                break
                
            rng = random.Random(current_M * 1000 + restart)
            
            # Initial state: no rows, no used diffs
            result = backtrack([], set(), current_M, rng)
            
            if result is not None:
                # Verify the solution
                all_diffs = []
                for r in result:
                    all_diffs.extend(get_row_diffs(r))
                
                if len(all_diffs) == len(set(all_diffs)):
                    # Valid solution
                    if current_M < best_M:
                        best_M = current_M
                        best_solution = result
                    found = True
                    break
        
        if found:
            # Found a solution for this M. 
            # Can we go lower? No, we started from lower_bound and went up.
            # So this is the minimal M for this specific search path.
            # However, it's possible that a solution exists for a lower M but we missed it due to randomization?
            # No, we tried all M from lower_bound up to current_M. 
            # If we found one at current_M, and we tried all lower M's (with restarts), 
            # it's highly likely this is the minimum or close to it.
            # We can return this immediately as it's a valid upper bound.
            # To be more rigorous, we could continue searching for a lower M, 
            # but we already searched lower M's.
            # So we can break.
            break
            
        current_M += 1
        
        # If we are taking too long per M, we might want to reduce restarts or increase M faster
        # But for now, linear increment is fine.

    if best_solution is not None:
        return best_solution
    
    # Fallback: If no solution found within the loop (e.g., time out or M cap reached),
    # return the last valid solution found or a greedy one.
    # Since we increment M, if we didn't break, it means we didn't find one in the loop?
    # Actually, if time runs out, we might have a partial best_solution.
    # If best_solution is still None, we need a fallback.
    
    # Greedy fallback: Construct rows with increasing M until valid.
    # This is unlikely to be reached if the loop works, but good for safety.
    rows_fb = []
    used = set()
    M = lower_bound
    for _ in range(n_rows):
        row = [0]
        # Greedy pick smallest available
        while len(row) < num_elements:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    break
                x += 1
                if x > 500:
                    # Emergency break
                    row.append(x)
                    used.update(new_diffs)
                    break
            row.append(x)
            used.update({x - a for a in row})
        rows_fb.append(row)
        
    return rows_fb

# EVOLVE-BLOCK-END
