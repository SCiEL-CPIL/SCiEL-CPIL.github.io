import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: Difference-space CSP with Randomized Backtracking.
    1. Determine the lower bound (LB) for the scope.
    2. Iterate target_scope from LB upwards.
    3. For each target_scope, attempt to construct the set using randomized backtracking.
       - Variables: The elements of the rows (excluding the leading 0).
       - Domains: Values in (prev_val, target_scope].
       - Constraints: 
         a) Monotonicity (handled by domain).
         b) Global distinctness of all pairwise differences.
         c) Local consistency within a row (differences must be unique within the row too, 
            which is implied by global distinctness).
       - Heuristics:
         - Order rows/positions to maximize constraints early? 
           Actually, standard backtracking on rows is fine.
         - Randomize candidate order to escape local minima.
         - Forward checking: Maintain a set of `used_diffs`. When placing a value, 
           check against `used_diffs` and local row diffs.
    
    This approach directly searches the solution space for the minimal scope, 
    avoiding the "construct then refine" plateau that value-space SA often hits.
    """
    
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # The largest difference in any row is at most scope.
    # The differences are distinct integers from 1 to scope.
    # We need total_diffs_needed distinct differences.
    # So scope >= total_diffs_needed.
    lower_bound = total_diffs_needed
    
    # We will try to find a solution for scope = LB, LB+1, ...
    # To handle the case where a specific scope is hard but a slightly larger one is easy,
    # we iterate.
    
    # Optimization: 
    # The "counting bound" is total_diffs_needed.
    # For (7,5), LB = 7 * 15 = 105.
    # Best known is 112.
    # So we expect to find it between 105 and 112.
    
    # We'll use a randomized backtracking solver.
    # To avoid getting stuck in one "basin", we'll do multiple restarts for each scope
    # if the first few attempts fail quickly.
    
    def _solve_for_scope(target_scope, rng, time_limit):
        """
        Attempt to construct a valid set with max element <= target_scope.
        Returns list of rows if found, None otherwise.
        """
        rows = []
        used_diffs = set()
        
        # We construct rows one by one.
        # For each row, we use backtracking to find a valid set of values.
        
        def build_row(pos, last_val, current_row):
            if time.time() > time_limit:
                return False
            
            if pos == num_elements:
                return True
            
            # Generate candidates: last_val + 1 to target_scope
            # To improve speed, we can randomize the order.
            # However, for minimizing scope, we want to find *any* valid row.
            # Randomization helps escape dead ends.
            
            candidates = list(range(last_val + 1, target_scope + 1))
            rng.shuffle(candidates)
            
            # Heuristic: Try values that are "spread out" or "dense"?
            # Random is a good baseline.
            
            for val in candidates:
                # Check local differences
                local_diffs = []
                valid = True
                for i in range(len(current_row)):
                    d = val - current_row[i]
                    if d in used_diffs:
                        valid = False
                        break
                    if d in local_diffs:
                        valid = False
                        break
                    local_diffs.append(d)
                
                if valid:
                    current_row.append(val)
                    used_diffs.update(local_diffs)
                    
                    if build_row(pos + 1, val, current_row):
                        return True
                    else:
                        current_row.pop()
                        for d in local_diffs:
                            used_diffs.discard(d)
            
            return False

        # Main loop for rows
        for r in range(n_rows):
            if time.time() > time_limit:
                return None
            
            current_row = [0]
            if build_row(1, 0, current_row):
                rows.append(current_row)
            else:
                return None
        
        return rows

    best_rows = None
    best_scope = float('inf')
    
    # Iterate over possible scopes
    # We don't know exactly how far above LB we need to go.
    # Let's try up to LB + 50 to be safe, but break early if found.
    
    max_scope_to_try = lower_bound + 50
    
    for scope in range(lower_bound, max_scope_to_try + 1):
        if time.time() > DEADLINE - 10:
            break
            
        # Try multiple random seeds for this scope to increase success probability
        # The search space is large, so a single deterministic pass might fail.
        found = False
        num_attempts = 5
        if scope <= lower_bound + 5:
            num_attempts = 10  # More attempts for tighter scopes
        
        for attempt in range(num_attempts):
            if time.time() > DEADLINE - 10:
                break
                
            rng = random.Random(hash((scope, attempt, n_rows, k)) % (2**32))
            
            # Time limit per attempt: 
            # We want to try several scopes. 
            # If a scope is hard, we shouldn't spend all time on it.
            # Let's allocate a small time slice per attempt.
            # Total time ~ 140s. 
            # If we try 10 scopes * 10 attempts = 100 attempts.
            # 140s / 100 = 1.4s per attempt.
            # This is very tight.
            
            # Better strategy: 
            # Spend more time on the first few scopes (LB, LB+1, LB+2).
            # If they fail, expand time or move on?
            # Given the problem structure, if LB is infeasible, LB+1 might be feasible.
            
            # Let's use a dynamic time limit.
            # Start with 5s per attempt for the first scope.
            # Decrease if we move to higher scopes?
            # Or just use a fixed small time and rely on many restarts?
            
            # Let's try 2 seconds per attempt.
            time_limit = time.time() + 2.0
            
            rows = _solve_for_scope(scope, rng, time_limit)
            if rows is not None:
                # Verify validity just in case
                all_diffs = []
                for row in rows:
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            all_diffs.append(row[j] - row[i])
                if len(all_diffs) == len(set(all_diffs)):
                    s = max(r[-1] for r in rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = rows
                    found = True
                    break # Found a solution for this scope, try next (tighter) scope?
                    # Actually, we are iterating upwards. 
                    # If we found one at scope S, we should check if we can find one at S-1?
                    # No, we start at LB. So the first one found is the minimum scope 
                    # *among those we tried*. Since we go up, the first success is optimal 
                    # for the range we've searched.
                    # But wait, we might have failed at LB due to bad luck, but a solution exists.
                    # So we should try LB again with more time?
                    # For now, if we find one, we can stop if we are confident.
                    # But to be safe, if we find one at scope S, we should try to see if S-1 is possible 
                    # if we haven't exhausted time.
                    # However, the loop goes LB, LB+1...
                    # So if we find at LB+2, we've already tried LB and LB+1 and they failed (or timed out).
                    # So this is the best we found.
        
        if found:
            # We found a solution for this scope.
            # Should we continue to see if a smaller scope was possible but we missed it?
            # Since we iterate from LB upwards, and we tried multiple attempts for each,
            # it is unlikely that a smaller scope exists but we missed it in 5-10 attempts.
            # However, to be rigorous, if we find a solution at scope S, 
            # and S > LB, we could optionally try to refine S down to LB with more time.
            # But given the 150s limit, and that we've already spent time on lower scopes,
            # let's assume the first success is good enough.
            # Actually, let's just break. The probability of missing a feasible LB after 10 attempts
            # is low if the search is good.
            break
            
    if best_rows is None:
        # Fallback: Construct a valid set with a larger scope using a simple greedy method
        # that is guaranteed to find a solution (though not minimal).
        # We use the "sequential greedy" where we pick the smallest possible next value.
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            # We need to pick k more numbers.
            # To ensure we can finish, we might need a large scope.
            # Let's just pick numbers 1, 2, ... but skipping those that cause conflicts.
            # This might lead to a very large scope, but it's valid.
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows) if hasattr(self, '_scope') else max(r[-1] for r in rows)
        
    return best_rows

# Helper for fallback scope calculation if needed
def _scope(rows):
    return max(r[-1] for r in rows)

# EVOLVE-BLOCK-END
