import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 118.0  # Leave buffer for 120s limit

    def time_up():
        return time.time() - start_time > time_limit

    # We need to construct n_rows rows, each with k+1 elements starting at 0.
    # The differences a_j - a_i for i < j must be all distinct across all rows.
    # Total differences per row: k*(k+1)/2.
    # Total differences overall: n_rows * k*(k+1)/2.
    # If the scope is S (max value), the possible differences are 1..S.
    # So we need S >= n_rows * k*(k+1)/2.
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope
    min_scope = total_diffs
    
    # We use backtracking to construct rows one by one.
    # State:
    # - used_diffs: set of used differences (or bitmask if S is small)
    # - rows: list of completed rows
    # - current_row: list of elements in the current row being built
    
    # For efficiency, we'll use a set for used_diffs since S can be up to ~200.
    # Actually, a bitmask is faster for membership testing.
    # Let's use a set for simplicity and clarity, but optimize if needed.
    # Given the small size (S <= 200), a set is fine.
    
    best_result = None
    best_scope = float('inf')
    
    # Greedy initialization to get a good starting point
    def greedy_construct():
        nonlocal best_result, best_scope
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Find the smallest x > row[-1] such that {x - a for a in row} is disjoint from used
                x = row[-1] + 1
                while True:
                    new_diffs = set(x - a for a in row)
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used |= new_diffs
            rows.append(row)
        
        scope = max(r[-1] for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_result = [r[:] for r in rows]
        return rows, used

    # Run greedy first
    _, _ = greedy_construct()
    
    # Now try to improve using backtracking with restarts
    # We'll try to find a solution with scope < best_scope
    
    def solve_with_scope(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                # All rows constructed
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Upper bound for next element x
            # x must be > last_val
            # x must be < target_scope (strictly, because if x == target_scope, scope is at least target_scope)
            # Actually, if we're looking for scope < target_scope, then x < target_scope
            # But if we're looking for scope <= target_scope, then x <= target_scope
            # Let's say we're looking for scope < target_scope, so x < target_scope
            upper_bound = target_scope - 1
            
            if last_val + 1 > upper_bound:
                return False
            
            # Generate candidates
            # Heuristic: prefer candidates that use "large" differences (low scarcity)
            # and keep x small
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                # Check if new differences conflict with used_diffs
                new_diffs = set(x - a for a in current_row)
                if new_diffs & used_diffs:
                    continue
                
                # Calculate cost: sum of 1/d for d in new_diffs
                # This penalizes using small differences
                cost = sum(1.0 / d for d in new_diffs)
                
                # Add small random noise to break ties
                noise = rng.random() * 0.01
                
                candidates.append((cost + noise, x, new_diffs))
                
                # Limit candidates to avoid explosion
                if len(candidates) > 15:
                    break
            
            if not candidates:
                return False
            
            # Sort candidates by cost
            candidates.sort(key=lambda item: item[0])
            
            # Try top candidates
            # Randomize the order of the top few to explore different branches
            top_n = min(8, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining_candidates = candidates[top_n:]
            
            for cost, x, new_diffs in top_candidates + remaining_candidates:
                # Recurse
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                # Restore state
                used_diffs = old_used
            
            return False
        
        return backtrack(0, [0])

    # Strategy:
    # 1. Start with target_scope = best_scope (from greedy)
    # 2. Try to find a solution with scope < best_scope
    # 3. If found, update best_scope and repeat
    # 4. If not found after several attempts, try a larger target_scope? No, that doesn't make sense.
    #    Actually, if we can't find a solution with scope < best_scope, we should keep trying with different seeds.
    #    If we fail many times, we might assume it's optimal or close to optimal.
    
    seed_counter = 0
    consecutive_failures = 0
    max_consecutive_failures = 50
    
    while time.time() - start_time < time_limit:
        # Try to find a solution with scope < best_scope
        target = best_scope
        
        found = solve_with_scope(target, seed_counter)
        
        if found:
            # If we found a solution, best_scope is updated inside solve_with_scope
            # Reset consecutive failures
            consecutive_failures = 0
        else:
            # Failed to find a solution with scope < best_scope
            consecutive_failures += 1
            
            # If we've failed many times, we might want to stop
            # But we have time, so keep trying
            # However, if we've failed max_consecutive_failures times, we might assume it's optimal
            # Let's keep trying but with a cap
            if consecutive_failures >= max_consecutive_failures:
                # Try a different strategy: increase the target slightly to see if we can find any solution
                # Actually, no. If we can't find a solution with scope < best_scope, 
                # increasing the target won't help because we already have a solution with scope = best_scope.
                # We want to find a better solution.
                # Let's just keep trying with different seeds.
                pass
        
        seed_counter += 1
    
    if best_result is None:
        # Fallback to greedy if backtracking failed completely (unlikely)
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return best_result
