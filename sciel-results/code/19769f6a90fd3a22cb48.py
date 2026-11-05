import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for 120s limit
    
    # Precompute max possible difference? 
    # If scope is S, max diff is S.
    # We can use a bitmask if S is small enough.
    # 112 is small. 105 is small.
    # Let's assume scope <= 200 for bitmask efficiency.
    MAX_SCOPE_ESTIMATE = 256
    MASK_SIZE = 1 << MAX_SCOPE_ESTIMATE
    
    # Helper to check time
    def time_up():
        return time.time() - start_time > time_limit

    # State:
    # used_diffs: bitmask of used differences
    # rows: list of rows completed so far
    # current_row: list of elements in current row
    
    best_result = None
    best_scope = float('inf')
    
    # We will try to find a solution with scope <= target_scope
    # We use backtracking with randomized tie-breaking to explore different paths.
    
    def solve_attempt(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = 0
        rows = []
        
        def _backtrack(row_idx, current_row, current_max_val):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                # All rows constructed
                # Check if we found a better solution
                # Since we enforce x < target_scope, current_max_val < target_scope
                # But we want the actual max.
                # Actually, if we complete all rows, the scope is max(r[-1] for r in rows)
                # But we are building row by row.
                # The current_max_val is the max of all elements seen so far.
                # Since rows are built sequentially and we don't backtrack previous rows' maxes easily in this structure,
                # let's just calculate it.
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0], current_max_val):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Upper bound for next element x
            # x must be > last_val
            # x must be < target_scope (strictly, because if x == target_scope, scope is at least target_scope,
            # but we are looking for scope < target_scope? 
            # The task says "largest entry as small as possible".
            # If we set target_scope = S, we are looking for a solution with max entry < S.
            # So x < S.
            upper_bound = target_scope - 1
            
            # Also, if we have a best_scope, we can prune if x >= best_scope
            # But we are trying to find a solution with scope < target_scope.
            # If target_scope <= best_scope, we are trying to improve.
            # If we find a solution, best_scope will be updated.
            
            # Generate candidates
            # To optimize, we can limit the search space.
            # If best_scope is known, we can prune if x >= best_scope?
            # No, if we are searching for scope < target_scope, and target_scope < best_scope,
            # then any solution found will be < best_scope.
            
            candidates = []
            
            # Heuristic:
            # We want to prefer candidates that use "large" differences (low scarcity).
            # Cost = sum(1.0 / d) for d in new_diffs.
            # We also want to keep x small.
            
            for x in range(last_val + 1, upper_bound + 1):
                # Check if new differences conflict with used_diffs
                # new_diffs = {x - a for a in current_row}
                # We need to check if any (x - a) is in used_diffs.
                # Also, we need to check if (x - a) are distinct among themselves?
                # Yes, because a are distinct, x - a are distinct.
                
                # Bitmask check
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d >= MAX_SCOPE_ESTIMATE:
                        # Should not happen if target_scope is small
                        continue
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                # Calculate cost
                # We can approximate cost by summing 1/d
                # To make it faster, we can just use the smallest d as a proxy?
                # Or sum of 1/d is fine for small k.
                cost = 0.0
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                
                # Add small random noise to break ties and allow exploration
                noise = rng.random() * 0.01
                
                candidates.append((cost + noise, x, new_mask))
                
                # Limit candidates to avoid explosion
                # If we have too many, we might want to pick the best few.
                # But for now, we break after a certain number to keep it fast.
                if len(candidates) > 20:
                    break
            
            if not candidates:
                return False
            
            # Sort candidates
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates
            # Randomize the order of the top candidates to explore different branches
            # We can shuffle the top 5 or so.
            top_n = min(5, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining_candidates = candidates[top_n:]
            
            for cost, x, new_mask in top_candidates + remaining_candidates:
                # Recurse
                old_used = used_diffs
                used_diffs = used_diffs | new_mask
                
                if _backtrack(row_idx, current_row + [x], max(current_max_val, x)):
                    return True
                
                # Restore state
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0], 0)

    # Strategy:
    # 1. Start with a high target_scope (e.g., 200).
    # 2. Run solve_attempt.
    # 3. If a solution is found, update best_scope.
    # 4. Set target_scope = best_scope. (We want to find a solution with scope < best_scope)
    # 5. Repeat with different random seeds until time runs out or no improvement.
    
    current_target = 200
    seed_counter = 0
    
    while time.time() - start_time < time_limit:
        # If we have a best solution, try to improve it
        if best_result is not None:
            # Try to find a solution with scope < best_scope
            # So target_scope = best_scope
            # If we find one, best_scope will be updated.
            target = best_scope
        else:
            # First attempt, use a large target
            target = current_target
            
        found = solve_attempt(target, seed_counter)
        
        if found:
            # If we found a solution, best_scope is updated inside solve_attempt
            # If best_scope didn't change, it means we found a solution with the same scope?
            # Or we found a solution but it wasn't better?
            # If best_result was None, it is now set.
            # If best_result was set, and we found a solution with scope < best_scope, best_scope is updated.
            # If we found a solution with scope >= best_scope, best_scope is NOT updated.
            # In that case, we should try a tighter target?
            # No, if we targeted best_scope, we are looking for scope < best_scope.
            # If we didn't find one, we should try again with a different seed.
            
            # To avoid infinite loop if we keep finding the same best solution,
            # we can check if best_scope improved.
            # But solve_attempt returns True if it finds ANY valid solution with scope < target.
            # If target = best_scope, it must find scope < best_scope to update best_scope.
            # If it returns True but best_scope didn't change, that's impossible?
            # Wait, solve_attempt updates best_scope if actual_scope < best_scope.
            # If it returns True, it means it completed all rows.
            # The actual_scope is max(r[-1] for r in rows).
            # Since x < target, actual_scope < target.
            # If target = best_scope, then actual_scope < best_scope.
            # So best_scope MUST be updated if it returns True and target == best_scope.
            
            # So if it returns True, best_scope is strictly smaller.
            # We can continue searching for even smaller.
            
            # If it returns False, we failed to find a solution with scope < best_scope.
            # We can try again with a different seed.
            # If we fail many times, we might assume it's optimal?
            # But we have time, so keep trying.
            
            pass
        else:
            # Failed to find a solution with scope < target
            # If we didn't have a best result, we need to increase target?
            # No, we started with 200. If we fail, maybe 200 is too small?
            # But 112 is known. 200 should be enough.
            # If we fail, it's likely due to search difficulty.
            # Try again with a different seed.
            
            pass
            
        seed_counter += 1
        
        # If we have a best result and we've tried a few times to improve without success,
        # we might want to stop? Or just keep going until time limit.
        # The problem is small, so keep going.

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
