"""Difference triangle set solver using Stochastic CSP with Scope Targeting."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_set(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy Baseline ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)
    
    # If time is tight, return immediately
    if time_left() < 10:
        return best_rows

    # ---------- Stochastic CSP Solver for Fixed Scope ----------
    # Tries to find n_rows rows of k+1 ints in [0, limit] with distinct diffs.
    # Returns list of rows if found, else None.
    def solve_fixed_scope(limit, n_restarts=50, time_per_restart=0.5):
        """
        Attempts to construct n_rows valid rows with max entry <= limit.
        Uses randomized backtracking for each row.
        """
        for _ in range(n_restarts):
            if time_left() < 5:
                return None
            
            # Randomize the order in which we assign rows to break symmetry
            # but we must return them in a consistent order for the final result?
            # No, the set of rows is unordered. We just need the collection.
            # However, for the backtracking to be efficient, we might want to 
            # place "harder" rows first. Let's just shuffle.
            row_indices = list(range(n_rows))
            random.shuffle(row_indices)
            
            current_rows = [None] * n_rows
            used_diffs = set()
            success = True
            
            for idx in row_indices:
                # Try to find a row for position idx
                # Constraint: max entry <= limit
                # Forbidden diffs: used_diffs
                row = find_row_backtracking(k, limit, used_diffs, time_budget=time_per_restart)
                if row is None:
                    success = False
                    break
                current_rows[idx] = row
                # Update used diffs
                used_diffs |= row_diffs_set(row)
            
            if success:
                # Verify validity just in case
                if is_valid(current_rows):
                    return current_rows
        
        return None

    def find_row_backtracking(k, limit, forbidden_diffs, time_budget=0.5):
        """
        Finds a Sidon set of size k+1 in [0, limit] avoiding forbidden_diffs.
        Uses depth-first search with randomization.
        """
        start_t = time.time()
        # We need k+1 numbers: x_0=0, x_1, ..., x_k
        # x_0 is fixed at 0.
        # We need to choose x_1 < ... < x_k <= limit.
        
        # To make it faster, we can use a greedy approach with backtracking.
        # State: (current_row, next_index, last_value)
        # We'll use a recursive helper or iterative stack.
        
        # Optimization: Precompute valid candidates for each position?
        # That's dynamic.
        
        # Heuristic: Try to place numbers as small as possible, but with some randomness.
        
        def backtrack(row, next_idx, last_val):
            nonlocal time_budget
            if time.time() - start_t > time_budget:
                return None
            
            if next_idx == k + 1:
                return list(row)
            
            # Determine range for next value
            # Must be > last_val and <= limit
            # Also, if next_idx == k (last element), it must be <= limit.
            
            # To improve success rate, we can look ahead or just try candidates.
            # Candidates: integers from last_val + 1 to limit.
            # To avoid huge branching, we can sample or limit the window.
            # However, for k=7, the depth is small.
            
            # Strategy: Try a few small candidates, then some random ones.
            # Or, just iterate from last_val+1 to limit, but with a "step" or randomization.
            
            # Let's try a randomized greedy approach first:
            # Pick a random valid candidate? No, that might miss tight solutions.
            # Better: Iterate candidates, but shuffle the order of "close" candidates?
            
            # Simple DFS with pruning:
            # For x in range(last_val + 1, limit + 1):
            #   Check conflicts with current row and forbidden_diffs.
            #   If ok, recurse.
            
            # This can be slow if limit is large.
            # Optimization: If we are near the limit, we must be careful.
            
            # Let's use a "window" approach: try the next 10 candidates, then jump?
            # Actually, for DTS, the gaps are usually small.
            
            # Let's try candidates in increasing order, but with a random restart inside?
            # No, backtracking handles that.
            
            # To speed up, we can pre-check: 
            # The remaining elements must fit in [x, limit].
            
            # Let's just do standard DFS but with a time check.
            # To avoid worst-case, we can randomize the starting point of the loop?
            # No, increasing order is best for finding small scopes.
            
            # We can add a "kick": if we've tried many candidates, try a larger jump?
            
            # Let's stick to increasing order for now, as we want the smallest possible row 
            # to leave room for others? Actually, we just need *any* valid row.
            # But smaller rows are generally better for the global scope? 
            # No, the limit is fixed. We just need to fit.
            
            # One optimization: If the number of remaining slots is large, 
            # and the remaining range is small, prune.
            # Min gap is 1. So we need at least (k+1 - next_idx) - 1 space?
            # Actually, we need k+1-next_idx numbers in [x, limit].
            # So limit - x >= k+1 - next_idx - 1?
            # limit - x >= (k+1 - next_idx) - 1 is not quite right.
            # We need m numbers from {x, ..., limit}. Count is limit - x + 1.
            # Need limit - x + 1 >= k+1 - next_idx.
            
            min_needed = k + 1 - next_idx
            if limit - (last_val + 1) + 1 < min_needed:
                return None
            # Actually, we pick x > last_val. So x >= last_val + 1.
            # Max x we can pick is limit - (min_needed - 1).
            
            max_x = limit - (min_needed - 1)
            if max_x < last_val + 1:
                return None
            
            # Iterate x from last_val + 1 to max_x
            # To add randomness, we can shuffle the first few or pick a random start?
            # Let's try a "randomized greedy": pick a random valid x from the feasible set?
            # No, that might lead to dead ends.
            # Let's try the first few candidates deterministically, then randomize.
            
            # Candidates:
            # 1. last_val + 1
            # 2. last_val + 2
            # ...
            # Let's just iterate.
            
            for x in range(last_val + 1, max_x + 1):
                # Check conflicts
                # New diffs: x - a for a in row
                conflict = False
                for a in row:
                    d = x - a
                    if d in forbidden_diffs:
                        conflict = True
                        break
                    # Check internal conflict: d must not be in current row's diffs
                    # We don't have a set for current row diffs yet, but we can check
                    # against previous elements.
                    # Actually, d = x - a. Since x > a, d > 0.
                    # We need d not to be equal to any (row[j] - row[i]) for i<j.
                    # We can just check if d is in the set of diffs generated so far.
                    # For efficiency, we can maintain a set of current diffs.
                    # But passing a set is expensive. 
                    # Since k is small, O(k^2) check is fine.
                    for b in row:
                        if a != b and x - a == b - a: # Wait, diff is x-a. Existing diffs are row[j]-row[i].
                            pass
                # Let's do the check properly:
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden_diffs:
                    continue
                # Check internal conflicts
                # new_diffs must be distinct from each other (guaranteed if x-a are distinct? 
                # Yes, if a are distinct, x-a are distinct.
                # But new_diffs must not intersect with existing diffs in the row.
                # Existing diffs:
                existing_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        existing_diffs.add(row[j] - row[i])
                if new_diffs & existing_diffs:
                    continue
                
                # Recurse
                row.append(x)
                res = backtrack(row, next_idx + 1, x)
                if res is not None:
                    return res
                row.pop()
                
                # Optional: Break early if we've spent too much time on this branch?
                # The time check is at the top of backtrack.
            
            return None

        # Start recursion
        # Initial row: [0]
        # Forbidden diffs: empty set for internal, given set for external.
        # Note: The "forbidden_diffs" passed to find_row_backtracking are from OTHER rows.
        # Internal conflicts are handled in the loop.
        
        # Optimization: We can pre-compute the set of diffs for the current row as we go?
        # The current implementation recalculates existing_diffs every step. 
        # For k=7, this is fine.
        
        return backtrack([0], 1, 0)

    # Main Optimization Loop
    # We try to reduce the scope.
    # Current best scope is best_scope.
    # We try to find a solution with scope best_scope - 1, then - 2, etc.
    
    # To make it faster, we can start from best_scope - 1 and go down.
    # If we fail at S, we might fail at S-1, but a different arrangement might work.
    # So we should try multiple restarts for each S.
    
    # We will iterate target_scope from best_scope - 1 down to 110 (lower bound).
    # For each target_scope, we call solve_fixed_scope.
    # If it finds a solution, we update best_rows and best_scope, and continue to target_scope - 1.
    # If it fails, we move to the next lower target_scope? 
    # Actually, if we can't find a solution for S, we can't find one for S-1 (with the same row set structure), 
    # but we *can* find one for S-1 with a *different* row set.
    # So we should try S, then S-1, etc.
    
    # However, if we spend too much time on S and fail, we might run out of time.
    # We should limit the time per target_scope.
    
    current_target = best_scope - 1
    while current_target > 110 and time_left() > 20:
        # Allocate time for this target
        # If many targets left, spend less time per target.
        # Simple heuristic: spend 2s per target if > 10 targets left, else 5s.
        time_per_target = 2.0 if (best_scope - current_target) < 10 else 1.0
        
        # Modify solve_fixed_scope to accept time limit
        # We'll just wrap it or pass time budget.
        # The current solve_fixed_scope uses n_restarts and time_per_restart.
        # Let's adjust n_restarts based on time.
        
        # Estimate restarts: time_per_target / 0.5
        n_restarts = max(5, int(time_per_target / 0.5))
        
        res = solve_fixed_scope(current_target, n_restarts=n_restarts, time_per_restart=0.5)
        
        if res is not None:
            # Found a valid set with scope <= current_target
            # Actual scope might be less than current_target?
            # solve_fixed_scope enforces max <= current_target.
            # The actual scope is max(max(r) for r in res).
            new_scope = scope(res)
            if new_scope < best_scope:
                best_rows = res
                best_scope = new_scope
                # Continue to next lower target
                # We can jump down to new_scope - 1
                current_target = new_scope - 1
            else:
                # This shouldn't happen if logic is correct, but if it does, break?
                # Or just continue.
                pass
        else:
            # Failed to find a solution for this target.
            # Try the next lower target.
            current_target -= 1

    # Final polish: Try to reduce scope by 1 more time with remaining time
    if time_left() > 5 and best_scope > 112:
        res = solve_fixed_scope(best_scope - 1, n_restarts=20, time_per_restart=0.2)
        if res is not None:
            new_scope = scope(res)
            if new_scope < best_scope:
                best_rows = res
                best_scope = new_scope

    return best_rows
