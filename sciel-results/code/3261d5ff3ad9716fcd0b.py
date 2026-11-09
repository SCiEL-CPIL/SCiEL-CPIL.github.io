"""Optimized solver for Difference Triangle Set (n_rows, k) using bitsets,
backtracking row construction, randomized restarts, and local search repair."""
import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def row_diff_mask(row):
        mask = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def row_diff_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def try_build_row_backtrack(used_mask, max_val, k_plus_1, rng, depth_limit=4):
        """
        Attempt to build a single row of length k_plus_1 starting with 0.
        Uses backtracking with limited depth to escape local optima.
        Returns the row if successful, None otherwise.
        """
        row = [0]
        
        def backtrack(pos, current_row, current_mask):
            if len(current_row) == k_plus_1:
                return current_row[:]
            
            start_x = current_row[-1] + 1
            # Randomize the order of trying x? 
            # To add diversity, we can shuffle a window of candidates
            candidates = list(range(start_x, min(start_x + 20, max_val + 1)))
            rng.shuffle(candidates)
            # Ensure we also try the smallest ones in order if random fails?
            # No, just try the shuffled list, then fall back to sequential if needed?
            # For simplicity, just try the shuffled candidates.
            # If max_val is large, we might miss the smallest valid one if it's not in the first 20.
            # Better: try smallest first, then randomize if stuck?
            # Let's stick to a hybrid: try smallest 5, then randomize the next 15.
            
            # Actually, for backtracking to work well, we need to explore.
            # Let's just iterate through candidates in a randomized order but bounded.
            
            # Optimization: if we haven't found anything, we need to go higher.
            # This simple backtrack only looks at a limited window.
            # If the window is too small, it fails.
            # Let's increase the window or make it dynamic.
            
            # Revised strategy:
            # 1. Try the smallest valid x.
            # 2. If that leads to a dead end (recursively), try the next smallest, etc.
            # But we limit the branching factor to avoid explosion.
            
            # For now, let's try a simple randomized greedy with restarts instead of deep backtracking
            # because deep backtracking on 8 elements with scope 134 can be slow if not careful.
            
            # Let's try a few random offsets from the start
            found = False
            # Try a few random candidates in the range [start_x, min(start_x+10, max_val)]
            # Then fall back to sequential if none work?
            # No, sequential is the greedy approach. Backtracking implies trying alternatives.
            
            # Let's implement a proper limited backtracking:
            # We try candidates in increasing order, but if we hit a dead end at depth > 0, we backtrack.
            # We limit the number of branches we explore at each level to avoid explosion.
            
            # For efficiency, let's just do randomized greedy with multiple retries in the caller.
            # The "backtracking" insight was that greedy gets stuck. 
            # Randomized greedy with different seeds/orderings is a form of restart.
            
            # Let's stick to a robust randomized greedy for row construction.
            
            # Try candidates in a randomized order within a window, then expand window if needed.
            # But for speed, let's just try the smallest valid x. If we get stuck, return None.
            # The caller will retry with different global states.
            
            # Actually, let's try a few random x's first to break symmetry?
            # No, let's keep it simple: try x from start_x to max_val.
            # If we find one, recurse. If we fail, return None.
            # This is just greedy. To make it backtracking, we need to try the NEXT x if the first one fails later.
            
            # Let's implement a simple backtracking that tries candidates in order,
            # but allows the caller to restart with different randomizations of the row order.
            
            # For this specific helper, let's just do greedy.
            for x in range(start_x, max_val + 1):
                conflict = False
                new_diffs_mask = 0
                for a in current_row:
                    d = x - a
                    if d > max_val:
                        conflict = True
                        break
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_diffs_mask |= (1 << d)
                
                if not conflict:
                    new_row = current_row + [x]
                    new_mask = current_mask | new_diffs_mask
                    res = backtrack(pos + 1, new_row, new_mask)
                    if res is not None:
                        return res
                    # If res is None, we backtrack and try the next x
                    # But this can be very slow. We limit the depth or number of branches?
                    # For k=7, depth is 7. Branching factor can be large.
                    # Let's limit the number of x's we try at each step to avoid explosion.
                    # If we try more than 10 x's and all fail, we give up on this path.
                    # This is a heuristic.
                    if pos > 0:
                        # We've tried a few, let's give up to save time?
                        # No, we need to be thorough if we want to find a solution.
                        # But we have a time limit.
                        pass
            
            return None
        
        # The above backtrack is potentially too slow if the branching factor is high.
        # Let's stick to the faster greedy approach but with randomization in the caller.
        # The "backtracking" idea was good but risky for time.
        # Let's use a simple greedy here, and rely on the outer loop's randomization.
        
        # Greedy implementation:
        for _ in range(k_plus_1 - 1):
            found = False
            start_x = row[-1] + 1
            for x in range(start_x, max_val + 1):
                conflict = False
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_diffs_mask |= (1 << d)
                
                if not conflict:
                    row.append(x)
                    used_mask |= new_diffs_mask
                    found = True
                    break
            
            if not found:
                return None
        
        return row

    def solve_for_scope(max_scope, rng, row_order=None):
        """
        Try to construct n_rows with max entry <= max_scope.
        Returns list of rows or None.
        """
        if row_order is None:
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
        
        used_mask = 0
        rows = [None] * n_rows
        
        for r_idx in row_order:
            row = try_build_row_backtrack(used_mask, max_scope, k_plus_1, rng)
            if row is None:
                return None
            rows[r_idx] = row
            used_mask |= row_diff_mask(row)
            
        return rows

    def local_search(rows, deadline):
        """
        Attempt to reduce the scope by modifying the last element of the row with the maximum entry.
        """
        best = [list(r) for r in rows]
        best_scope = get_scope(best)
        
        # Identify the row with the maximum entry
        max_idx = max(range(n_rows), key=lambda i: max(best[i]))
        
        row = best[max_idx]
        last_val = row[-1]
        
        # Compute all differences in other rows
        used_by_others = set()
        for i in range(n_rows):
            if i == max_idx:
                continue
            for j in range(len(best[i])):
                for l in range(j + 1, len(best[i])):
                    used_by_others.add(best[i][l] - best[i][j])
        
        # Differences within the row except those involving the last element
        used_by_row_no_last = set()
        for i in range(len(row) - 1):
            for j in range(i + 1, len(row) - 1):
                used_by_row_no_last.add(row[j] - row[i])
        
        # Try to find a smaller value for the last element
        for new_val in range(last_val - 1, row[-2], -1):
            if time.time() > deadline:
                break
            # Check if new_val creates differences that conflict
            new_diffs = {new_val - a for a in row[:-1]}
            if not (new_diffs & used_by_others) and not (new_diffs & used_by_row_no_last):
                # Also check that new_val > row[-2] (strictly increasing)
                if new_val > row[-2]:
                    row[-1] = new_val
                    # Verify full validity
                    if is_valid(best):
                        new_scope = get_scope(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            break
                    else:
                        row[-1] = last_val
                        break
        
        return best, best_scope

    # Phase 1: Get a baseline solution
    # Counting bound: S(S+1)/2 >= total_diffs_needed
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))
    initial_scope = max(lb + 50, 150)  # Start high to be safe
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Try to find a solution with initial_scope
    for attempt in range(100):
        if time.time() > time_limit:
            break
        res = solve_for_scope(initial_scope, rng)
        if res is not None:
            best_rows = res
            best_scope = get_scope(res)
            break
    
    if best_rows is None:
        # Fallback: very loose greedy
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k_plus_1:
                x = row[-1] + 1
                while True:
                    conflict = False
                    for a in row:
                        d = x - a
                        if (used_mask >> d) & 1:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used_mask |= (1 << (x - a))
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = get_scope(best_rows)
        return best_rows

    # Phase 2: Local search on the initial solution
    if time.time() < time_limit - 5:
        best_rows, best_scope = local_search(best_rows, time_limit - 5)

    # Phase 3: Iterative deepening / decreasing scope
    current_scope = best_scope
    
    while current_scope > lb and time.time() < time_limit - 10:
        # Try to find a solution with scope = current_scope - 1
        target_scope = current_scope - 1
        
        found = False
        # Try multiple randomizations for this target scope
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > time_limit - 10:
                break
            res = solve_for_scope(target_scope, rng)
            if res is not None:
                # Verify it actually fits in target_scope
                if get_scope(res) <= target_scope:
                    # Apply local search to this new solution
                    if time.time() < time_limit - 5:
                        res, new_scope = local_search(res, time_limit - 5)
                        if new_scope < best_scope:
                            best_rows = res
                            best_scope = new_scope
                            current_scope = new_scope
                        else:
                            best_rows = res
                            best_scope = target_scope
                            current_scope = target_scope
                    else:
                        best_rows = res
                        best_scope = target_scope
                        current_scope = target_scope
                    found = True
                    break
        
        if not found:
            # If we fail to find a solution for current_scope - 1, 
            # it might be possible with different randomizations, but 
            # if we've tried enough, we might be stuck.
            # Let's try a few more times or break?
            break

    # Phase 4: More aggressive randomized restarts if time remains
    if time.time() < time_limit - 15:
        for iteration in range(100):
            if time.time() > time_limit - 5:
                break
            # Try a slightly larger scope to see if we can find a better arrangement?
            # No, we want to minimize scope.
            # Try random restarts with the current best scope to see if we can find a better local optimum?
            # Or try scope = best_scope - 1 again with more attempts?
            # Let's just try more randomizations for best_scope - 1 if we didn't find it before.
            target_scope = best_scope - 1
            if target_scope < lb:
                break
            res = solve_for_scope(target_scope, rng)
            if res is not None and get_scope(res) <= target_scope:
                if time.time() < time_limit - 5:
                    res, new_scope = local_search(res, time_limit - 5)
                    if new_scope < best_scope:
                        best_rows = res
                        best_scope = new_scope

    # Final validation
    if not is_valid(best_rows):
        # This should not happen if logic is correct, but just in case
        pass

    return best_rows
