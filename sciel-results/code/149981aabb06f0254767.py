import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    timeout = 115  # Leave 5s margin for 120s limit
    
    # k+1 integers, starting at 0. So we need to pick k-1 more integers.
    # Total elements per row: k+1.
    # Prompt says: "list of k + 1 integers that starts at 0"
    # Task says: "Choose 4 rows of 8 increasing integers starting at 0"
    # Wait, the task description says "8 increasing integers".
    # The interface says "list of k + 1 integers".
    # If k=7, then k+1=8.
    # The evaluator passes n_rows and k.
    # The task title is "Difference triangle set (4,7)".
    # So likely n_rows=4, k=7.
    # Let's assume k is the number of *steps* or such that total length is k+1.
    
    # Greedy initial solution to get an upper bound
    def greedy_construct():
        used = 0  # Bitset of used differences
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try next integer
                x = row[-1] + 1
                while True:
                    # Check differences with existing row elements
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if d <= 0:
                            continue
                        # Check if d is already in used
                        if (used >> d) & 1:
                            valid = False
                            break
                        # Check if d is already in new_diffs (duplicate within new additions)
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, max(max(r) for r in rows)
    
    # Get initial bound
    greedy_rows, initial_max = greedy_construct()
    best_rows = greedy_rows
    best_max = initial_max
    
    # If we have time, try to improve
    # We will try to find a solution with max < best_max
    # Using randomized backtracking
    
    def search(target_max, random_seed=0):
        random.seed(random_seed)
        used = 0
        rows = []
        
        def dfs(row_idx, current_row):
            nonlocal used, rows, best_rows, best_max
            
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                # Found a valid solution
                return True
            
            # Build the next row
            # current_row is a list, initially [0]
            # We need to add k elements (so length becomes k+1)
            # Wait, current_row starts as [0]. We need to reach length k+1.
            # So we need to add k more elements.
            
            # Heuristic: Try to keep numbers small.
            # But randomize to escape local minima?
            # For backtracking, we usually try values in order.
            # To improve, we can shuffle the order of candidates?
            # Candidates are integers from current_row[-1] + 1 to target_max.
            
            # Optimization: Pre-compute valid candidates?
            # No, just iterate.
            
            # To avoid deep recursion stack issues or slow iteration,
            # we can limit the search depth or use a heuristic.
            
            # Let's try values from smallest to largest, but with some randomness in later steps?
            # Actually, for this problem, smallest-first is a good heuristic for minimizing max.
            
            # However, pure DFS might get stuck.
            # We can use "Randomized Restart" strategy.
            # This function is called with a specific seed.
            
            # We implement a recursive DFS.
            # To speed up, we pass the current row and used bitset.
            
            # Note: Python recursion limit might be an issue if k is large, but k=7 is small.
            
            # We need to try all possible next values x.
            # x must be > current_row[-1] and <= target_max.
            
            # To improve pruning, we can check if it's possible to complete the row
            # with the remaining slots given the target_max.
            # Not easy.
            
            # Let's just do the check.
            
            # Optimization: If current_row[-1] is already close to target_max,
            # and we have many slots left, it might be impossible.
            
            # We iterate x from current_row[-1] + 1 to target_max.
            # To add randomness, we can shuffle the range?
            # No, that breaks the "minimize max" logic.
            # Instead, we rely on multiple restarts with different seeds
            # to potentially find different structures?
            # Actually, the structure is determined by the choices.
            # Smallest-first is deterministic.
            # To find a *better* solution, we need to make different choices.
            # So we SHOULD randomize the order of candidates, but bias towards small.
            
            # Strategy: Generate a list of candidates.
            # Sort them by a heuristic score.
            # Score: prefer small x, but also prefer x that leaves more room?
            
            # Let's try a simple randomization:
            # With probability p, pick a random valid x from the range.
            # Otherwise, pick the smallest valid x.
            
            # But we need to backtrack.
            
            # Let's implement a standard DFS but with a "random tie-breaking"
            # or just try a few random permutations of the search order?
            
            # Actually, a better approach for improvement:
            # Local Search on the Greedy Solution.
            
            # But let's finish the DFS implementation first.
            
            # To avoid Python overhead in deep recursion, we can use a stack.
            # But k is small.
            
            # We'll use a recursive helper.
            
            def backtrack(current_row, used_mask):
                nonlocal best_rows, best_max
                
                if time.time() - start_time > timeout:
                    return False
                
                if len(current_row) == k + 1:
                    # Row complete
                    rows.append(current_row)
                    return dfs(row_idx + 1, [0])
                
                # Determine next element
                # Start from current_row[-1] + 1
                # To add diversity, we can shuffle the candidate order?
                # No, we want to find *a* solution under target_max.
                # If we find one, we update best_max.
                
                # Let's try candidates in increasing order.
                # If we fail, we backtrack.
                
                # Optimization: 
                # If we have many slots left (slots_left = (k+1) - len(current_row)),
                # the minimum possible next value is current_row[-1] + 1.
                # The minimum possible final value is current_row[-1] + slots_left.
                # If current_row[-1] + slots_left > target_max, prune.
                
                slots_left = (k + 1) - len(current_row)
                if current_row[-1] + slots_left > target_max:
                    return False
                
                # Iterate candidates
                # To improve search, we can randomize the order of candidates 
                # if we are not finding a solution quickly?
                # For now, let's stick to increasing order for determinism in a single run,
                # but we will call this function with different seeds in a loop.
                # Wait, if the order is always increasing, the seed doesn't change the path!
                # So we MUST randomize the order of candidates to explore different branches.
                
                # Generate candidates
                candidates = list(range(current_row[-1] + 1, target_max + 1))
                
                # Shuffle candidates?
                # If we shuffle completely, we might pick large numbers early, 
                # which is bad for minimizing max, but we are already constrained by target_max.
                # We want to find ANY solution under target_max.
                # So shuffling is good to escape local minima.
                
                # However, shuffling the entire range is expensive if range is large.
                # We can use a "biased random" order.
                # Or just shuffle.
                
                random.shuffle(candidates)
                
                # But we should still prefer smaller numbers?
                # Let's try a mix: 50% smallest first, 50% random?
                # Or just random.
                
                for x in candidates:
                    # Check if x is valid
                    valid = True
                    new_diffs_mask = 0
                    
                    for a in current_row:
                        d = x - a
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        if (new_diffs_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs_mask |= (1 << d)
                    
                    if not valid:
                        continue
                    
                    # Recursive step
                    # We need to copy current_row and used_mask?
                    # Python integers are immutable, lists are mutable.
                    # We pass a copy of the list.
                    
                    if backtrack(current_row + [x], used_mask | new_diffs_mask):
                        return True
                    
                return False
            
            # Start backtracking for this row
            if backtrack([0], 0):
                return True
            
            return False
        
        # The dfs function above was a bit messy with closures.
        # Let's refactor into a cleaner nested function.
        
        # ... (Refactoring omitted in thought, will write in code)
        
        return False

    # The above logic is a bit complex to nest. Let's write a cleaner version.
    
    def try_find_solution(target_max, seed):
        random.seed(seed)
        used_global = 0
        rows_found = []
        
        def build_row(row_idx, current_row, used_mask):
            # Returns True if row is complete and added to rows_found
            if len(current_row) == k + 1:
                rows_found.append(current_row)
                return True
            
            slots_left = (k + 1) - len(current_row)
            if current_row[-1] + slots_left > target_max:
                return False
            
            # Generate candidates
            # To improve performance, we can limit the candidate range
            # but we need to check all.
            
            # Randomize order
            # We'll pick a random subset or shuffle?
            # Shuffling a large range is O(N log N).
            # We can use a "random walk" or just pick random candidates?
            # Let's shuffle.
            
            start_val = current_row[-1] + 1
            candidates = list(range(start_val, target_max + 1))
            random.shuffle(candidates)
            
            # To bias towards small values (better for future pruning),
            # we can sort by a random key? No, that's just shuffle.
            # We can try the smallest few first, then random?
            # Let's just shuffle.
            
            for x in candidates:
                # Check validity
                valid = True
                new_diffs = 0
                for a in current_row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    if (new_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if not valid:
                    continue
                
                # Recurse
                if build_row(row_idx, current_row + [x], used_mask | new_diffs):
                    return True
            
            return False
        
        # Main loop for rows
        def build_all_rows(row_idx, used_mask):
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Start building this row
            if build_row(row_idx, [0], used_mask):
                # Row built successfully, update used_mask?
                # Wait, build_row doesn't return the new used_mask.
                # We need to calculate it.
                # The last row added is rows_found[-1].
                # We can calculate its diffs.
                last_row = rows_found[-1]
                row_diffs = 0
                for i in range(len(last_row)):
                    for j in range(i+1, len(last_row)):
                        d = last_row[j] - last_row[i]
                        row_diffs |= (1 << d)
                
                if build_all_rows(row_idx + 1, used_mask | row_diffs):
                    return True
            
            # Backtrack: remove last row
            if rows_found:
                rows_found.pop()
            
            return False
        
        if build_all_rows(0, 0):
            return rows_found
        return None

    # Loop to improve
    # Try decreasing target_max
    # Start from best_max - 1
    # If found, update best_max and repeat.
    
    # To avoid infinite loop or slow search, we limit the number of attempts per target_max.
    
    current_target = best_max
    while time.time() - start_time < timeout:
        current_target -= 1
        if current_target < 1:
            break
        
        found = False
        # Try a few random seeds for this target
        for seed in range(5):
            if time.time() - start_time > timeout:
                break
            res = try_find_solution(current_target, seed)
            if res:
                best_rows = res
                best_max = current_target
                found = True
                break
        
        if not found:
            # If we failed for this target, should we continue to lower targets?
            # Probably not, as lower targets are harder.
            # But maybe a different structure works?
            # Let's stop decreasing if we fail a few times?
            # Or just break.
            break

    return best_rows
# EVOLVE-BLOCK-END
