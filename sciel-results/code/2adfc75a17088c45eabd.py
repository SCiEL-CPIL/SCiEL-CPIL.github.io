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
    time_limit = 115.0  # Leave buffer
    
    # We need to find a set of n_rows rows, each of length k+1.
    # Total differences needed = n_rows * (k*(k+1)//2)
    # Let D be the total number of differences.
    # The maximum difference in any row is at most M (the scope).
    # The total number of available differences in [1, M] is M.
    # So M >= D is a necessary condition.
    # Actually, since differences are distinct across ALL rows, the total number of distinct differences used is exactly D.
    # So we need M >= D.
    # For n_rows=7, k=6: D = 7 * (6*7//2) = 7 * 21 = 147.
    # So M >= 147? Wait.
    # The differences are a_j - a_i for 0 <= i < j <= k.
    # There are k(k+1)/2 such differences per row.
    # These must be distinct across all rows.
    # So we need M (max value) to be at least the largest difference used.
    # The largest difference is at most M.
    # The number of distinct differences used is D.
    # So we need at least D distinct integers in [1, M].
    # Thus M >= D.
    # For n_rows=7, k=6, D=147. So M >= 147.
    # But the problem statement says "best known set has largest entry 112".
    # This implies my counting is wrong or I misunderstand "differences".
    # "all differences between two entries of the same row are distinct over all rows"
    # This means if row 1 has diff 5, no other row can have diff 5.
    # Yes, that's what I used.
    # Why is 112 possible?
    # Ah, the differences don't have to cover all integers up to M. They just have to be distinct.
    # The largest entry is M. The largest difference is at most M.
    # The number of distinct differences is D=147.
    # The available differences are integers in [1, M].
    # So we need M >= 147.
    # Is it possible that M=112 is sufficient?
    # No, you can't have 147 distinct integers in [1, 112].
    # Let me re-read carefully.
    # "Choose 7 rows of 6 increasing integers starting at 0" -> k=6 means 7 integers?
    # "rows of 6 increasing integers" usually means 6 numbers.
    # But "list of k + 1 integers" in the interface description: `construct(n_rows, k) -> list of n_rows rows; ... Each row is a list of k + 1 integers`.
    # So if k=6, row length is 7.
    # Number of differences per row = 7*6/2 = 21.
    # Total differences = 7 * 21 = 147.
    # If M=112, we can only have 112 distinct differences.
    # Contradiction?
    # Let's check the source "Shehadeh, Kingsford and Kschischang, 2025".
    # Maybe the problem is "Difference Triangle Set" where the condition is different?
    # Or maybe "6 increasing integers" means k=5?
    # The prompt says: `construct(n_rows, k)` ... `Each row is a list of k + 1 integers`.
    # Task title: "Difference triangle set (7,5)".
    # Usually (n, k) notation in these problems might mean n rows, k columns? Or n rows, k differences?
    # If it's (7, 5), and it says "6 increasing integers", maybe k=5 in the title corresponds to row length 6?
    # If row length is 6, differences = 6*5/2 = 15.
    # Total diffs = 7 * 15 = 105.
    # "simple counting bound is 105". This matches M >= 105.
    # "best known set has largest entry 112".
    # So, the task is likely n_rows=7, row_length=6 (which implies k=5 in the interface if k+1=6).
    # The evaluator passes `n_rows` and `k`.
    # If the evaluator passes k=5, row length is 6.
    # If the evaluator passes k=6, row length is 7.
    # The title says (7,5). This strongly suggests n_rows=7, k=5.
    # So I should assume the evaluator will call construct(7, 5).
    # My code should be general for any n_rows, k.
    
    # Let's refine the solver.
    # We will use a backtracking search that builds rows one by one.
    # To minimize the maximum entry, we can use a binary search on the scope M, or iterative deepening.
    # However, the search space is large.
    # A better heuristic:
    # 1. Sort rows by "difficulty" or just build them sequentially?
    # 2. Use a "most constrained variable" approach for choosing the next element in the current row.
    
    # Let's stick to building rows sequentially, but use a strong heuristic for element selection.
    # Heuristic: When choosing the next element x for the current row, prefer x that uses "rare" differences?
    # Or prefer x that leaves the most options for the rest of the row?
    
    # Let's implement a solver that tries to find a solution with scope M.
    # We will try M from the lower bound (D) upwards.
    
    def solve_with_scope(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # We need to choose x > last_val such that all diffs x-a are new.
            # Also x <= M.
            
            # Heuristic: 
            # We want to pick x that doesn't block future choices too much.
            # A common heuristic in such packing problems is to prefer smaller x to keep the row compact,
            # but also to use differences that are "harder" to place later?
            # Actually, small differences are more "scarce" in the sense that they appear more frequently in random sets?
            # No, all differences are equally likely if we pick randomly.
            # But small differences restrict the next element more?
            # If we use a small difference d, it means we have an element at x-d.
            # This doesn't directly restrict the next element y > x except via the diff y-d and y-x.
            
            # Let's try a simple heuristic: try x in increasing order.
            # This tends to keep the row compact, which might help with the global scope M.
            
            candidates = []
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Additional pruning:
                # Check if there are enough remaining differences for the rest of the row?
                # This is complex. Let's just try the candidates.
                
                candidates.append(x)
                
                # Limit branching to avoid explosion.
                # If we find a solution, we return immediately.
                # If we don't, we backtrack.
                # To speed up, we can limit the number of candidates we try in each step.
                # But if we limit too much, we might miss the solution.
                # Given M is small (around 105-112), the branching factor is manageable.
                
                if len(candidates) > 50:
                    break
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                # We already checked new_diffs & used_diffs == empty in the loop above,
                # but we need to re-calculate or store them.
                # For efficiency, let's store (x, new_diffs).
                
            # Refactor loop to store diffs
            valid_candidates = []
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                valid_candidates.append((x, new_diffs))
                
                if len(valid_candidates) > 50:
                    break
            
            for x, new_diffs in valid_candidates:
                # Try this candidate
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                # Restore
                used_diffs = old_used
            
            return False
        
        return backtrack(0, [0])

    # Determine the lower bound for M
    total_diffs = n_rows * (k * (k + 1) // 2)
    min_M = total_diffs
    
    # We know the best known is 112 for (7,5).
    # If n_rows=7 and k=5, total_diffs = 105.
    # So we start from 105.
    
    # To be safe and fast, we can try a few values around the known optimum if we recognize the parameters.
    # But the code should be general.
    
    # Strategy:
    # 1. Try M = min_M.
    # 2. If fail, try M = min_M + 1, etc.
    # 3. Use a time limit per attempt? Or global time limit?
    #    Global time limit is better.
    
    # To improve performance, we can use a randomized restart strategy if the deterministic search is slow.
    # But let's first try the deterministic search with increasing M.
    
    # Optimization: 
    # The order of rows might matter.
    # Symmetry breaking: We can fix the first row to be as "standard" as possible?
    # Or just try all permutations? No, too expensive.
    # Let's just build rows in the given order.
    
    # Let's add a simple randomization to the candidate selection to escape local minima if we add restarts.
    # For now, let's stick to deterministic increasing M.
    
    # To handle the case where search is too slow for small M, we can set a timeout per M value.
    # If we don't find a solution for M within, say, 10 seconds, we assume it's not possible (or too hard) and try M+1.
    # This is a heuristic assumption.
    
    best_result = None
    current_M = min_M
    
    while time.time() - start_time < time_limit:
        if solve_with_scope(current_M):
            # We found a solution. Is it the best?
            # Since we started from the lower bound and incremented, yes.
            # But wait, solve_with_scope modifies `rows` and `used_diffs` internally?
            # No, `rows` is local to `solve_with_scope` in my draft above?
            # I defined `rows` inside `solve_with_scope`. So it's clean.
            # But I need to return the rows.
            
            # Let's modify solve_with_scope to return the rows if successful.
            
            break
        
        current_M += 1
        
        # If we go too high, we might run out of time.
        # If we reach a high M (e.g., 200) and still no solution, something is wrong.
        if current_M > 200:
            break

    # The above implementation of solve_with_scope doesn't return rows.
    # I need to fix that.
    
    # Let's rewrite the solver properly.
    
    def find_solution(M):
        """
        Finds a solution with max entry <= M.
        Returns the list of rows if found, else None.
        """
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Generate candidates
            # Heuristic: Sort candidates by x (ascending) to keep rows compact.
            # This usually leads to smaller maximums.
            
            valid_candidates = []
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                valid_candidates.append((x, new_diffs))
                
                # Pruning: if we have too many candidates, we might want to limit them.
                # But for small M, it's fine.
                
                if len(valid_candidates) > 100:
                    break
            
            # Try candidates
            for x, new_diffs in valid_candidates:
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        success = backtrack(0, [0])
        if success:
            return [r[:] for r in rows]
        return None

    # Search for the minimal M
    min_M = n_rows * (k * (k + 1) // 2)
    
    # We can try to jump to a known good value if we recognize the problem size?
    # No, let's just iterate.
    
    # To speed up, we can use a "fail-fast" heuristic.
    # If we fail to find a solution for M, we try M+1.
    
    # However, the search space can be huge.
    # For (7,5), M=105 is the bound. The solution is 112.
    # So we will try 105, 106, ..., 112.
    # If the search for 105 is fast (proves impossibility), we move to 106.
    # If the search for 112 is fast (finds solution), we stop.
    
    # The risk is that proving impossibility for 105-111 is slow.
    # We can add a time limit per M value.
    
    result = None
    current_M = min_M
    
    while time.time() - start_time < time_limit:
        # Reset time limit for this specific M? No, global limit is safer.
        # But if we spend 100s on M=105, we have 15s left for M=106..112.
        # Let's give each M a sub-limit, e.g., 10s.
        
        sub_start = time.time()
        sub_limit = 10.0
        
        # We need to modify find_solution to respect a sub-limit or check global time.
        # The global time check in backtrack is already there.
        
        sol = find_solution(current_M)
        
        if sol is not None:
            result = sol
            break
        
        # If we didn't find a solution, and we used a lot of time, we might just move on.
        # The global time check in backtrack will return False if time is up.
        
        current_M += 1
        
        # Safety break
        if current_M > 200:
            break

    if result is None:
        # Fallback: Greedy construction
        # This ensures we return a valid solution even if the search fails.
        # It won't be optimal, but it's better than an error.
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    return result
