import time
import sys
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # The task specifies n_rows=7 and k=5 for the specific instance, 
    # but the function signature is general. 
    # We assume the evaluator passes the specific task parameters.
    # For robustness, we handle general n_rows and k, but optimize for the known case.
    
    # Deadline guard
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    # The lower bound for M is roughly sqrt(n_rows * k * (k+1) / 2) or similar counting arguments.
    # For n=7, k=5, total differences = 7 * 15 = 105.
    # The differences are from 1 to M. So M must be at least 105.
    # Actually, the counting bound is M >= n_rows * k(k+1)/2.
    # 7 * 15 = 105.
    
    # We will search for M starting from the lower bound.
    # If we find a solution, we return it.
    
    # Since the known optimum is 112, we can start from 105.
    # If the problem instance is different, we should adjust.
    # However, the prompt says "The evaluator passes this task's n_rows and k".
    # And "The best known set has largest entry 112".
    # So we can safely assume the target is around 112.
    
    # To be general and safe:
    # Calculate a safe lower bound.
    # Total distinct differences needed = n_rows * (k * (k+1) // 2)
    # Let D = n_rows * k * (k+1) // 2
    # The differences are integers in [1, M]. So M >= D.
    
    D = n_rows * k * (k + 1) // 2
    lower_bound = D
    
    # We'll try M from lower_bound upwards.
    # We can add a small buffer to lower_bound if the search is too hard, 
    # but starting from the bound is best for minimizing M.
    
    # Optimization: For the specific case (7,5), we know 112 works.
    # We can try to jump to 112 if the lower bound search is too slow?
    # No, we want the minimum.
    
    # Let's use a backtracking function that tries to construct the set for a given M.
    
    def solve_for_M(M):
        used_diffs = set()
        rows = []
        
        # Heuristic: Sort rows by "difficulty" or just build them in order.
        # We build row by row.
        
        def backtrack_row(row_idx, current_row, used_diffs_local):
            if time.time() - start_time > time_limit:
                return None
            
            if len(current_row) == k + 1:
                # Row complete
                # Check if all differences in this row are unique within the row (guaranteed by construction if we check against used_diffs_local which includes previous rows and current row's diffs?)
                # Wait, we need to ensure diffs in current_row are not in used_diffs (global) and not in each other.
                # We maintain used_diffs_local as the set of all diffs used so far (previous rows + current row's diffs).
                # When we add a new element x to current_row, we generate new diffs x - y for y in current_row.
                # If any of these are in used_diffs_local, it's a conflict.
                # If not, we add them to used_diffs_local and proceed.
                
                # So if we reach here, the row is valid and its diffs are added to used_diffs_local.
                # We need to return the row and the updated used_diffs.
                # But used_diffs_local is a set, so we should pass it by reference or return it.
                # For clarity, let's pass used_diffs as a mutable set and backtrack by removing.
                
                if row_idx == n_rows - 1:
                    return [current_row]
                
                # Start next row
                next_row = [0]
                res = backtrack_row(row_idx + 1, next_row, used_diffs_local)
                if res is not None:
                    return [current_row] + res
                return None
            
            # Try to extend current_row
            # The last element in current_row is last_x
            last_x = current_row[-1]
            
            # We need to choose the next element x > last_x
            # The maximum possible x is M.
            # Also, the smallest possible x is last_x + 1.
            
            # Pruning: The remaining slots in the row are (k+1) - len(current_row).
            # Let rem = (k+1) - len(current_row)
            # We need to pick 'rem' numbers from last_x+1 to M.
            # The smallest possible sum of these numbers is (last_x+1) + ... + (last_x+rem)
            # This doesn't directly constrain M, but it ensures we don't pick too small numbers if M is tight?
            # Actually, we just iterate x from last_x + 1 to M.
            
            for x in range(last_x + 1, M + 1):
                # Check conflicts
                new_diffs = set()
                conflict = False
                for y in current_row:
                    d = x - y
                    if d in used_diffs_local:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # No conflict, add x to row and new_diffs to used
                current_row.append(x)
                used_diffs_local.update(new_diffs)
                
                res = backtrack_row(row_idx, current_row, used_diffs_local)
                if res is not None:
                    return res
                
                # Backtrack
                current_row.pop()
                used_diffs_local.difference_update(new_diffs)
            
            return None
        
        # Start with first row
        first_row = [0]
        res = backtrack_row(0, first_row, used_diffs)
        return res
    
    # Search for the smallest M
    # We can parallelize or use random restarts if needed, but sequential M search is good.
    # Start from lower_bound.
    # To be safe against timeout if the lower bound is infeasible but close,
    # we can limit the time spent per M.
    
    # For (7,5), D=105.
    # Let's try M from lower_bound to lower_bound + 20 (just in case).
    # If we don't find it, we might need a smarter heuristic or accept a larger M.
    # But 112 is known, so lower_bound+7 should cover it.
    
    # To handle potential timeouts in the inner loop, we check time in backtrack_row.
    
    best_solution = None
    # We'll try a few values of M.
    # If the problem is exactly (7,5), we know 112 is the answer.
    # We can start from 105.
    
    # Optimization: If n_rows==7 and k==5, we know the answer is 112.
    # We can verify 112 quickly. If it works, return it.
    # If not, search upwards.
    # But we want the *minimum*.
    
    # Let's just search from lower_bound upwards.
    # We'll break if we find a solution.
    
    # To improve performance, we can use a heuristic for the order of x.
    # Trying smaller x first is standard greedy.
    # Sometimes trying larger x first (or middle) helps?
    # For difference triangles, smaller differences are more constrained?
    # Let's stick to increasing x.
    
    # One improvement: If we fail to find a solution for M, we can try randomizing the order of x or rows.
    # But for now, let's try deterministic first.
    
    # If the deterministic search is too slow, we can add random restarts.
    
    # Let's add a time check inside the main loop too.
    for M in range(lower_bound, lower_bound + 50):
        if time.time() - start_time > time_limit - 5:
            break
        
        sol = solve_for_M(M)
        if sol is not None:
            return sol
            
    # If we didn't find a solution in the range, or timed out,
    # we should return a valid solution if we have one, or the best partial?
    # The problem requires a valid solution.
    # If we fail, we can fall back to a greedy construction which is always valid but might have larger M.
    
    # Let's implement a fast greedy fallback to ensure we return *something* valid.
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # If the backtracker failed or timed out, return the greedy result.
    # This ensures we don't crash or return None.
    # The greedy result is valid, just not optimal.
    return greedy_construct()

# EVOLVE-BLOCK-END
