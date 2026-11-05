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

    # Total number of differences needed
    # Each row has k+1 elements, so C(k+1, 2) differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # If we have a scope M, the available differences are 1..M
    # We need at least total_diffs_needed distinct differences
    # So M must be at least total_diffs_needed (actually M(M+1)/2 >= total_diffs_needed is the counting bound)
    # But M is the max element, not the max difference.
    # The max difference in a row is at most M.
    # So the set of available differences is {1, 2, ..., M}.
    # We need total_diffs_needed distinct values from {1, ..., M}.
    # So M >= total_diffs_needed.
    
    # Counting bound: M(M+1)/2 >= total_diffs_needed
    # M ~ sqrt(2 * total_diffs_needed)
    # For n_rows=7, k=6: diffs_per_row = 21, total = 147.
    # M ~ sqrt(294) ~ 17.2. But M is the max element.
    # The differences are a_j - a_i. The largest difference is a_k - a_0 = a_k.
    # So the max difference is the scope.
    # We need 147 distinct differences from {1, ..., M}.
    # So M >= 147.
    # Wait, the counting bound in the problem statement says 105.
    # Let's re-read: "simple counting bound is 105".
    # Maybe the bound is on the sum of differences? Or something else?
    # Actually, the number of available differences is M.
    # We need 147 distinct differences. So M >= 147.
    # But the problem says best known is 112 and counting bound is 105.
    # This implies my understanding of "counting bound" is wrong or the problem has a different structure.
    # Ah, the differences are a_j - a_i.
    # In a row [0, a1, ..., ak], the differences are {aj - ai | i < j}.
    # The largest difference is ak.
    # So all differences are in {1, ..., ak}.
    # We need 147 distinct differences in total.
    # So the union of differences from all rows must be a subset of {1, ..., M} where M is the max entry.
    # So M must be at least 147.
    # Why does the problem say counting bound is 105?
    # Maybe I am misinterpreting "counting bound".
    # Or maybe the differences don't have to be distinct across all rows?
    # "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
    # Yes, they must be distinct.
    # So M >= 147.
    # But the best known is 112. This is a contradiction.
    # Let me re-read the task carefully.
    # "Choose 7 rows of 6 increasing integers starting at 0" -> 7 rows, each with 6 integers?
    # "k+1 integers" in the interface.
    # If k=6, then 7 integers per row.
    # Differences per row = C(7, 2) = 21.
    # Total = 7 * 21 = 147.
    # So M >= 147.
    # But the task says "best known set has largest entry 112".
    # This is impossible if M >= 147.
    # Unless... "6 increasing integers" means k=6, so 6 integers?
    # Interface: `construct(n_rows, k)` -> list of n_rows rows.
    # "Each row is a list of k + 1 integers".
    # So if k=6, row has 7 integers.
    # Maybe the task description "Choose 7 rows of 6 increasing integers" means k=5?
    # If k=5, row has 6 integers.
    # Differences per row = C(6, 2) = 15.
    # Total = 7 * 15 = 105.
    # Then M >= 105.
    # And the counting bound is 105.
    # And best known is 112.
    # This fits perfectly!
    # So the task uses k=5 for the specific case mentioned, but the interface takes k.
    # The evaluator will pass n_rows and k.
    # For the specific case (7, 6) in the title, it likely means 7 rows, 6 integers per row, so k=5.
    # But the interface says `construct(n_rows, k)`.
    # I will assume the evaluator passes the correct k.
    # My code should work for general n_rows and k.

    # Backtracking with pruning
    # State: row_idx, current_row, used_diffs
    # Heuristic: Choose next element x such that the new differences are "large" (to preserve small ones)
    # and x is as small as possible.
    
    best_result = None
    best_scope = float('inf')
    
    # We'll use a global counter for nodes visited to stop if too many
    nodes_visited = 0
    max_nodes = 500000  # Adjust this
    
    def solve_attempt(max_scope):
        nonlocal best_result, best_scope, nodes_visited, max_nodes, start_time, time_limit
        
        used_diffs = set()
        rows = []
        
        def _backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, nodes_visited, best_scope, best_result
            
            if time.time() - start_time > time_limit:
                return False
            
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return False
            
            if row_idx == n_rows:
                # All rows constructed
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Upper bound for x
            # Must be < max_scope
            # Also, if we have a best_scope, we can prune if x >= best_scope
            upper_bound = min(max_scope - 1, best_scope - 1) if best_scope < float('inf') else max_scope - 1
            
            candidates = []
            
            # Generate candidates
            # To speed up, we can skip x values that create small differences if we already have many small ones used?
            # Heuristic: Prefer x that creates large differences.
            # Cost: sum(1/d for d in new_diffs)
            
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Check if we can still complete the row with remaining elements
                # We need (k + 1 - len(current_row) - 1) more elements.
                # The next element must be > x.
                # The smallest possible next element is x + 1.
                # The smallest difference it can create is 1.
                # But 1 might be used.
                # This is complex. Let's just use the cost heuristic.
                
                cost = sum(1.0 / d for d in new_diffs)
                
                # Also, prefer smaller x to keep scope small
                # Combine: cost + 0.1 * x
                score = cost + 0.1 * x
                
                candidates.append((score, x, new_diffs))
                
                # Limit candidates to avoid explosion
                if len(candidates) > 20:
                    break
            
            if not candidates:
                return False
            
            # Sort by score
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates
            for score, x, new_diffs in candidates:
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                if _backtrack(row_idx, current_row + [x]):
                    return True
                
                # Restore state
                used_diffs = old_used
                
            return False
        
        return _backtrack(0, [0])

    # Strategy:
    # 1. Start with a high max_scope (e.g., 200).
    # 2. If a solution is found, lower max_scope to best_scope - 1 and retry.
    # 3. Stop when time limit is reached or no improvement is made.
    
    current_max_scope = 200
    
    while time.time() - start_time < time_limit:
        nodes_visited = 0
        if solve_attempt(current_max_scope):
            if best_scope < current_max_scope:
                current_max_scope = best_scope - 1
            else:
                # If we found a solution at the boundary, try to improve by lowering bound
                current_max_scope = best_scope - 1
        else:
            # No solution found with current_max_scope
            # This might mean current_max_scope is too small, or search failed.
            # If we have a best_result, we can stop.
            # If not, we might need to increase max_scope? But we started high.
            # So if we fail, it's likely because current_max_scope is too small (optimal > current_max_scope)
            # or search got stuck.
            # Since we are decreasing current_max_scope, if we fail, it means the optimal is > current_max_scope.
            # But we already have a best_result with scope best_scope.
            # So we can stop.
            break
    
    if best_result is None:
        # Fallback to greedy
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
