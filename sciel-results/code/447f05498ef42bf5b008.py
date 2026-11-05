import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    best_result = None
    best_scope = float('inf')
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer for the evaluator
    
    # Number of restarts
    max_restarts = 10000 # Increase restarts as each step is slightly more expensive
    
    # Threshold for "small" differences
    # Small differences are critical because there are only 1..T of them.
    # If we use them up, we are forced to use larger gaps.
    small_threshold = 20 

    def construct_row(used_diffs, limit):
        """
        Backtracking search to construct a single row of length k+1.
        Returns the row if successful, None otherwise.
        """
        row = [0]
        
        def backtrack(pos):
            if pos == k:
                return True
            
            last_val = row[-1]
            candidates = []
            
            # Generate valid candidates
            for x in range(last_val + 1, limit + 1):
                new_diffs = [x - a for a in row]
                # Check if any new difference is already used
                conflict = False
                for d in new_diffs:
                    if d in used_diffs:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Heuristic:
                # 1. Minimize the "heat": avoid small differences.
                #    Score 1: -min(new_diffs) (we want larger min diffs)
                # 2. Minimize the spread: sum(new_diffs) (we want compact rows)
                #    Score 2: sum(new_diffs)
                
                min_d = min(new_diffs)
                sum_d = sum(new_diffs)
                
                # We prioritize candidates that have a larger minimum difference.
                # This saves small differences for other rows or later positions.
                # Secondary key: smaller sum to keep the row tight.
                candidates.append((min_d, sum_d, x, new_diffs))
                
                # Optimization: If we have many candidates, we might not need to check all.
                # But for backtracking, we need the full sorted list to try best-first.
            
            if not candidates:
                return False
            
            # Sort: 
            # Primary: min_d descending (largest min diff first -> avoids small diffs)
            # Secondary: sum_d ascending (smallest sum first -> keeps row compact)
            candidates.sort(key=lambda item: (-item[0], item[1]))
            
            # Try candidates in order
            for min_d, sum_d, x, new_diffs in candidates:
                # Update state
                row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack(pos + 1):
                    return True
                
                # Backtrack
                row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False
        
        if backtrack(0):
            return row
        return None

    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        used_diffs = set()
        rows = []
        
        success = True
        # Determine the limit for this restart.
        # Start with a tight bound if we have a best_scope, otherwise a reasonable default.
        # The counting bound is 105. Best known is 112.
        # Let's try to beat the current best.
        if best_scope < float('inf'):
            limit = best_scope - 1
        else:
            limit = 115 # Start slightly above the counting bound
        
        # If limit is too small, it's impossible.
        # The minimum possible scope is at least the counting bound.
        
        for _ in range(n_rows):
            row = construct_row(used_diffs, limit)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success:
            current_scope = max(r[-1] for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
                # If we hit the theoretical lower bound, we can stop
                if best_scope <= 105:
                    break
    
    if best_result is None:
        # Fallback to deterministic greedy if random search failed (unlikely)
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
