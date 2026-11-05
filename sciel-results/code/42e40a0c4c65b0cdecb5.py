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
    time_limit = 145.0  # Use more of the 150s budget
    
    best_result = None
    best_scope = float('inf')
    
    def solve_attempt(max_scope):
        nonlocal start_time, time_limit, best_scope, best_result
        
        used_diffs = set()
        rows = []
        
        def _backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
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
            
            upper_bound = min(max_scope - 1, best_scope - 1) if best_scope < float('inf') else max_scope - 1
            
            candidates = []
            
            # Generate valid candidates
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Heuristic: Lookahead
                # Count how many valid next elements y > x exist
                potential_nexts = 0
                for y in range(x + 1, upper_bound + 1):
                    # Check if y is valid given current_row + [x]
                    # Diffs: {y - a for a in current_row} U {y - x}
                    # Must be disjoint from used_diffs U new_diffs
                    
                    # Quick check: y - x
                    if (y - x) in used_diffs or (y - x) in new_diffs:
                        continue
                        
                    # Check y - a
                    valid = True
                    for a in current_row:
                        d = y - a
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                            
                    if valid:
                        potential_nexts += 1
                        if potential_nexts > 20:
                            break
                
                # Score: Maximize potential_nexts, Minimize x
                candidates.append((-potential_nexts, x, new_diffs))
                
                if len(candidates) > 40:
                    break
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: (item[0], item[1]))
            
            # Try candidates
            for neg_potential, x, new_diffs in candidates:
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                if _backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used

        return _backtrack(0, [0])

    # Strategy:
    # Start with a high scope and tighten.
    
    current_max_scope = 130
    
    while time.time() - start_time < time_limit:
        if solve_attempt(current_max_scope):
            if best_scope < current_max_scope:
                current_max_scope = best_scope - 1
            else:
                current_max_scope = best_scope - 1
        else:
            # Failed to find solution
            if current_max_scope <= 110:
                break
            else:
                # Retry logic? For now, break.
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
