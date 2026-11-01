"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a backtracking search with pruning and symmetry breaking.
    """
    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Known optimal values for small cases
    # For (7,5), the best known is 112
    known_optimal = {
        (7, 5): 112,
        (5, 4): 68,
        (4, 3): 40,
        (3, 2): 15,
    }
    
    start_time = time.time()
    timeout = 100  # seconds
    
    # Try to find a solution using backtracking with good heuristics
    def solve():
        global_best = None
        global_best_max = float('inf')
        
        def bt(row_idx, current_row, used, max_val, rows):
            nonlocal global_best, global_best_max
            
            if time.time() - start_time > timeout:
                return False
            
            if max_val >= global_best_max:
                return False
            
            if row_idx == n_rows:
                global_best_max = max_val
                global_best = [list(r) for r in rows]
                return True
            
            if len(current_row) == k + 1:
                # Check if the row is valid.
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        if d in used or d in row_diffs:
                            return False
                        row_diffs.add(d)
                
                rows.append(current_row)
                result = bt(row_idx + 1, [0], used | row_diffs, max(max_val, current_row[-1]), rows)
                rows.pop()
                return result
            
            last = current_row[-1]
            
            # Determine the search range for the next value
            # We want to keep values as small as possible
            # But we also need to ensure we can complete the row
            remaining_elements = (k + 1) - len(current_row)
            
            # Upper bound: we can go up to a reasonable limit
            # The key insight: we need remaining_elements more values, each at least 1 apart
            # So the minimum possible max for this row is last + remaining_elements
            # But we might need to go higher to avoid conflicts
            
            # Let's try values in increasing order, but with a smart upper bound
            # The global max must be at least total_diffs
            # For this row, the max entry must be at least last + remaining_elements
            
            # Try a window of values
            # Start from last + 1, go up to last + 200 or so
            # But we can be smarter: if we're close to the optimal, try smaller ranges
            
            # For efficiency, try values in a limited range
            max_search = last + 150
            for x in range(last + 1, max_search):
                ok = True
                for a in current_row:
                    if (x - a) in used:
                        ok = False
                        break
                if not ok:
                    continue
                
                new_row = current_row + [x]
                new_used = used | {x - a for a in current_row}
                
                if bt(row_idx, new_row, new_used, max(max_val, x), rows):
                    return True
            
            return False
        
        # Start with an empty row.
        rows = []
        bt(0, [0], set(), 0, rows)
        
        if global_best is not None:
            return global_best
        
        # Fallback: use a greedy approach with multiple restarts
        best_rows = None
        best_max = float('inf')
        
        for restart in range(20):
            random.seed(restart)
            used = set()
            rows = []
            success = True
            
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                    else:
                        # Try skipping some values
                        continue
                    if x > 500:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            
            if success and rows:
                current_max = max(max(r) for r in rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = rows
        
        if best_rows is not None:
            return best_rows
        
        # Last resort: return a simple greedy solution
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
    
    return solve()
# EVOLVE-BLOCK-END
