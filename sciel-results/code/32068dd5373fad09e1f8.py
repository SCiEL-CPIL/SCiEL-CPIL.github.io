"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from functools import lru_cache

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    # Each row has C(k+1, 2) differences. Total differences = n_rows * C(k+1, 2)
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # The minimum possible max difference is at least total_diffs (since diffs are 1..M and distinct)
    # But we need to find the smallest M such that a valid configuration exists.
    
    # We'll do a binary search on M (the maximum allowed value in any row)
    # For each candidate M, we check feasibility via backtracking.
    
    def is_feasible(M):
        """Check if we can construct n_rows rows with max entry <= M."""
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx):
            if row_idx == n_rows:
                return True
            
            # Try to build row row_idx
            row = [0]
            
            def build_row(pos):
                if pos == k + 1:
                    rows.append(row[:])
                    result = backtrack(row_idx + 1)
                    if result:
                        return True
                    rows.pop()
                    return False
                
                # Try values from row[-1] + 1 up to M
                start = row[-1] + 1
                for val in range(start, M + 1):
                    # Check all differences val - row[i] for i in range(pos)
                    valid = True
                    new_diffs = []
                    for i in range(pos):
                        d = val - row[i]
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Also check: can the remaining positions be filled?
                    # Pruning: the remaining (k+1 - pos - 1) positions need to create
                    # differences that are all distinct and not in used_diffs ∪ new_diffs
                    # Simple bound: remaining positions must be > val, so min next is val+1
                    # The smallest new difference we'd need is 1, but many are used.
                    # We'll just proceed and let deeper recursion prune.
                    
                    row.append(val)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    if build_row(pos + 1):
                        return True
                    
                    # Backtrack
                    row.pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
                
                return False
            
            return build_row(1)
        
        return backtrack(0)
    
    # Binary search on M
    # Lower bound: we need total_diffs distinct positive integers, so M >= total_diffs
    # But actually the max entry M must be at least... let's think.
    # The differences go up to at most M (since max entry is M, min is 0, so max diff is M).
    # We need total_diffs distinct values in [1, M], so M >= total_diffs.
    # Upper bound: the greedy solution gives some value. Let's compute it.
    
    # First, get an upper bound from greedy
    used, rows_greedy = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows_greedy.append(row)
    
    upper_bound = max(max(r) for r in rows_greedy)
    
    # Try to find a better solution via binary search
    low = total_diffs  # theoretical lower bound on M
    high = upper_bound
    
    best_rows = rows_greedy
    best_M = upper_bound
    
    # Try values from low to high, but that could be too many.
    # Instead, try a few promising values.
    # Since the search space is large, let's try a limited number of M values.
    
    # Actually, for n_rows=7, k=6, total_diffs=105, upper_bound is around 112 from greedy.
    # Let's try M values from total_diffs to upper_bound, but with a time limit.
    
    import time
    start_time = time.time()
    time_limit = 100  # seconds, leave some buffer for 120s total
    
    # Try binary search style
    lo = low
    hi = high
    found = None
    
    while lo <= hi and time.time() - start_time < time_limit:
        mid = (lo + hi) // 2
        if is_feasible(mid):
            found = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if found is not None and found < best_M:
        # Reconstruct the solution for M = found
        # We need to re-run is_feasible but also capture the rows
        used_diffs = set()
        rows = []
        
        def backtrack_row(row_idx):
            if row_idx == n_rows:
                return True
            
            row = [0]
            
            def build_row(pos):
                if pos == k + 1:
                    rows.append(row[:])
                    result = backtrack_row(row_idx + 1)
                    if result:
                        return True
                    rows.pop()
                    return False
                
                start = row[-1] + 1
                for val in range(start, found + 1):
                    valid = True
                    new_diffs = []
                    for i in range(pos):
                        d = val - row[i]
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    row.append(val)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    if build_row(pos + 1):
                        return True
                    
                    row.pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
                
                return False
            
            return build_row(1)
        
        if backtrack_row(0):
            best_rows = rows
            best_M = found
    
    return best_rows
# EVOLVE-BLOCK-END
