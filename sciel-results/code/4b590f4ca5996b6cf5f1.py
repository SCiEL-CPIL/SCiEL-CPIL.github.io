import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Lower bound for M
    # Total diffs = n_rows * (k+1)*k/2
    # Max diff = M
    # M >= total_diffs
    total_diffs = n_rows * (k + 1) * k // 2
    min_M = total_diffs
    
    # We know 134 is a solution. We can start from min_M and go up.
    # To be safe and fast, we can also try a known good heuristic first.
    # But let's try to find the optimal or near-optimal.
    
    # We will iterate M from min_M upwards.
    # For each M, we try to find a valid set.
    
    # To speed up, we can use a cache of valid rulers for a given M?
    # No, M changes.
    
    # Let's use a backtracking search for a given M.
    
    def search(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx):
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            # Try to construct row row_idx
            # Heuristic: try to keep the ruler as "dense" as possible?
            # Or try different starting points?
            # The row must start with 0.
            
            # We generate the row element by element.
            # State: current row, last element, used_diffs for this row (local)
            
            def build_row(current_row, last_elem, local_diffs):
                if time.time() > deadline:
                    return None
                
                if len(current_row) == k + 1:
                    # Check if local_diffs are disjoint from global used_diffs
                    if not local_diffs & used_diffs:
                        # Add to global
                        used_diffs.update(local_diffs)
                        rows.append(current_row[:])
                        res = backtrack(row_idx + 1)
                        if res is not None:
                            return res
                        # Backtrack
                        used_diffs.difference_update(local_diffs)
                        rows.pop()
                    return None
                
                # Try next element x
                # x must be > last_elem
                # x <= M
                # To prune, we can estimate the minimum possible span for the rest of the row.
                # But for now, just iterate.
                
                # Optimization: 
                # The differences x - a_i must be new in local_diffs and not in used_diffs.
                
                # Range for x:
                # Min x: last_elem + 1
                # Max x: M
                # Also, we must be able to fit the remaining (k + 1 - len(current_row)) elements.
                # The minimal additional span for r elements is r*(r+1)/2? No, for Golomb, it's larger.
                # For a Golomb ruler of length r, the minimal span is r*(r-1)/2 + 1? No.
                # Minimal span for length L (L marks) is L*(L-1)/2? No.
                # For L marks, there are L*(L-1)/2 differences.
                # The smallest Golomb ruler of length L has span roughly L^2/4?
                # Actually, the minimal span for a Golomb ruler of length n is n*(n-1)/2 is wrong.
                # For n=3 (0, 1, 3), span 3. Diffs: 1, 2, 3.
                # For n=4 (0, 1, 4, 6), span 6. Diffs: 1, 3, 5, 2, 5(no), 4. 
                # 0,1,4,6: diffs 1,4,6,3,5,2. All distinct. Span 6.
                # Formula for min span is not simple.
                
                # Let's just iterate x from last_elem + 1 to M.
                # To improve, we can shuffle the range or use a heuristic.
                # Greedy (smallest x) is a good start.
                
                for x in range(last_elem + 1, M + 1):
                    new_local_diffs = set()
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d in local_diffs or d in used_diffs:
                            valid = False
                            break
                        new_local_diffs.add(d)
                    
                    if valid:
                        res = build_row(current_row + [x], x, local_diffs | new_local_diffs)
                        if res is not None:
                            return res
                
                return None
            
            res = build_row([0], 0, set())
            return res
        
        return backtrack(0)

    # Try M from min_M to some upper bound
    # We know 134 works.
    # To be safe, go up to 150 or so.
    for M in range(min_M, 151):
        if time.time() > deadline:
            break
        # Randomize the order of rows? Or just try.
        # To break symmetry, we can enforce that the rows are lexicographically increasing.
        # But for now, let's just try.
        
        # We can add a bit of randomization to avoid local minima in the search order?
        # The search is deterministic.
        
        res = search(M)
        if res is not None:
            return res
    
    # If we run out of time or M, return the best we can?
    # The problem requires a valid set.
    # If we didn't find one, we should return a valid one even if suboptimal.
    # The greedy solution from the seed is valid.
    # Let's generate a greedy solution as a fallback.
    
    used, rows = set(), []
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
