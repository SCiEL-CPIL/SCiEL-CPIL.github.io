import time
import random

def construct(n_rows, k):
        start_time = time.time()
        deadline = start_time + 110 # 10s buffer

        # 1. Greedy to get upper bound
        def greedy():
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                row_diffs = set()
                last = 0
                while len(row) < k + 1:
                    found = False
                    # Try small values first
                    for v in range(last + 1, 1000): # Upper bound can be dynamic
                        new_diffs = {v - a for a in row}
                        if not (new_diffs & used) and not (new_diffs & row_diffs):
                            row.append(v)
                            row_diffs |= new_diffs
                            used |= new_diffs
                            last = v
                            found = True
                            break
                    if not found:
                        # If we go too high, relax? No, greedy should work for reasonable bounds
                        # But we need a hard stop for the search space size
                        return None
                rows.append(row)
            return rows

        greedy_rows = greedy()
        if not greedy_rows:
            # Fallback: return empty or dummy? Task implies valid output.
            # Greedy with 1000 limit should work for n=4, k=7.
            # If not, increase limit.
            return greedy_rows # Might be None, handle later

        M_high = max(max(r) for r in greedy_rows)
        
        # 2. Lower bound
        K = k * (k + 1) // 2
        M_low = n_rows * K

        # 3. Binary Search on M
        best_rows = greedy_rows
        best_M = M_high

        # We want to find the smallest M.
        # Binary search range: [M_low, M_high]
        
        def is_feasible(M, time_limit):
            # DFS with MRV
            rows = [[0] for _ in range(n_rows)]
            used_diffs = set()
            
            # State for DFS
            # We use a recursive function with explicit time check
            
            def dfs():
                if time.time() > time_limit:
                    return False # Timeout treated as infeasible for this M in this attempt
                
                # Check completion
                all_complete = True
                for r in range(n_rows):
                    if len(rows[r]) < k + 1:
                        all_complete = False
                        break
                if all_complete:
                    return True
                
                # Calculate candidates for each incomplete row
                min_candidates = None
                min_row_idx = -1
                row_candidates = [None] * n_rows
                
                for r in range(n_rows):
                    if len(rows[r]) == k + 1:
                        continue
                    
                    last = rows[r][-1]
                    # Pruning: check if we can fit remaining elements
                    remaining = k + 1 - len(rows[r])
                    if last + remaining > M:
                        return False # Cannot fit even with min increment 1
                    
                    cands = []
                    # Iterate v from last+1 to M
                    # Optimization: We can break early if no candidates?
                    for v in range(last + 1, M + 1):
                        new_diffs = {v - a for a in rows[r]}
                        # Check conflicts
                        if new_diffs & used_diffs:
                            continue
                        # Check local conflicts (already in rows[r])
                        # new_diffs are differences from v to previous elements.
                        # These are new to the row, so they only conflict with used_diffs (global)
                        # Wait, do they conflict with other diffs in the same row?
                        # The diffs in the same row are generated as we add elements.
                        # When we add v, we generate {v - a_i}.
                        # These must not be in used_diffs (global).
                        # Do they need to be distinct from *other* diffs in the same row?
                        # Yes, the problem says "all differences ... are distinct over all rows".
                        # This implies all differences in the entire set of rows must be distinct.
                        # So `used_diffs` contains ALL diffs from ALL rows so far.
                        # So checking `new_diffs & used_diffs` is sufficient.
                        
                        cands.append((v, new_diffs))
                    
                    if not cands:
                        return False # Dead end for this row
                    
                    row_candidates[r] = cands
                    if min_candidates is None or len(cands) < len(min_candidates):
                        min_candidates = cands
                        min_row_idx = r
                
                if min_row_idx == -1:
                    return True # Should have been caught by all_complete
                
                # Try candidates for min_row_idx
                # Heuristic: Try smaller v first
                for v, new_diffs in min_candidates:
                    # Apply
                    rows[min_row_idx].append(v)
                    used_diffs.update(new_diffs)
                    
                    if dfs():
                        return True
                    
                    # Backtrack
                    rows[min_row_idx].pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
                
                return False
            
            return dfs()

        # Binary Search Loop
        # To handle timeouts gracefully, we might want to try M values in increasing order?
        # No, binary search is better.
        # If a M is infeasible, we move low up.
        # If a M is feasible, we update best and move high down.
        
        low = M_low
        high = M_high
        
        # We will keep a record of the best feasible M and its rows.
        # Since DFS doesn't return the rows directly in the boolean check,
        # we need to modify dfs to return the rows or store them globally.
        
        # Let's modify is_feasible to return rows if found.
        
        def solve_for_M(M, time_limit):
            rows = [[0] for _ in range(n_rows)]
            used_diffs = set()
            found_rows = [None]
            
            def dfs():
                if time.time() > time_limit:
                    return False
                
                all_complete = True
                for r in range(n_rows):
                    if len(rows[r]) < k + 1:
                        all_complete = False
                        break
                if all_complete:
                    found_rows[0] = [list(r) for r in rows]
                    return True
                
                min_candidates = None
                min_row_idx = -1
                row_candidates = [None] * n_rows
                
                for r in range(n_rows):
                    if len(rows[r]) == k + 1:
                        continue
                    
                    last = rows[r][-1]
                    remaining = k + 1 - len(rows[r])
                    if last + remaining > M:
                        return False
                    
                    cands = []
                    for v in range(last + 1, M + 1):
                        new_diffs = {v - a for a in rows[r]}
                        if not (new_diffs & used_diffs):
                            cands.append((v, new_diffs))
                    
                    if not cands:
                        return False
                    
                    row_candidates[r] = cands
                    if min_candidates is None or len(cands) < len(min_candidates):
                        min_candidates = cands
                        min_row_idx = r
                
                if min_row_idx == -1:
                    return True
                
                # Heuristic: Sort by v
                for v, new_diffs in min_candidates:
                    rows[min_row_idx].append(v)
                    used_diffs.update(new_diffs)
                    
                    if dfs():
                        return True
                    
                    rows[min_row_idx].pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
                
                return False
            
            if dfs():
                return found_rows[0]
            return None

        # Binary Search
        # We iterate M from M_low to M_high?
        # Binary search is O(log(range)).
        # However, if DFS is slow for some M, binary search might waste time on infeasible M's.
        # Given the small range (112 to ~134), linear search from M_low upwards might be safer and simpler.
        # 134 - 112 = 22 values.
        # If each DFS takes < 1s, 22s is fine.
        # Let's try linear search from M_low to M_high.
        
        for M in range(M_low, M_high + 1):
            if time.time() > deadline - 5:
                break
            
            # Time limit per M: 2 seconds?
            # If we find a solution, we return it immediately as it's the minimal M.
            t_start = time.time()
            limit = t_start + 5 # 5s per M
            result = solve_for_M(M, limit)
            if result:
                return result
        
        # If no better M found, return greedy
        return best_rows
