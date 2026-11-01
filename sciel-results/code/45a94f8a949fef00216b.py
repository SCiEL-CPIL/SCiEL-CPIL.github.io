import time
import random
import copy

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _generate_golomb_rulers(max_val):
    """
    Generate all Golomb rulers of order 6 (6 marks, starting at 0) 
    with max entry <= max_val.
    Returns a list of (row_tuple, diff_mask).
    """
    rulers = []
    
    def dfs(pos, last_val, mask, row):
        if pos == 6:
            rulers.append((tuple(row), mask))
            return
        
        # Optimization: If we have already used many small diffs, 
        # maybe we should jump? 
        # For now, simple iteration.
        for v in range(last_val + 1, max_val + 1):
            new_mask = 0
            valid = True
            # Check conflicts with existing diffs in this row
            for i in range(pos):
                d = v - row[i]
                bit = 1 << d
                if mask & bit:
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                row.append(v)
                dfs(pos + 1, v, mask | new_mask, row)
                row.pop()
                
    dfs(1, 0, 0, [0])
    return rulers

def _select_disjoint_rows(rulers, n_needed, deadline):
    """
    Select n_needed rows from rulers such that their diff masks are disjoint.
    Returns the list of selected rows or None.
    """
    if len(rulers) < n_needed:
        return None
    
    # Heuristic: Sort rulers by max value (ascending) to try smaller ones first?
    # Or randomize? 
    # Let's try sorting by max value first.
    # Actually, for disjointness, it might be better to pick rows that use "rare" diffs.
    # But we don't know rarity yet.
    # Let's just shuffle to avoid systematic bias, or sort by max val.
    # Sorting by max val ascending is a good bet for minimizing the global max, 
    # but since we are already bounding by M, any solution is valid for this M.
    
    # To speed up search, we can use a heuristic: 
    # Rows with larger max values might have larger differences, which are less likely to conflict.
    # So maybe sort by max value descending? 
    # Let's try a mix: Sort by max value, but maybe randomize slightly.
    
    # For now, let's just use the generated order (which is ascending max val due to DFS).
    # This might be slow if small rows conflict heavily.
    # Let's try sorting by "popularity" of differences? 
    # Too complex. Let's stick to simple DFS with a time check.
    
    selected = []
    used_mask = 0
    n_rulers = len(rulers)
    
    # Pre-filter: If a ruler's mask is a subset of another's? No, they all have 15 bits.
    
    def dfs(idx, count):
        if time.time() > deadline:
            return None
        
        if count == n_needed:
            return selected[:]
        
        # Pruning: Not enough rulers left
        if idx + (n_needed - count) > n_rulers:
            return None
        
        # Option 1: Skip rulers[idx]
        res = dfs(idx + 1, count)
        if res is not None:
            return res
        
        # Option 2: Take rulers[idx]
        row, mask = rulers[idx]
        if used_mask & mask == 0:
            selected.append(row)
            # Update used_mask (pass by value or use nonlocal/closure)
            # Since used_mask is an int, we can pass it or use a list/variable in outer scope.
            # Using nonlocal in nested function is cleaner.
            nonlocal used_mask
            old_mask = used_mask
            used_mask |= mask
            
            res = dfs(idx + 1, count + 1)
            if res is not None:
                return res
            
            used_mask = old_mask
            selected.pop()
            
        return None
    
    # To improve performance, we can shuffle the rulers list before DFS
    # to avoid getting stuck in a bad ordering.
    # But we want to be deterministic? No, random is fine.
    # Let's try a few random shuffles if the first one fails?
    # Or just one shuffle.
    
    # Actually, a better heuristic for Set Packing:
    # Pick the set that has the fewest options?
    # Here, all sets have the same size (15 bits).
    # Let's just shuffle.
    shuffled_rulers = rulers[:]
    random.shuffle(shuffled_rulers)
    
    # Reset state for the new list
    selected = []
    used_mask = 0
    n_rulers = len(shuffled_rulers)
    
    def dfs_shuffled(idx, count):
        if time.time() > deadline:
            return None
        
        if count == n_needed:
            return selected[:]
        
        if idx + (n_needed - count) > n_rulers:
            return None
        
        # Option 1: Skip
        res = dfs_shuffled(idx + 1, count)
        if res is not None:
            return res
        
        # Option 2: Take
        row, mask = shuffled_rulers[idx]
        if used_mask & mask == 0:
            selected.append(row)
            nonlocal used_mask
            old_mask = used_mask
            used_mask |= mask
            
            res = dfs_shuffled(idx + 1, count + 1)
            if res is not None:
                return res
            
            used_mask = old_mask
            selected.pop()
            
        return None
        
    return dfs_shuffled(0, 0)

def _greedy_fallback(n_rows, k):
    """Simple greedy construction to ensure we return something."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            found = False
            for x in range(row[-1] + 1, 500):
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def construct(n_rows, k):
    deadline = time.time() + 140
    rng = random.Random(42)
    
    # Start from the theoretical lower bound or slightly above
    # Counting bound: 105
    # Best known: 112
    start_m = 105
    
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Try exact bitmask solver for increasing M
    for M in range(start_m, 150):
        if time.time() > deadline - 10:
            break
            
        # Generate all valid rulers <= M
        # This can be expensive if M is large, but for M=105..120 it's okay.
        # Optimization: If we generated for M-1, we could reuse? 
        # But the masks are different. 
        # Let's just regenerate.
        
        # To speed up, we can limit the number of rulers generated?
        # No, we need all to ensure we don't miss a combination.
        
        rulers = _generate_golomb_rulers(M)
        
        if len(rulers) < n_rows:
            continue
        
        # Try to find disjoint rows
        # Use a sub-deadline for the selection
        sel_deadline = min(deadline - 5, time.time() + 5)
        rows = _select_disjoint_rows(rulers, n_rows, sel_deadline)
        
        if rows is not None:
            # Validate just in case
            if _validate(rows):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best_rows = [list(r) for r in rows]
                # If we found a solution for M, and M is close to lower bound, 
                # we might want to stop? 
                # The task is to minimize max entry. 
                # If we found a solution with max entry <= M, 
                # and we are iterating M upwards, the first solution found 
                # has the smallest possible max entry (bounded by M).
                # Wait, the max entry of the found rows might be < M.
                # But we are iterating M. 
                # If we find a solution for M=105, its max entry is <= 105.
                # If we find a solution for M=106, its max entry is <= 106.
                # We want the minimum max entry.
                # So if we find a solution for M, we can stop?
                # Yes, because any solution for M+1 will have max entry <= M+1, 
                # which is not better than a solution with max entry <= M.
                # However, the solution for M might have max entry 105, 
                # and a solution for M+1 might have max entry 105 too? 
                # No, we want the global minimum.
                # If we find a valid set for M, the max entry is at most M.
                # Is it possible that a solution with max entry 104 exists?
                # We started at 105. 
                # So if we find a solution for M=105, the max entry is <= 105.
                # We can return it.
                # But wait, the rows generated are <= M. 
                # So the max entry of the solution is <= M.
                # Since we iterate M from low to high, the first success gives the min M.
                # The max entry of the returned rows will be the actual max entry of those rows, 
                # which is <= M.
                # This is the optimal or near-optimal solution.
                break
    
    # Phase 2: Fallback if bitmask solver failed or timed out
    if best_rows is None:
        # Try a few random greedy attempts
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            rows = _greedy_fallback(n_rows, k)
            if rows is not None and _validate(rows):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best_rows = rows
    
    if best_rows is None:
        # Emergency fallback
        best_rows = [[0, 1, 3, 7, 15, 31]] * n_rows
        # This is invalid, but better than crashing? 
        # Let's try to make a valid one quickly.
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                for x in range(row[-1] + 1, 1000):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
            best_rows.append(row)
            
    return best_rows
