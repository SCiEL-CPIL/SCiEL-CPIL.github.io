import random
import time
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct a difference triangle set.
    n_rows: number of rows
    k: number of intervals, so k+1 points per row.
    Returns a list of n_rows rows, each a list of k+1 integers.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    # Total number of differences per row
    num_diffs_per_row = k * (k + 1) // 2
    
    def get_row_diffs(row):
        """Compute the set of positive differences for a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid_dts(rows):
        """Check if the set of rows forms a valid DTS."""
        all_diffs = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            row_diffs = get_row_diffs(row)
            if len(row_diffs) != num_diffs_per_row:
                return False
            if all_diffs & row_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    def greedy_construct():
        """Deterministic greedy construction."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                for x in range(row[-1] + 1, 500):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows
    
    def random_greedy_construct():
        """Randomized greedy construction."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                candidates = []
                for x in range(row[-1] + 1, 400):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                        if len(candidates) > 10:
                            break
                if not candidates:
                    return None
                # Pick randomly from the first few candidates
                choice = random.choice(candidates[:5])
                row.append(choice)
                new_diffs = {choice - a for a in row[:-1]}
                used_diffs |= new_diffs
            rows.append(row)
        return rows
    
    def local_search(rows, max_time=10.0):
        """
        Local search to minimize the maximum entry.
        Tries to reduce each element of each row.
        """
        if rows is None:
            return None
        
        rows = [r[:] for r in rows]
        current_max = max(max(r) for r in rows)
        best_rows = rows[:]
        best_max = current_max
        start_ls = time.time()
        
        # Precompute all differences
        all_used = set()
        for r in rows:
            all_used |= get_row_diffs(r)
        
        # Iteratively try to reduce the largest element
        while time.time() - start_ls < max_time:
            # Find the row with the largest maximum
            max_row_idx = -1
            max_val = -1
            for i, r in enumerate(rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_row_idx = i
            
            if max_row_idx == -1:
                break
            
            row = rows[max_row_idx]
            improved = False
            
            # Try to reduce the last element
            old_val = row[-1]
            min_x = row[-2] + 1
            
            # Try values from min_x to old_val - 1
            for x in range(min_x, old_val):
                if x in row:
                    continue
                
                new_diffs_last = {x - a for a in row[:-1]}
                old_diffs_last = {old_val - a for a in row[:-1]}
                
                # Check for conflicts
                # A conflict occurs if any d in new_diffs_last is in all_used 
                # and d is not in old_diffs_last (since we're freeing those)
                has_conflict = False
                for d in new_diffs_last:
                    if d in all_used and d not in old_diffs_last:
                        has_conflict = True
                        break
                
                if not has_conflict:
                    # Valid move
                    row[-1] = x
                    rows[max_row_idx] = row
                    
                    # Update all_used
                    all_used -= old_diffs_last
                    all_used |= new_diffs_last
                    
                    current_max = max(max(r) for r in rows)
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in rows]
                    
                    improved = True
                    break
            
            if not improved:
                # Try to reduce other elements? Or try a different row?
                # For simplicity, if we can't reduce the current max, we might try
                # to swap or move other elements. But for now, let's try to reduce
                # the second-to-last element to potentially allow reducing the last.
                
                # Try to reduce row[-2]
                old_val2 = row[-2]
                min_x2 = row[-3] + 1 if len(row) > 2 else 1
                max_x2 = row[-1] - 1
                
                found_move = False
                for x in range(min_x2, max_x2 + 1):
                    if x in row:
                        continue
                    
                    # Compute new diffs involving x
                    new_diffs_x = set()
                    for a in row:
                        if a != x:
                            new_diffs_x.add(abs(x - a))
                    
                    old_diffs_x = set()
                    for a in row:
                        if a != old_val2:
                            old_diffs_x.add(abs(old_val2 - a))
                    
                    has_conflict = False
                    for d in new_diffs_x:
                        if d in all_used and d not in old_diffs_x:
                            has_conflict = True
                            break
                    
                    if not has_conflict:
                        # Valid move
                        # Find index of old_val2
                        idx = row.index(old_val2)
                        row[idx] = x
                        row.sort()
                        rows[max_row_idx] = row
                        
                        all_used -= old_diffs_x
                        all_used |= new_diffs_x
                        
                        current_max = max(max(r) for r in rows)
                        if current_max < best_max:
                            best_max = current_max
                            best_rows = [r[:] for r in rows]
                        
                        found_move = True
                        break
                
                if not found_move:
                    # Stuck, break out
                    break
        
        return best_rows
    
    # Main loop
    best_solution = None
    best_score = float('inf')
    
    # 1. Try deterministic greedy first
    rows = greedy_construct()
    if rows is not None:
        score = max(max(r) for r in rows)
        if score < best_score:
            best_score = score
            best_solution = rows[:]
    
    # 2. Try randomized greedy and local search
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        rows = random_greedy_construct()
        if rows is None:
            continue
        
        score = max(max(r) for r in rows)
        
        if score < best_score:
            best_score = score
            best_solution = rows[:]
        
        # Apply local search
        improved_rows = local_search(rows, max_time=min(5.0, deadline - time.time()))
        if improved_rows is not None:
            improved_score = max(max(r) for r in improved_rows)
            if improved_score < best_score:
                best_score = improved_score
                best_solution = improved_rows[:]
        
        # If we have a very good solution, do more polishing
        if best_score <= 115:
            polished = local_search(best_solution, max_time=min(10.0, deadline - time.time()))
            if polished is not None:
                polished_score = max(max(r) for r in polished)
                if polished_score < best_score:
                    best_score = polished_score
                    best_solution = polished[:]
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Fallback
    if best_solution is None:
        rows = greedy_construct()
        if rows is not None:
            return rows
        else:
            # Should not happen, but provide a fallback
            return [[0] + list(range(1, k + 1)) for _ in range(n_rows)]
    
    return best_solution
