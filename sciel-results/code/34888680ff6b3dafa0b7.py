import time
import random

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    timeout = 145  # Safety margin for 150s limit
    
    # Helper to calculate differences of a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Greedy construction of a single row given a set of forbidden differences
    # Returns the row list
    def greedy_row(forbidden, size):
        row = [0]
        row_diffs = set()
        for _ in range(size - 1):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                # Check if new_diffs are distinct (they are, since row is strictly increasing and x is new)
                # Actually, x - a are distinct for distinct a.
                # Check if they conflict with forbidden or existing row_diffs
                if not (new_diffs & forbidden) and not (new_diffs & row_diffs):
                    row.append(x)
                    row_diffs |= new_diffs
                    break
                x += 1
        return row

    # Full construction with a specific order
    def construct_full(order, size):
        used = set()
        rows = [None] * len(order)
        for idx, row_idx in enumerate(order):
            r = greedy_row(used, size)
            rows[row_idx] = r
            used |= get_diffs(r)
        return rows

    best_rows = None
    best_max = float('inf')
    
    # Initial best from simple greedy (0..n-1)
    init_order = list(range(n_rows))
    init_rows = construct_full(init_order, k + 1)
    init_max = max(max(r) for r in init_rows)
    best_rows = init_rows
    best_max = init_max
    
    # Local search function
    def local_search(rows):
        nonlocal best_max
        rows = [r[:] for r in rows] # copy
        current_max = max(max(r) for r in rows)
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            # Randomize order of rows to try for re-greedy
            indices = list(range(len(rows)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > timeout:
                    break
                
                # Remove current row's diffs
                old_diffs = get_diffs(rows[i])
                # We need to know the 'forbidden' set from OTHER rows
                # Instead of maintaining a global set, let's just recompute or maintain carefully.
                # For simplicity and speed in this small case, recompute forbidden set.
                forbidden = set()
                for j in range(len(rows)):
                    if j != i:
                        forbidden |= get_diffs(rows[j])
                
                new_row = greedy_row(forbidden, k + 1)
                new_max_i = max(new_row)
                old_max_i = max(rows[i])
                
                if new_max_i < old_max_i:
                    rows[i] = new_row
                    current_max = max(max(r) for r in rows)
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in rows]
        
        return rows, current_max

    # Main search loop
    # Try many random orders
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        # Random permutation
        order = list(range(n_rows))
        random.shuffle(order)
        
        rows = construct_full(order, k + 1)
        current_max = max(max(r) for r in rows)
        
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in rows]
            if best_max == 105: # Lower bound
                break
        
        # Apply local search
        if time.time() - start_time < timeout - 5: # Leave time for final check
             _, _ = local_search(rows)
             
    return best_rows

