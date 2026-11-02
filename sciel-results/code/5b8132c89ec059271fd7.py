import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 140  # 10 second margin before 150s kill
    
    def get_max_val(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)
    
    def is_valid(rows, k):
        """Check if all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def greedy_row(used, k, randomize=False, bias=0):
        """
        Construct a single row of k+1 elements starting at 0,
        given the used differences. Returns the row or None if impossible.
        """
        row = [0]
        x = 0
        for _ in range(k):
            found = False
            # Try increasing x
            while True:
                x += 1
                # Check if x is too large (safety)
                if x > 500:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                if randomize and bias > 0 and random.random() < bias:
                    # Skip some values to randomize
                    x += random.randint(1, 3)
        return row
    
    def construct_solution(seed_val):
        """Construct a full solution using greedy with optional randomization."""
        random.seed(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            # Try with some randomization
            row = greedy_row(used, k, randomize=True, bias=0.3)
            if row is None:
                # Fallback to pure greedy
                row = greedy_row(used, k, randomize=False, bias=0)
                if row is None:
                    return None
            rows.append(row)
        return rows
    
    def local_improve(rows, max_iter=1000):
        """
        Try to improve the solution by reducing the max value.
        Uses a simple approach: reconstruct the row with the largest max value.
        """
        if not rows:
            return rows
        current_rows = [list(r) for r in rows]
        current_max = get_max_val(current_rows)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find the row with the largest max value
            max_idx = max(range(len(current_rows)), key=lambda i: max(current_rows[i]))
            max_val = max(current_rows[max_idx])
            
            if max_val <= 105:  # Can't do better than counting bound
                break
            
            # Remove this row's differences from used
            used = set()
            for i, row in enumerate(current_rows):
                if i != max_idx:
                    for a in range(len(row)):
                        for b in range(a+1, len(row)):
                            used.add(row[b] - row[a])
            
            # Try to reconstruct this row with a lower max
            best_row = None
            for attempt in range(50):
                random.seed(int(time.time()*1000) + attempt)
                row = greedy_row(used, k, randomize=True, bias=0.5)
                if row is not None:
                    new_max = max(row)
                    if best_row is None or new_max < max(best_row):
                        best_row = row
                        if new_max < max_val:
                            break
            
            if best_row is not None and max(best_row) < max_val:
                current_rows[max_idx] = best_row
                current_max = get_max_val(current_rows)
                if current_max <= 105:
                    break
            else:
                # No improvement found, try a different row
                pass
        
        return current_rows
    
    # Phase 1: Generate initial solutions
    best_solution = None
    best_max = float('inf')
    
    # First, try the pure greedy (like seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used, k, randomize=False, bias=0)
        if row is None:
            break
        rows.append(row)
    if len(rows) == n_rows:
        best_solution = rows
        best_max = get_max_val(rows)
    
    # Phase 2: Try randomized constructions
    for seed_val in range(1000):
        if time.time() > deadline:
            break
        rows = construct_solution(seed_val)
        if rows is not None and len(rows) == n_rows:
            curr_max = get_max_val(rows)
            if curr_max < best_max:
                best_max = curr_max
                best_solution = rows
                if best_max <= 105:
                    break
    
    # Phase 3: Local improvement
    if best_solution is not None:
        improved = local_improve(best_solution, max_iter=500)
        if improved is not None:
            imp_max = get_max_val(improved)
            if imp_max < best_max:
                best_max = imp_max
                best_solution = improved
    
    if best_solution is None:
        # Fallback: should not happen
        return [[0] + list(range(1, k+1)) for _ in range(n_rows)]
    
    return best_solution
