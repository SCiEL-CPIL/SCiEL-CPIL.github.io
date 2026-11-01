import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set for n_rows rows of k+1 integers.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate initial valid solution using greedy
    initial_rows = _greedy_construct(n_rows, k)
    best_score = _get_scope(initial_rows)
    best_rows = [row[:] for row in initial_rows]
    
    # 2. Simulated Annealing to minimize scope
    # Cost = scope + C * conflicts
    # We want to minimize scope. We allow temporary increases in scope or conflicts.
    
    current_rows = [row[:] for row in initial_rows]
    current_score = _get_scope(current_rows)
    
    T = 50.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute some stats if possible? 
    # The cost function is: max(max(row) for row in rows) + 100 * num_conflicts
    # We need to calculate conflicts efficiently.
    
    def get_cost(rows):
        # Check validity and compute scope
        used = set()
        conflicts = 0
        max_scope = 0
        for row in rows:
            max_scope = max(max_scope, max(row) if row else 0)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        return max_scope + 100 * conflicts

    current_cost = get_cost(current_rows)
    
    # If initial is already optimal or we are running out of time, return
    if current_cost == best_score:
         # It's valid and we have a baseline.
         pass

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Generate a neighbor
        new_rows = _make_move(current_rows, k)
        new_cost = get_cost(new_rows)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0:
            # Better
            current_rows = new_rows
            current_cost = new_cost
            if current_cost == _get_scope(current_rows): # Valid
                if current_cost < best_score:
                    best_score = current_cost
                    best_rows = [row[:] for row in current_rows]
            # Even if invalid, if cost decreased, keep it to explore
            # But we only update best if valid (cost == scope)
        else:
            # Worse
            if T > T_min:
                # Accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    current_rows = new_rows
                    current_cost = new_cost
        
        # Cooling
        T = max(T_min, T * cooling_rate)
        
        # Occasionally restart or perturb strongly if stuck?
        # For now, simple SA.
        
        # Check time
        if time.time() > deadline:
            break

    # Ensure we return the best VALID solution found
    # The greedy one is always valid.
    # If SA found a better valid one, best_rows holds it.
    # If SA didn't find a better valid one, best_rows is the greedy one.
    
    return best_rows

def _greedy_construct(n_rows, k):
    """
    Greedy construction of rows.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = set()
            valid = True
            for a in row:
                d = x - a
                if d in used:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                row.append(x)
                used.update(new_diffs)
        
        rows.append(row)
    return rows

def _get_scope(rows):
    """
    Returns the maximum entry in any row. Assumes rows are valid.
    """
    scope = 0
    for row in rows:
        if row:
            scope = max(scope, row[-1])
    return scope

def _make_move(rows, k):
    """
    Pick a random row and modify one of its entries.
    To maintain strict increase and start at 0, we can only modify indices 1..k.
    We pick a random index and a random delta.
    """
    n_rows = len(rows)
    row_idx = random.randint(0, n_rows - 1)
    # Indices available to change: 1 to k (0 is fixed at 0)
    # If k is small, this is limited.
    if k == 0:
        return [row[:] for row in rows]
    
    pos_idx = random.randint(1, k)
    
    # Determine range for new value
    # Must be > rows[row_idx][pos_idx-1]
    # Must be < rows[row_idx][pos_idx+1] (if pos_idx < k)
    # Also generally bounded by current scope + some margin
    
    row = rows[row_idx]
    lo = row[pos_idx - 1] + 1
    hi = row[pos_idx + 1] - 1 if pos_idx < k else lo + 100 # Allow growth
    
    # If the range is invalid, maybe revert or pick a different move
    if hi < lo:
        # Cannot decrease below previous, cannot increase above next.
        # Try a different index or row? For now, just return a copy (no change)
        # or try a random other index.
        return [r[:] for r in rows]
    
    new_val = random.randint(lo, hi)
    
    # Create new rows
    new_rows = [r[:] for r in rows]
    new_rows[row_idx][pos_idx] = new_val
    
    # Check if it broke strict increase (shouldn't given lo/hi, but good to be safe)
    # And check if it broke the "start at 0" (we didn't touch 0)
    
    return new_rows
# EVOLVE-BLOCK-END
