import time
import random
import sys

def _row_diffs(row):
    """Return the set of all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _scope_of(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows) if rows else 0

def _validate(rows, n_rows, k):
    """Check if the solution is valid."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        row_d = _row_diffs(row)
        if row_d & used:
            return False
        used |= row_d
    return True

def _pure_greedy(n_rows, k, seed=0):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _simulated_annealing(n_rows, k, start_rows, time_limit, seed=42):
    """
    Simulated Annealing to minimize the scope.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Deep copy start rows
    current_rows = [list(r) for r in start_rows]
    current_scope = _scope_of(current_rows)
    
    # Global state for differences
    used_diffs = set()
    row_diffs = []
    for r in current_rows:
        d = _row_diffs(r)
        row_diffs.append(d)
        used_diffs |= d
        
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # Initial temperature
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random row and a random index to modify
        # We exclude index 0 (which is always 0)
        row_idx = rng.randrange(n_rows)
        idx = rng.randrange(1, k + 1) # 1 to k inclusive
        
        old_val = current_rows[row_idx][idx]
        
        # Determine valid range for new_val
        min_val = current_rows[row_idx][idx - 1] + 1
        if idx < k:
            max_val = current_rows[row_idx][idx + 1] - 1
        else:
            # For the last element, we can try to go up to current_scope or slightly higher
            # But to improve, we should mostly look downwards or nearby
            max_val = current_scope + 10 
            
        if min_val > max_val:
            T *= cooling_rate
            continue
            
        # Randomly choose a new value in the range
        # Bias towards smaller values to reduce scope
        if rng.random() < 0.5:
            new_val = rng.randint(min_val, max_val)
        else:
            # Try to move towards the lower bound
            new_val = rng.randint(min_val, min(max_val, old_val + 5))
            
        if new_val == old_val:
            continue
            
        # Check validity
        new_row = current_rows[row_idx][:]
        new_row[idx] = new_val
        
        # Check strictly increasing
        valid = True
        for i in range(len(new_row) - 1):
            if new_row[i + 1] <= new_row[i]:
                valid = False
                break
        
        if not valid:
            continue
            
        new_diffs = _row_diffs(new_row)
        old_diffs = row_diffs[row_idx]
        
        # Remove old diffs from global set
        used_diffs -= old_diffs
        
        # Check for conflicts
        if new_diffs & used_diffs:
            # Conflict, revert
            used_diffs |= old_diffs
            continue
            
        # No conflict, calculate new scope
        new_scope = max(max(r) for r in current_rows)
        # Note: current_rows[row_idx] is still the old one for other rows, 
        # but we need to check the max of the new row too
        # Actually, we can just compute scope of the hypothetical state
        # To be efficient, we can just check if new_row's max is the new global max
        # But for simplicity and correctness, let's just update state if accepted.
        
        # Determine acceptance
        if new_scope < current_scope:
            accept = True
        elif new_scope == current_scope:
            accept = True # Accept neutral moves to explore
        else:
            # Worse scope
            delta = new_scope - current_scope
            if T > 0:
                accept = (rng.random() < pow(2.71828, -delta / T))
            else:
                accept = False
                
        if accept:
            # Update state
            current_rows[row_idx] = new_row
            row_diffs[row_idx] = new_diffs
            used_diffs |= new_diffs
            current_scope = new_scope
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
                
        else:
            # Revert
            used_diffs |= old_diffs
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1 # Floor temperature
            
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set with minimum scope."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Get a strong baseline
    best_rows = _pure_greedy(n_rows, k)
    best_scope = _scope_of(best_rows)
    
    # 2. Run Simulated Annealing with multiple restarts/seeds
    # We will allocate most of the time to this
    
    # Try a few different starting points? 
    # Greedy is usually good enough as a start.
    # Let's run SA with a time budget.
    
    # We can run multiple SA instances if time permits, 
    # but one long SA run is often better than many short ones for escaping deep local minima.
    
    # Let's try a few different random seeds for the SA process itself
    # or restarts if it gets stuck.
    
    # Since SA is stochastic, we can just run it for the available time.
    # To be safe against getting stuck in a very bad local minimum from the start,
    # we could also try a "randomized greedy" start.
    
    # Let's try 2 different start strategies if time allows.
    
    # Strategy A: Pure Greedy Start
    time_left = deadline - time.time()
    if time_left > 10:
        sa_result = _simulated_annealing(n_rows, k, best_rows, time_left - 5, seed=42)
        if _validate(sa_result, n_rows, k):
            scope = _scope_of(sa_result)
            if scope < best_scope:
                best_rows = sa_result
                best_scope = scope
                
    # Strategy B: Randomized Greedy Start (if time remains)
    time_left = deadline - time.time()
    if time_left > 10:
        # Generate a slightly different greedy start
        used = set()
        rows = []
        rng = random.Random(123)
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Look ahead a bit
                best_x = None
                candidates = []
                for try_x in range(x + 1, x + 20):
                    new = {try_x - a for a in row}
                    if not (new & used):
                        candidates.append(try_x)
                if candidates:
                    # Pick randomly from first few valid
                    if len(candidates) > 1 and rng.random() < 0.3:
                        best_x = rng.choice(candidates[:5])
                    else:
                        best_x = candidates[0]
                    x = best_x
                    row.append(x)
                    used |= new
                else:
                    # Fallback to linear search
                    try_x = x + 1
                    while True:
                        new = {try_x - a for a in row}
                        if not (new & used):
                            candidates = [try_x]
                            break
                        try_x += 1
                    x = candidates[0]
                    row.append(x)
                    used |= new
            rows.append(row)
        
        start_scope = _scope_of(rows)
        if start_scope < best_scope:
            best_rows = rows
            best_scope = start_scope
            
        # Run SA on this start
        time_left = deadline - time.time()
        if time_left > 5:
            sa_result = _simulated_annealing(n_rows, k, best_rows, time_left - 2, seed=999)
            if _validate(sa_result, n_rows, k):
                scope = _scope_of(sa_result)
                if scope < best_scope:
                    best_rows = sa_result
                    best_scope = scope

    # Final validation
    if not _validate(best_rows, n_rows, k):
        best_rows = _pure_greedy(n_rows, k)
        
    return best_rows
