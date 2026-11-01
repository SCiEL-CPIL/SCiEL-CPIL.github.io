import time
import random
import math

def _greedy_init(n_rows, k, scope_limit):
    """Greedy initialization: build rows one by one, choosing smallest valid values."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            for x in range(row[-1] + 1, scope_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
                    break
            else:
                return None  # Failed to complete
        rows.append(row)
    return rows

def _row_diffs_set(row):
    """Compute all pairwise differences for a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute all used differences across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _global_max(rows):
    """Return the maximum value across all rows."""
    return max(max(r) for r in rows)

def _build_static_diffs(rows):
    """Build a dict mapping (row_idx, pos_idx) to the set of diffs that position contributes."""
    # For each position (r, p), the diffs it creates are:
    #   row[r][p] - row[r][q] for all q < p  (differences with earlier positions in same row)
    #   row[r][p] - row[r][q] for all q > p  (differences with later positions in same row)
    # Actually, for a single position change, we need to know which diffs are "owned" by that position.
    # Let's define: for position (r, p), its owned diffs are:
    #   {row[r][p] - row[r][q] for q in range(p)}  -- diffs where this position is the larger
    #   {row[r][q] - row[r][p] for q in range(p+1, len(row))}  -- diffs where this position is the smaller
    # But when we change row[r][p], all these diffs change.
    # For efficiency, let's precompute for each position the list of other positions in the same row.
    
    pos_info = {}
    for r in range(len(rows)):
        row = rows[r]
        for p in range(len(row)):
            # Diffs involving position (r, p):
            # With positions q < p: diff = row[p] - row[q]
            # With positions q > p: diff = row[q] - row[p]
            before = list(range(p))
            after = list(range(p + 1, len(row)))
            pos_info[(r, p)] = (before, after)
    return pos_info

def _compute_new_diffs_fast(row, p, new_val):
    """Compute the new set of diffs for position p if set to new_val, given the rest of the row."""
    diffs = set()
    for q in range(len(row)):
        if q == p:
            continue
        if q < p:
            d = new_val - row[q]
        else:
            d = row[q] - new_val
        if d > 0:
            diffs.add(d)
    return diffs

def _is_move_valid(rows, r, p, new_val, used_diffs, old_diffs_for_pos):
    """Check if setting rows[r][p] = new_val is valid (no duplicate diffs)."""
    row = rows[r]
    if new_val <= row[p - 1] if p > 0 else new_val <= 0:
        return False
    if p < len(row) - 1 and new_val >= row[p + 1]:
        return False
    
    new_diffs = _compute_new_diffs_fast(row, p, new_val)
    # Remove old diffs, check for conflicts with remaining used_diffs
    remaining = used_diffs - old_diffs_for_pos
    if new_diffs & remaining:
        return False
    return True

def _get_new_diffs_set(rows, r, p, new_val):
    """Get the set of diffs for position (r, p) if set to new_val."""
    row = rows[r]
    diffs = set()
    for q in range(len(row)):
        if q == p:
            continue
        if q < p:
            d = new_val - row[q]
        else:
            d = row[q] - new_val
        if d > 0:
            diffs.add(d)
    return diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin before 120s limit
    
    # Counting bound
    lower_bound = 105
    
    # Start with a greedy solution
    scope_limit = 150
    best_rows = _greedy_init(n_rows, k, scope_limit)
    if best_rows is None:
        # Try larger scope
        scope_limit = 200
        best_rows = _greedy_init(n_rows, k, scope_limit)
    if best_rows is None:
        # Fallback: very large scope
        scope_limit = 300
        best_rows = _greedy_init(n_rows, k, scope_limit)
    if best_rows is None:
        # Last resort: just return something valid with large scope
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_scope = _global_max(best_rows)
    best_solution = [list(r) for r in best_rows]
    
    # Local search with simulated annealing
    # State: rows, used_diffs
    # Move: change one position (r, p) to a new value
    # Objective: minimize max value
    
    random.seed(42)
    
    # Precompute position info
    pos_info = _build_static_diffs(best_rows)
    
    # For each position, precompute the set of diffs it currently contributes
    def get_pos_diffs(rows, r, p):
        row = rows[r]
        diffs = set()
        for q in range(len(row)):
            if q == p:
                continue
            if q < p:
                d = row[p] - row[q]
            else:
                d = row[q] - row[p]
            if d > 0:
                diffs.add(d)
        return diffs
    
    # Initial used diffs
    used_diffs = _compute_all_used(best_rows)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    current_rows = [list(r) for r in best_rows]
    current_used = set(used_diffs)
    current_scope = _global_max(current_rows)
    
    # Precompute all positions
    all_positions = [(r, p) for r in range(n_rows) for p in range(1, k + 1)]
    
    step = 0
    while time.time() < deadline:
        step += 1
        if step % 10000 == 0 and current_scope <= lower_bound:
            break
        
        # Choose a random position to modify
        r, p = random.choice(all_positions)
        row = current_rows[r]
        old_val = row[p]
        
        # Determine valid range for new value
        min_val = row[p - 1] + 1 if p > 0 else 1
        max_val = row[p + 1] - 1 if p < len(row) - 1 else best_scope + 20
        
        if max_val <= min_val:
            continue
        
        # Try a few random new values
        found_move = False
        for _ in range(3):
            new_val = random.randint(min_val, max_val)
            if new_val == old_val:
                continue
            
            # Check validity
            new_diffs = _compute_new_diffs_fast(row, p, new_val)
            old_diffs = get_pos_diffs(current_rows, r, p)
            remaining = current_used - old_diffs
            if new_diffs & remaining:
                continue
            
            # Compute new scope
            new_scope = current_scope
            if new_val > current_scope:
                new_scope = new_val
            
            # Acceptance criterion
            delta_scope = new_scope - current_scope
            if delta_scope <= 0 or random.random() < math.exp(-delta_scope / T):
                # Make the move
                row[p] = new_val
                current_used = (current_used - old_diffs) | new_diffs
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_solution = [list(rr) for rr in current_rows]
                    if best_scope <= lower_bound:
                        break
                
                found_move = True
                break
        
        if best_scope <= lower_bound:
            break
        
        # Cool down
        T = max(T_min, T * cooling)
    
    return best_solution
