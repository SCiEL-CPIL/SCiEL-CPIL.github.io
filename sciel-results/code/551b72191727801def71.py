import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Phase 1: Fast greedy construction with multiple restarts
    best_rows = None
    best_scope = float('inf')
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 60:
            break
        
        rows = _fast_greedy(n_rows, k, rng)
        if rows is not None and _is_valid_solution(rows, n_rows, k):
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        best_rows = _fast_greedy(n_rows, k, rng)
        best_scope = _scope(best_rows)
    
    # Phase 2: Bitset-accelerated simulated annealing
    if time.time() < deadline - 10:
        improved = _simulated_annealing(best_rows, n_rows, k, rng, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Targeted polish - try to reduce the max element
    if time.time() < deadline - 5:
        improved = _polish_max(best_rows, n_rows, k, rng, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _fast_greedy(n_rows, k, rng):
    """Fast greedy construction using bitset for used differences."""
    max_diff = 200
    used_mask = 0  # bitset of used differences
    
    rows = []
    for _ in range(n_rows):
        row = [0]
        for pos in range(k):
            last_val = row[-1]
            found = False
            # Try values from last_val+1 upward
            for val in range(last_val + 1, last_val + 30):
                # Check if all new differences are available
                valid = True
                new_diffs = []
                for prev in row:
                    d = val - prev
                    if d > max_diff:
                        valid = False
                        break
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    # Check if this difference already exists in current row
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    # Accept this value
                    row.append(val)
                    for d in new_diffs:
                        used_mask |= (1 << d)
                    found = True
                    break
            
            if not found:
                # Fallback: try harder
                val = last_val + 1
                while True:
                    valid = True
                    new_diffs = []
                    for prev in row:
                        d = val - prev
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(val)
                        for d in new_diffs:
                            used_mask |= (1 << d)
                        break
                    val += 1
        
        rows.append(row)
    
    return rows

def _is_valid_row(row):
    """Check if a row is strictly increasing with distinct internal differences."""
    n = len(row)
    for i in range(n - 1):
        if row[i + 1] <= row[i]:
            return False
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _is_valid_solution(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i+1] <= row[i]:
                return False
        diffs = _row_diffs(row)
        if len(diffs) != (k+1)*k//2:
            return False
        if all_diffs & diffs:
            return False
        all_diffs.update(diffs)
    return True

def _build_diff_bitset(rows, k):
    """Build a bitset where bit d is set if difference d is used by any row."""
    max_d = 256
    mask = 0
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d < max_d:
                    mask |= (1 << d)
    return mask

def _row_conflicts(row, other_rows, k):
    """Check if row has any conflicts with other_rows. Returns True if conflict exists."""
    row_diffs = _row_diffs(row)
    for other in other_rows:
        if other is row:
            continue
        other_diffs = _row_diffs(other)
        if row_diffs & other_diffs:
            return True
    return False

def _simulated_annealing(initial_rows, n_rows, k, rng, deadline):
    """
    Bitset-accelerated simulated annealing.
    Moves: replace an entire row with a newly constructed row.
    Acceptance: Metropolis criterion based on scope change.
    """
    rows = [list(r) for r in initial_rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute diff sets for all rows
    row_diffs = [_row_diffs(row) for row in rows]
    
    # Build global used diff set
    used_diffs = set()
    for rd in row_diffs:
        used_diffs.update(rd)
    
    temperature = 20.0
    cooling = 0.9995
    min_temp = 0.1
    
    step = 0
    while time.time() < deadline - 5 and temperature > min_temp:
        step += 1
        
        # Pick a random row to replace
        idx = rng.randint(0, n_rows - 1)
        
        # Generate a new candidate row
        new_row = _generate_candidate_row(k, used_diffs, rng, max_val=best_scope + 5)
        if new_row is None:
            temperature *= cooling
            continue
        
        new_row_diffs = _row_diffs(new_row)
        
        # Check if new row conflicts with other rows
        conflict = False
        for j in range(n_rows):
            if j == idx:
                continue
            if new_row_diffs & row_diffs[j]:
                conflict = True
                break
        
        if conflict:
            temperature *= cooling
            continue
        
        # Compute new scope
        old_scope = best_scope
        new_rows_list = [rows[j] if j != idx else new_row for j in range(n_rows)]
        new_scope = _scope(new_rows_list)
        
        delta = new_scope - old_scope
        
        # Metropolis acceptance
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            # Accept
            rows[idx] = new_row
            row_diffs[idx] = new_row_diffs
            
            # Update used_diffs
            used_diffs = set()
            for j in range(n_rows):
                used_diffs.update(row_diffs[j])
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        
        temperature *= cooling
    
    return best_rows

def _generate_candidate_row(k, used_diffs, rng, max_val=120):
    """
    Generate a valid row of k+1 elements starting at 0,
    with no differences conflicting with used_diffs.
    Returns None if no such row found within max_val.
    """
    row = [0]
    for pos in range(k):
        last_val = row[-1]
        found = False
        
        # Try values from last_val+1 upward, with some randomness
        candidates = []
        for val in range(last_val + 1, min(last_val + 25, max_val + 1)):
            valid = True
            new_diffs = set()
            for prev in row:
                d = val - prev
                if d in used_diffs:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                candidates.append((val, new_diffs))
        
        if not candidates:
            return None
        
        # Choose randomly from valid candidates, weighted toward smaller values
        if rng.random() < 0.7:
            # Prefer smaller values
            val, new_diffs = candidates[0]
        else:
            # Random choice
            val, new_diffs = rng.choice(candidates)
        
        row.append(val)
        used_diffs.update(new_diffs)
    
    # Verify the row is valid
    if not _is_valid_row(row):
        return None
    
    return row

def _polish_max(initial_rows, n_rows, k, rng, deadline):
    """
    Targeted polish: repeatedly try to reduce the maximum element
    by regenerating the row that contains it.
    """
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Build used diffs
    row_diffs = [_row_diffs(row) for row in rows]
    used_diffs = set()
    for rd in row_diffs:
        used_diffs.update(rd)
    
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        
        # Find the row with the maximum last element
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        max_val = rows[max_idx][-1]
        
        # Try to regenerate this row with a smaller max
        new_row = _generate_candidate_row(k, used_diffs, rng, max_val=max_val - 1)
        if new_row is None:
            continue
        
        new_row_diffs = _row_diffs(new_row)
        
        # Check conflicts with other rows
        conflict = False
        for j in range(n_rows):
            if j == max_idx:
                continue
            if new_row_diffs & row_diffs[j]:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Accept if scope improves
        new_scope = max(rows[j][-1] for j in range(n_rows) if j != max_idx)
        new_scope = max(new_scope, new_row[-1])
        
        if new_scope < best_scope:
            rows[max_idx] = new_row
            row_diffs[max_idx] = new_row_diffs
            
            # Update used_diffs
            used_diffs = set()
            for j in range(n_rows):
                used_diffs.update(row_diffs[j])
            
            best_scope = new_scope
            best_rows = [list(r) for r in rows]
    
    return best_rows
