import time
import random
import math
from collections import Counter

_DEADLINE = None
_BEST_SCOPE = float('inf')
_BEST_ROWS = None

def _fast_greedy(k, n_rows):
    """Generate a valid baseline using greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _compute_diffs(rows, k):
    """Compute the set of all differences and their counts."""
    diff_counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                diff_counts[d] += 1
    return diff_counts

def _get_scope(rows):
    return max(max(r) for r in rows)

def construct(n_rows, k):
    global _DEADLINE, _BEST_SCOPE, _BEST_ROWS
    
    _DEADLINE = time.time() + 115.0
    _BEST_SCOPE = float('inf')
    _BEST_ROWS = None
    
    # 1. Generate a valid starting point using greedy
    current_rows = _fast_greedy(k, n_rows)
    current_scope = _get_scope(current_rows)
    
    # Update best
    _BEST_SCOPE = current_scope
    _BEST_ROWS = [list(r) for r in current_rows]
    
    # Prepare data structures for fast updates
    # diff_count[d] = number of times difference d appears
    diff_count = _compute_diffs(current_rows, k)
    
    # Precompute for each row, the list of (i, j) pairs? 
    # It's better to iterate and compute on the fly for the specific element being moved.
    
    # Simulated Annealing Parameters
    # We want to minimize scope. 
    # Cost function: current_scope. 
    # However, we need to ensure validity. 
    # Strategy: 
    #   - Randomly pick a row i and an index j (1 <= j < k+1).
    #   - Try to move element r[i][j] to a new value x.
    #   - The new value must maintain increasing order in the row.
    #   - The new differences must not conflict with existing differences (count must become 1, not >1).
    #   - Actually, the constraint is "all differences ... are distinct over all rows".
    #   - This means every difference d must appear exactly once in the entire set of rows.
    #   - Wait, the problem says "all differences between two entries of the same row are distinct over all rows".
    #   - This implies the set of all differences from all rows must be a set of unique integers.
    #   - So, if we move an element, we remove its old differences and add new ones.
    #   - The new differences must not already exist in the global set (count 0), and the old differences must be removed.
    
    # Let's refine the validity check for a move:
    # Remove element x_old from row i at index j.
    #   - Remove diffs (x_old - row[i][t]) for t < j
    #   - Remove diffs (row[i][t] - x_old) for t > j
    #   - These diffs must have count 1 in diff_count (which they do, since valid).
    #   - Decrement their counts.
    # Add element x_new at row i at index j.
    #   - Must be > row[i][j-1] and < row[i][j+1] (if exists).
    #   - Add diffs (x_new - row[i][t]) for t < j
    #   - Add diffs (row[i][t] - x_new) for t > j
    #   - These diffs must have count 0 in diff_count (after removal).
    #   - Increment their counts.
    
    # If the move is valid, the new scope is max(current_scope, x_new) but potentially lower if x_old was the max.
    # Actually, scope is max of all entries.
    
    # Let's implement the SA loop.
    
    rng = random.Random(42)
    
    # Temperature schedule
    # Start high enough to accept moves that don't improve scope immediately?
    # Actually, we only accept if the move is valid. 
    # The "energy" is the scope. 
    # If new_scope < current_scope, accept.
    # If new_scope == current_scope, accept with some probability? Or always? 
    # If new_scope > current_scope, we might accept to escape local minima, but scope is a hard upper bound we want to minimize.
    # Usually in these problems, we just look for valid configurations with lower scope.
    # Since validity is a hard constraint, we only consider valid moves.
    # Among valid moves, we want to minimize scope.
    # So, if new_scope < current_scope, accept.
    # If new_scope == current_scope, accept with probability 0.5 (to shuffle).
    # If new_scope > current_scope, accept with probability exp(-delta/T).
    
    # However, finding a valid move with lower scope is hard.
    # A better heuristic: 
    #   Focus on the largest element in the largest row.
    #   Try to reduce it.
    
    # Let's try a targeted search:
    #   1. Identify the row with the maximum element.
    #   2. Try to reduce the last element of that row.
    #   3. If that fails, try other elements.
    
    # But SA is more robust. Let's stick to SA but with a focus on scope reduction.
    
    current_rows = [list(r) for r in current_rows]
    current_scope = _get_scope(current_rows)
    _BEST_SCOPE = current_scope
    _BEST_ROWS = [list(r) for r in current_rows]
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < _DEADLINE:
        if T < 0.01:
            break
            
        # Pick a random row and index
        i = rng.randrange(n_rows)
        # We can move any element except the first (0)
        j = rng.randrange(1, k + 1)
        
        x_old = current_rows[i][j]
        
        # Determine range for x_new
        lo = current_rows[i][j-1] + 1
        hi = current_rows[i][j+1] - 1 if j < k else current_scope # Upper bound is current scope? 
        # Actually, x_new can be anything as long as it maintains order.
        # But to reduce scope, we should try values <= current_scope.
        # If x_old is the global max, we must try x_new < x_old.
        
        # Let's try a random value in [lo, hi]
        # If hi < lo, skip
        if hi < lo:
            T *= cooling
            continue
            
        # To be efficient, if x_old is the max, bias towards lower values
        if x_old == current_scope:
            # Try values in [lo, x_old - 1]
            if x_old - 1 < lo:
                T *= cooling
                continue
            x_new = rng.randrange(lo, x_old)
        else:
            # Try values in [lo, hi]
            # Maybe bias towards current values to keep changes small?
            # Or just random.
            x_new = rng.randrange(lo, hi + 1)
            
        if x_new == x_old:
            continue
            
        # Check validity of move
        # 1. Remove old diffs
        valid = True
        row = current_rows[i]
        old_diffs = []
        for t in range(j):
            d = x_old - row[t]
            if diff_count[d] != 1:
                valid = False
                break
            old_diffs.append(d)
        if valid:
            for t in range(j + 1, k + 1):
                d = row[t] - x_old
                if diff_count[d] != 1:
                    valid = False
                    break
                old_diffs.append(d)
                
        if not valid:
            T *= cooling
            continue
            
        # Decrement counts
        for d in old_diffs:
            diff_count[d] -= 1
            
        # 2. Check new diffs
        new_diffs = []
        for t in range(j):
            d = x_new - row[t]
            if diff_count[d] != 0:
                valid = False
                break
            new_diffs.append(d)
        if valid:
            for t in range(j + 1, k + 1):
                d = row[t] - x_new
                if diff_count[d] != 0:
                    valid = False
                    break
                new_diffs.append(d)
                
        if not valid:
            # Rollback
            for d in old_diffs:
                diff_count[d] += 1
            T *= cooling
            continue
            
        # 3. Apply move
        for d in new_diffs:
            diff_count[d] += 1
        current_rows[i][j] = x_new
        
        new_scope = max(current_scope, x_new)
        # If x_old was the unique max, new_scope might be lower
        if x_old == current_scope:
            # Recalculate scope? 
            # It's expensive. 
            # But we can check if any other element is == current_scope.
            # If not, new_scope is max of remaining.
            # For simplicity, let's just take max(current_rows) if x_old was max.
            # Or better, maintain a count of how many times each value appears?
            # Or just calculate max if x_old == current_scope.
            if any(v == current_scope for r in current_rows for v in r if v != x_new):
                new_scope = current_scope
            else:
                new_scope = max(max(r) for r in current_rows)
        else:
            new_scope = current_scope # x_new <= x_old < current_scope, so scope doesn't increase
            
        # Acceptance
        delta = new_scope - current_scope
        if delta < 0 or (delta == 0 and rng.random() < 0.5) or (delta > 0 and rng.random() < math.exp(-delta / T)):
            current_scope = new_scope
            if current_scope < _BEST_SCOPE:
                _BEST_SCOPE = current_scope
                _BEST_ROWS = [list(r) for r in current_rows]
        else:
            # Rollback
            for d in new_diffs:
                diff_count[d] -= 1
            for d in old_diffs:
                diff_count[d] += 1
            current_rows[i][j] = x_old
            
        T *= cooling
        
    return _BEST_ROWS
