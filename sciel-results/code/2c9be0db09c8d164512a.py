import time
import random
import sys

_DEADLINE = 115.0 # Safety margin

def _validate(rows):
    """Check that all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _score(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)

def _solve_dts_4_7(max_val):
    """
    Backtracking solver for DTS(4,7) with max entry <= max_val.
    Returns a list of 4 rows if found, else None.
    """
    n_rows = 4
    k = 7 # 8 points total (0 to k)
    points_per_row = k + 1
    
    # State
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    # Heuristic: Order of rows. We can try different orders.
    # For now, fixed order 0, 1, 2, 3.
    
    def backtrack(r, c):
        if time.time() > _DEADLINE:
            return False
            
        if r == n_rows:
            return True
        
        if c == points_per_row:
            return backtrack(r + 1, 1) # Next row, start from index 1 (0 is fixed)
            
        if c == 0:
            # 0 is already placed
            return backtrack(r, c + 1)
            
        prev_val = rows[r][c - 1]
        
        # Determine range for v
        # v must be > prev_val
        # v must be <= max_val
        # Also, we need (points_per_row - 1 - c) more values after v.
        # The minimum next value is v + 1.
        # The minimum last value is v + (points_per_row - 1 - c).
        # So v + (points_per_row - 1 - c) <= max_val
        # => v <= max_val - (points_per_row - 1 - c)
        
        max_v = max_val - (points_per_row - 1 - c)
        
        # Optimization: If we are in the last row, we can be more aggressive?
        # No, just stick to the bound.
        
        # Randomization: We can shuffle the candidate values?
        # For determinism in finding a solution, we usually go ascending.
        # But to escape local minima (if we were doing local search), we'd randomize.
        # For backtracking, ascending is usually best for finding *a* solution quickly if one exists with small values.
        
        # However, sometimes "large" values early on allow "small" differences later?
        # Actually, differences are v - prev.
        # If we pick small v, differences are small.
        # If we pick large v, differences are large.
        # We want to avoid using small differences if they are "expensive"?
        # No, all differences must be distinct.
        
        # Let's try candidates from prev_val + 1 to max_v.
        # To add some randomness and robustness, we can occasionally pick from the middle?
        # No, pure backtracking with good pruning should work.
        
        for v in range(prev_val + 1, max_v + 1):
            # Check differences
            valid = True
            new_diffs = []
            for i in range(c):
                d = v - rows[r][i]
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # Place
            rows[r].append(v)
            used.update(new_diffs)
            
            if backtrack(r, c + 1):
                return True
                
            # Undo
            rows[r].pop()
            used.difference_update(new_diffs)
            
        return False

    # Start backtracking
    # We start at row 0, col 1 (since col 0 is 0)
    # But wait, the `backtrack` function handles c=0 by moving to c=1.
    # So we call backtrack(0, 0).
    
    if backtrack(0, 0):
        return [row[:] for row in rows]
    return None

def _randomized_solve_dts_4_7(max_val, num_restarts=100):
    """
    Try multiple randomized restarts.
    We can randomize the order of value selection.
    """
    n_rows = 4
    points_per_row = 8
    
    # We will modify the backtrack to accept a random seed and use it to shuffle candidates.
    # But shuffling candidates in backtracking can be expensive if we do it naively.
    # Instead, we can just try different "biases" or just run the deterministic solver.
    # The deterministic solver might get stuck in a bad branch.
    # Randomized restarts with *different* initial row orders or value biases help.
    
    # Let's try a different strategy:
    # 1. Fix row 0 to a specific set of candidates?
    # 2. Or just run the deterministic solver. It's often fast enough for 134.
    
    # If deterministic fails, we can try to randomize the value order.
    
    for i in range(num_restarts):
        if time.time() > _DEADLINE:
            break
            
        # Create a local copy of the solver logic with randomization
        rows = [[0] for _ in range(n_rows)]
        used = set()
        
        def backtrack(r, c, rng):
            if time.time() > _DEADLINE:
                return False
            if r == n_rows:
                return True
            if c == points_per_row:
                return backtrack(r + 1, 1, rng)
            if c == 0:
                return backtrack(r, c + 1, rng)
                
            prev_val = rows[r][c - 1]
            max_v = max_val - (points_per_row - 1 - c)
            
            # Generate candidates
            candidates = list(range(prev_val + 1, max_v + 1))
            
            # Randomize order slightly?
            # If we shuffle, we might find a solution faster if the "good" value is not at the start.
            # But for small max_val, the range is small.
            # Let's try a "biased" randomization: 50% chance to sort ascending, 50% descending?
            # Or just shuffle.
            rng.shuffle(candidates)
            
            for v in candidates:
                valid = True
                new_diffs = []
                for i in range(c):
                    d = v - rows[r][i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                rows[r].append(v)
                used.update(new_diffs)
                
                if backtrack(r, c + 1, rng):
                    return True
                    
                rows[r].pop()
                used.difference_update(new_diffs)
                
            return False
        
        rng = random.Random(i)
        if backtrack(0, 0, rng):
            return [row[:] for row in rows]
            
    return None

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    with all positive differences distinct across all rows.
    Minimize the largest entry.
    """
    t0 = time.time()
    
    if n_rows != 4 or k != 7:
        # Fallback for other cases (not expected in this task)
        # Use greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows

    # Specific case DTS(4,7)
    # Target: 134
    # Try to find a solution with max_val = 134
    
    # 1. Try deterministic backtracking
    rows = _solve_dts_4_7(134)
    if rows:
        if _validate(rows):
            return rows
    
    # 2. Try randomized backtracking
    rows = _randomized_solve_dts_4_7(134, num_restarts=50)
    if rows:
        if _validate(rows):
            return rows
    
    # 3. If 134 fails, try 135, 136... up to 140
    for mv in range(135, 141):
        if time.time() > _DEADLINE:
            break
        rows = _solve_dts_4_7(mv)
        if rows:
            if _validate(rows):
                return rows
        
    # 4. Fallback: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
            x += 1
        rows.append(row)
    
    return rows
