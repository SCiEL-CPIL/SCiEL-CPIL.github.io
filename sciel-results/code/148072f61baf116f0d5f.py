import random
import time
import math

def _row_diffs(row):
    """Compute the set of positive differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs

def _get_global_diffs(rows):
    """Compute the global set of all differences."""
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used

def _scope(rows):
    """Calculate the scope (maximum element in all rows)."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _validate(rows):
    """Check if the solution is valid."""
    used = set()
    for row in rows:
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        # Check differences
        diffs = _row_diffs(row)
        if len(diffs) != len(row)*(len(row)-1)//2:
            return False
        if diffs & used:
            return False
        used |= diffs
    return True

def _greedy_init(n_rows, k, rng):
    """
    Construct an initial valid set using a randomized greedy approach.
    We try to keep numbers small.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        # We need k more elements
        while len(row) < k + 1:
            # Try x, x+1, x+2... with some randomization to avoid dead ends
            # For a valid start, simple greedy often works for small n, k
            found = False
            limit = 1000 # Safety bound
            start_x = x
            # Add a bit of randomness to the starting x to diversify
            offset = rng.randint(0, 10)
            candidate = start_x + offset
            while candidate < start_x + limit:
                new_diffs = {candidate - a for a in row}
                if not (new_diffs & used):
                    row.append(candidate)
                    used |= new_diffs
                    x = candidate + 1
                    found = True
                    break
                candidate += 1
            if not found:
                # If stuck, restart this row with larger random jump
                row = [0]
                x = rng.randint(10, 50)
                # Re-add previous rows' used diffs? No, 'used' is global for all rows so far.
                # This simple restart might break validity if we don't track carefully.
                # For robustness, let's just do a deterministic greedy with a random seed shuffle?
                # Actually, for n=7, k=6, simple greedy usually finds a valid solution, just not optimal.
                # Let's try a more robust initialization:
                return _greedy_init_robust(n_rows, k, rng)
        rows.append(row)
    return rows

def _greedy_init_robust(n_rows, k, rng):
    """
    More robust greedy: tries multiple random seeds/orders if simple fails, 
    or uses a larger step.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        # Choose next elements one by one
        for _idx in range(k):
            found = False
            # Search for a valid next element
            # Start from last element + 1
            last = row[-1]
            # Try a range of values
            for delta in range(1, 100):
                candidate = last + delta
                # Add randomization to delta selection
                new_diffs = {candidate - a for a in row}
                if not (new_diffs & used):
                    row.append(candidate)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                # If stuck, this greedy failed. 
                # For this specific problem size, we can fallback to a known structure or random restart
                # Let's just force a larger jump
                candidate = last + rng.randint(20, 50)
                new_diffs = {candidate - a for a in row}
                if new_diffs & used:
                    # Still stuck? Add to used? No, that breaks validity.
                    # Let's try a simple backtracking or just accept a larger scope
                    # For initialization, we just need *a* valid solution.
                    # We can construct rows independently if we ensure disjoint diff sets.
                    # Since the space of diffs is small (max diff ~ 150), we can pick a random valid row.
                    pass
                row.append(candidate)
                used |= new_diffs
        rows.append(row)
    
    # Verify
    if not _validate(rows):
        # Fallback: very conservative greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            next_val = 1
            while len(row) < k + 1:
                while True:
                    new_diffs = {next_val - a for a in row}
                    if not (new_diffs & used):
                        row.append(next_val)
                        used |= new_diffs
                        break
                    next_val += 1
                next_val = row[-1] + 1
            rows.append(row)
    return rows

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Simulated Annealing to minimize the scope.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute diffs
    row_diffs = [_row_diffs(r) for r in rows]
    used = set()
    for s in row_diffs:
        used |= s
        
    temperature = best_scope * 0.5
    cooling = 0.9995
    min_temp = 0.1
    
    def try_shift(r, j):
        """
        Try to shift rows[r][j:] to the left by 1.
        Returns (success, new_scope_delta, old_diffs, new_diffs, old_suffix, new_suffix)
        """
        row = rows[r]
        if j <= 0 or j >= len(row):
            return False, 0, set(), set(), [], []
        if row[j] - 1 <= row[j-1]:
            return False, 0, set(), set(), [], []
            
        old_suffix = row[j:]
        new_suffix = [x - 1 for x in old_suffix]
        
        # Calculate old diffs
        # Diffs within suffix
        old_suffix_diffs = set()
        for i in range(len(old_suffix)):
            ai = old_suffix[i]
            for l in range(i+1, len(old_suffix)):
                old_suffix_diffs.add(old_suffix[l] - ai)
                
        # Diffs between prefix and suffix
        prefix = row[:j]
        cross_diffs = set()
        for a in prefix:
            for b in old_suffix:
                cross_diffs.add(b - a)
                
        old_diffs = old_suffix_diffs | cross_diffs
        
        # Calculate new diffs
        new_suffix_diffs = set()
        for i in range(len(new_suffix)):
            ai = new_suffix[i]
            for l in range(i+1, len(new_suffix)):
                new_suffix_diffs.add(new_suffix[l] - ai)
                
        new_cross_diffs = set()
        for a in prefix:
            for b in new_suffix:
                new_cross_diffs.add(b - a)
                
        new_diffs = new_suffix_diffs | new_cross_diffs
        
        # Check internal collisions in new_diffs
        # Expected number of diffs:
        L = len(new_suffix)
        expected_size = L*(L-1)//2 + j * L
        if len(new_diffs) < expected_size:
            return False, 0, set(), set(), [], []
            
        # Check collisions with other rows' diffs
        others = used - old_diffs
        if new_diffs & others:
            return False, 0, set(), set(), [], []
            
        return True, 0, old_diffs, new_diffs, old_suffix, new_suffix

    while time.time() < deadline and temperature > min_temp:
        r = rng.randrange(n_rows)
        j = rng.randrange(1, k + 1)
        
        old_scope = _scope(rows)
        
        # Save state
        saved_row = list(rows[r])
        saved_used = used.copy()
        saved_row_diffs_r = row_diffs[r].copy()
        
        success, _, old_diffs, new_diffs, old_suffix, new_suffix = try_shift(r, j)
        
        if success:
            # Apply change
            row = rows[r]
            row[j:] = new_suffix
            used = (used - old_diffs) | new_diffs
            row_diffs[r] = _row_diffs(row)
            
            new_scope = _scope(rows)
            delta = new_scope - old_scope
            
            if delta <= 0:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(x) for x in rows]
            else:
                if rng.random() < math.exp(-delta / temperature):
                    pass # Accept
                else:
                    # Revert
                    rows[r] = saved_row
                    used = saved_used
                    row_diffs[r] = saved_row_diffs_r
        temperature *= cooling

    return best_rows

def construct(n_rows, k):
    start = time.time()
    # Safety margin: evaluator kills at 150s. We should stop at 140s to be safe.
    deadline = start + 140.0 
    rng = random.Random(12345)
    
    # Initial solution
    rows = _greedy_init_robust(n_rows, k, rng)
    
    # Ensure validity
    if not _validate(rows):
        # Fallback to very simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            next_val = 1
            while len(row) < k + 1:
                while True:
                    new_diffs = {next_val - a for a in row}
                    if not (new_diffs & used):
                        row.append(next_val)
                        used |= new_diffs
                        break
                    next_val += 1
                next_val = row[-1] + 1
            rows.append(row)
            
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Local Search
    if time.time() < deadline - 5:
        improved = _local_search(rows, n_rows, k, deadline, rng)
        if _validate(improved) and _scope(improved) < best_scope:
            best = improved
            best_scope = _scope(improved)
            
    # Final validation
    if not _validate(best):
        # Last resort
        used = set()
        best = []
        for _ in range(n_rows):
            row = [0]
            next_val = 1
            while len(row) < k + 1:
                while True:
                    new_diffs = {next_val - a for a in row}
                    if not (new_diffs & used):
                        row.append(next_val)
                        used |= new_diffs
                        break
                    next_val += 1
                next_val = row[-1] + 1
            best.append(row)
        
    return best
