import time
import random
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of (k+1) increasing integers starting at 0,
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 145  # safety margin before 150s kill
    best_rows = None
    best_max = float('inf')
    
    n_elements = k + 1
    n_diffs_per_row = n_elements * (n_elements - 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    def get_differences(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            row_diffs = get_differences(row)
            if len(row_diffs) != len(row):
                # shouldn't happen for strictly increasing
                pass
            if row_diffs & all_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    def get_max_val(rows):
        return max(max(row) for row in rows)
    
    def greedy_construct(used_diffs, rng, bias=1.0):
        """Construct one row given already-used differences."""
        row = [0]
        while len(row) < n_elements:
            # Try candidates from small to large with some randomization
            x = row[-1] + 1
            found = False
            max_x = 500  # upper bound to avoid infinite loop
            while x <= max_x:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                return None
        return row
    
    def randomized_greedy(rng, bias_factor=1.0):
        """Build all rows using randomized greedy."""
        used_diffs = set()
        rows = []
        for i in range(n_rows):
            # Add some randomness to the construction
            row = [0]
            while len(row) < n_elements:
                # Try a range of candidates with randomization
                candidates = []
                x = row[-1] + 1
                max_x = row[-1] + 50 + rng.randint(0, 50)
                while x <= max_x:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append((x, new_diffs))
                    x += 1
                
                if not candidates:
                    return None
                
                # Weighted random selection - prefer smaller x but with some randomness
                if bias_factor > 0.5:
                    # More deterministic - pick smallest
                    best = min(candidates, key=lambda c: c[0])
                    x, new_diffs = best
                else:
                    # Random selection with preference for smaller
                    weights = [1.0 / (c[0] + 1) ** bias_factor for c in candidates]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates, weights):
                        cum += w
                        if r <= cum:
                            chosen = c
                            break
                    x, new_diffs = chosen
                
                row.append(x)
                used_diffs |= new_diffs
            
            rows.append(row)
        
        return rows
    
    def local_improve(rows, rng, max_iter=100):
        """Try to reduce the maximum by local moves."""
        rows = [list(r) for r in rows]
        current_max = get_max_val(rows)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try to reduce the largest element in the row with the global max
            max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]
            
            # Try replacing the last element with a smaller one
            last_val = row[-1]
            for new_val in range(last_val - 1, row[-2], -1):
                test_row = row[:-1] + [new_val]
                # Check validity
                test_diffs = get_differences(test_row)
                all_diffs = set()
                valid = True
                for i, r in enumerate(rows):
                    if i == max_row_idx:
                        continue
                    r_diffs = get_differences(r)
                    if test_diffs & r_diffs:
                        valid = False
                        break
                    all_diffs |= r_diffs
                if valid:
                    rows[max_row_idx] = test_row
                    new_max = get_max_val(rows)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        break
            
            if not improved:
                # Try random perturbations
                row_idx = rng.randint(0, len(rows) - 1)
                row = rows[row_idx]
                if len(row) > 1:
                    # Try changing a non-first element
                    pos = rng.randint(1, len(row) - 1)
                    old_val = row[pos]
                    # Try nearby values
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_val = old_val + delta
                        if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                            continue
                        test_row = row[:]
                        test_row[pos] = new_val
                        # Check validity
                        test_diffs = get_differences(test_row)
                        all_diffs = set()
                        valid = True
                        for i, r in enumerate(rows):
                            if i == row_idx:
                                continue
                            r_diffs = get_differences(r)
                            if test_diffs & r_diffs:
                                valid = False
                                break
                            all_diffs |= r_diffs
                        if valid:
                            new_max = get_max_val([rows[i] if i != row_idx else test_row for i in range(len(rows))])
                            if new_max <= current_max:
                                rows[row_idx] = test_row
                                if new_max < current_max:
                                    current_max = new_max
                                improved = True
                                break
            
            if not improved:
                break
        
        return rows
    
    # Main search loop
    rng = random.Random(42)
    
    # First, try a pure greedy to get a baseline
    rows = randomized_greedy(random.Random(0), bias_factor=1.0)
    if rows is not None and is_valid(rows):
        best_rows = rows
        best_max = get_max_val(rows)
    
    # Now do randomized restarts
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        rng = random.Random(attempts)
        
        # Vary the bias factor
        bias = rng.uniform(0.1, 1.0)
        
        rows = randomized_greedy(rng, bias_factor=bias)
        if rows is None:
            continue
        
        if not is_valid(rows):
            continue
        
        cur_max = get_max_val(rows)
        
        if cur_max < best_max:
            best_rows = rows
            best_max = cur_max
        
        # Try local improvement
        if time.time() < deadline - 15:
            improved_rows = local_improve(rows, rng, max_iter=50)
            if improved_rows is not None:
                if is_valid(improved_rows):
                    imp_max = get_max_val(improved_rows)
                    if imp_max < best_max:
                        best_rows = improved_rows
                        best_max = imp_max
    
    # Final attempt: try to construct with a target max
    if time.time() < deadline - 5:
        # Binary search on the maximum value
        lo, hi = best_max - 1, best_max
        target = best_max - 1
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            # Try to construct with max <= target
            success = False
            for trial in range(10):
                rng = random.Random(trial * 1000 + target)
                used_diffs = set()
                rows = []
                valid = True
                for i in range(n_rows):
                    row = [0]
                    while len(row) < n_elements:
                        x = row[-1] + 1
                        found = False
                        while x <= target:
                            new_diffs = set()
                            ok = True
                            for a in row:
                                d = x - a
                                if d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                            if ok:
                                row.append(x)
                                used_diffs |= new_diffs
                                found = True
                                break
                            x += 1
                        if not found:
                            valid = False
                            break
                    if not valid:
                        break
                    rows.append(row)
                if valid and is_valid(rows):
                    cur_max = get_max_val(rows)
                    if cur_max < best_max:
                        best_rows = rows
                        best_max = cur_max
                    success = True
                    break
            if success:
                target = best_max - 1
            else:
                break
    
    if best_rows is None:
        # Fallback: simple greedy
        used_diffs = set()
        best_rows = []
        for i in range(n_rows):
            row = [0]
            x = 1
            while len(row) < n_elements:
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    row.append(x)
                    used_diffs |= new_diffs
                x += 1
            best_rows.append(row)
    
    return best_rows
