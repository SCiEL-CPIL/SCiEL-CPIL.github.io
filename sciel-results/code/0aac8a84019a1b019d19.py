import time
import random
import math

_TIME_LIMIT = 145.0
_START_TIME = time.time()

def _time_left():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = 0
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if used & (1 << d):
                    return False
                used |= (1 << d)
    return True

def _scope(rows):
    return max(max(r) for r in rows) if rows else 0

def _row_diff_mask(row):
    """Compute bitmask of all positive differences in a row."""
    mask = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _greedy_row(used_mask, k, rng, strategy="shuffled"):
    """
    Build one row greedily.
    used_mask: bitmask of differences used by other rows.
    strategy: 'pure', 'shuffled', 'lookahead'
    """
    row = [0]
    x = 0
    local_mask = 0
    
    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new_diffs = 0
                valid = True
                for a in row:
                    d = x - a
                    if local_mask & (1 << d) or used_mask & (1 << d):
                        valid = False
                        break
                    new_diffs |= (1 << d)
                if valid:
                    break
                x += 1
            row.append(x)
            local_mask |= new_diffs
        elif strategy == "shuffled":
            base = x + 1
            window = 80
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new_diffs = 0
                valid = True
                for a in row:
                    d = cand - a
                    if local_mask & (1 << d) or used_mask & (1 << d):
                        valid = False
                        break
                    new_diffs |= (1 << d)
                if valid:
                    row.append(cand)
                    local_mask |= new_diffs
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = cand - a
                        if local_mask & (1 << d) or used_mask & (1 << d):
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        row.append(cand)
                        local_mask |= new_diffs
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 60):
                new_diffs = 0
                valid = True
                for a in row:
                    d = cand - a
                    if local_mask & (1 << d) or used_mask & (1 << d):
                        valid = False
                        break
                    new_diffs |= (1 << d)
                if valid:
                    valid_cands.append((cand, new_diffs))
                if len(valid_cands) >= 15:
                    break
            
            if not valid_cands:
                cand = base
                while True:
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = cand - a
                        if local_mask & (1 << d) or used_mask & (1 << d):
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        valid_cands.append((cand, new_diffs))
                        break
                    cand += 1
            
            # Score candidates: prefer smaller cand, and fewer "blocked" next steps?
            # Simple heuristic: smallest cand is usually best for tight packing
            best_cand, best_mask = valid_cands[0]
            # We could try to look ahead, but pure smallest is often strong for DTS
            row.append(best_cand)
            local_mask |= best_mask
            x = best_cand
            
    return row, local_mask

def _construct_set(n_rows, k, rng, strategy):
    """Construct a full set of rows."""
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row, mask = _greedy_row(used_mask, k, rng, strategy)
        rows.append(row)
        used_mask |= mask
    return rows

def _local_search_reduce(rows, n_rows, k, rng, max_iters=50000):
    """
    Local search specifically aimed at reducing the maximum element.
    """
    rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    row_masks = [_row_diff_mask(r) for r in rows]
    
    for iters in range(max_iters):
        if _time_left() < 2.0:
            break
            
        # Find row with max element
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        if current_max <= best_scope:
            target = best_scope - 1
        else:
            target = best_scope - 1
            
        if target < k + 1:
            break
            
        # Calculate mask of all other rows
        other_mask = 0
        for i in range(n_rows):
            if i != max_idx:
                other_mask |= row_masks[i]
                
        row = rows[max_idx]
        improved = False
        
        # Try to reduce the last element
        # We can try values from current_max - 1 down to row[-2] + 1
        # But we must ensure strictly increasing and unique diffs
        start_val = min(current_max - 1, target)
        end_val = row[-2] + 1
        
        # Randomize order to escape local minima? No, greedy reduction is good first.
        # Let's try a few random candidates in the range if greedy fails?
        # For now, linear scan downwards is fast.
        for new_val in range(start_val, end_val - 1, -1):
            if new_val <= row[-2]:
                break
            new_row = row[:-1] + [new_val]
            # Check internal validity
            new_mask = 0
            valid = True
            n = len(new_row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = new_row[j] - new_row[i]
                    if new_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
            if not valid:
                continue
            # Check against other rows
            if new_mask & other_mask:
                continue
                
            # Valid move
            rows[max_idx] = new_row
            row_masks[max_idx] = new_mask
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break
            
        if not improved:
            # Try to move other elements in the max row to free up space
            # Pick a random position (not 0, not last)
            if len(row) > 2:
                pos = rng.randint(1, len(row) - 2)
                old_val = row[pos]
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1
                
                # Try a few random values in [lower, upper]
                tried = 0
                while tried < 10:
                    new_val = rng.randint(lower, upper)
                    if new_val == old_val:
                        tried += 1
                        continue
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    new_mask = 0
                    valid = True
                    n = len(new_row)
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = new_row[j] - new_row[i]
                            if new_mask & (1 << d):
                                valid = False
                                break
                            new_mask |= (1 << d)
                    if not valid:
                        tried += 1
                        continue
                    if new_mask & other_mask:
                        tried += 1
                        continue
                        
                    # Valid move. Does it help reduce max?
                    # Only accept if it reduces the max of this row or allows last element to be smaller later.
                    # For now, accept if it reduces the current row's max (which is row[-1], unchanged here) 
                    # OR if it reduces the "potential" (e.g. smaller gaps).
                    # Let's just accept if it doesn't increase scope.
                    new_scope_row = max(new_row)
                    if new_scope_row <= current_max:
                        rows[max_idx] = new_row
                        row_masks[max_idx] = new_mask
                        improved = True
                        break
                    tried += 1
                    
    return rows, best_scope

def _simulated_annealing(rows, n_rows, k, rng, max_iters=20000):
    """
    Simulated annealing to find better configurations.
    """
    rows = [r[:] for r in rows]
    best_rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    current_scope = best_scope
    
    row_masks = [_row_diff_mask(r) for r in rows]
    
    start_temp = 50.0
    end_temp = 0.1
    cooling = 0.999
    
    temp = start_temp
    
    for iters in range(max_iters):
        if _time_left() < 2.0:
            break
            
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        if len(row) <= 1:
            continue
            
        pos = rng.randint(1, len(row) - 1)
        current_val = row[pos]
        
        lower = row[pos - 1] + 1
        upper = row[pos + 1] - 1 if pos < len(row) - 1 else current_scope + 10
        
        if upper < lower:
            continue
            
        new_val = rng.randint(lower, upper)
        if new_val == current_val:
            continue
            
        new_row = row[:pos] + [new_val] + row[pos + 1:]
        
        # Check strictly increasing
        if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
            continue
            
        new_mask = 0
        valid = True
        n = len(new_row)
        for i in range(n):
            for j in range(i + 1, n):
                d = new_row[j] - new_row[i]
                if new_mask & (1 << d):
                    valid = False
                    break
                new_mask |= (1 << d)
        if not valid:
            continue
            
        # Check against other rows
        other_mask = 0
        for i in range(n_rows):
            if i != r_idx:
                other_mask |= row_masks[i]
                
        if new_mask & other_mask:
            continue
            
        # Valid move
        new_scope = max(max(new_row), max(max(rows[i]) for i in range(n_rows) if i != r_idx))
        
        if new_scope < current_scope:
            rows[r_idx] = new_row
            row_masks[r_idx] = new_mask
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            delta = new_scope - current_scope
            if temp > 0 and rng.random() < math.exp(-delta / temp):
                rows[r_idx] = new_row
                row_masks[r_idx] = new_mask
                current_scope = new_scope
                
        temp = max(end_temp, temp * cooling)
        
    return best_rows, best_scope

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    strategies = ["pure", "shuffled", "lookahead"]
    
    # Phase 1: Diverse constructions
    for strategy in strategies:
        if _time_left() < 10.0:
            break
        rows = _construct_set(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            
    # More randomized restarts
    attempts = 0
    while _time_left() > 30.0 and attempts < 50:
        attempts += 1
        strategy = rng.choice(strategies)
        rows = _construct_set(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 112:
                break
                
    # Phase 2: Local Search to reduce scope
    if _time_left() > 20.0 and best_rows:
        rows_ls, scope_ls = _local_search_reduce(best_rows, n_rows, k, rng, max_iters=10000)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls
            
    # Phase 3: Simulated Annealing
    if _time_left() > 20.0 and best_rows:
        rows_sa, scope_sa = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=20000)
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa
            
    # Phase 4: Another Local Search pass
    if _time_left() > 10.0 and best_rows:
        rows_ls2, scope_ls2 = _local_search_reduce(best_rows, n_rows, k, rng, max_iters=5000)
        if scope_ls2 < best_scope:
            best_scope = scope_ls2
            best_rows = rows_ls2

    # Final validation
    if best_rows is None or not _validate(best_rows):
        # Fallback: pure greedy
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row, mask = _greedy_row(used_mask, k, None, "pure")
            rows.append(row)
            used_mask |= mask
        return rows

    return best_rows
