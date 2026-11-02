import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, rng=None):
    if rng is None:
        rng = random
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                if rng.random() < 0.1 and x < 200:
                    skip = rng.randint(1, 2)
                    x += skip
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        x -= skip
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _row_diffs_set(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    return max(max(row) for row in rows)

def _build_diff_count(rows, max_val):
    """Build a frequency array: diff_count[d] = number of times difference d appears across all rows."""
    n_rows = len(rows)
    k = len(rows[0]) - 1 if rows else 0
    total_diffs = n_rows * k * (k + 1) // 2
    # Max possible difference is max_val
    size = max_val + 1
    diff_count = [0] * size
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d < size:
                    diff_count[d] += 1
                else:
                    # Extend if needed (shouldn't happen normally)
                    diff_count.extend([0] * (d - size + 1))
                    size = len(diff_count)
    return diff_count

def _diffs_involving(row, pos):
    """Get set of differences that involve element at position pos."""
    n = len(row)
    diffs = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs.add(row[pos] - row[i])
        else:
            diffs.add(row[i] - row[pos])
    return diffs

def _is_valid_move(rows, r_idx, pos, new_val, diff_count, max_val):
    """Check if placing new_val at (r_idx, pos) is valid. Returns True/False."""
    row = rows[r_idx]
    old_val = row[pos]
    if new_val == old_val:
        return True
    
    # Check ordering
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < len(row) - 1 and new_val >= row[pos + 1]:
        return False
    
    # Compute old diffs involving this position
    old_diffs = _diffs_involving(row, pos)
    
    # Compute new diffs involving this position
    new_diffs = set()
    n = len(row)
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            new_diffs.add(new_val - row[i])
        else:
            new_diffs.add(row[i] - new_val)
    
    # For each new diff, check if it would conflict
    # A diff d conflicts if:
    #   - d is in new_diffs more than once (internal conflict)
    #   - d is used by other rows (diff_count[d] > count of d in old_diffs from this row)
    
    # Check internal uniqueness of new_diffs
    if len(new_diffs) != len(new_diffs):
        return False
    # Actually check for duplicates properly
    seen = set()
    for d in new_diffs:
        if d in seen:
            return False
        seen.add(d)
    
    # Check against global diff_count
    # For each new diff d:
    #   - If d was in old_diffs, it's "freed up" from this row
    #   - If d is not in old_diffs, it's "new" and needs diff_count[d] == 0
    #   - If d IS in old_diffs, it's being reused, so we need to make sure no OTHER row has it
    #     which means diff_count[d] - (1 if d in old_diffs else 0) == 0
    
    for d in new_diffs:
        if d > max_val:
            return False
        # How many times does d appear in old_diffs?
        # Since old_diffs is a set, it appears at most once
        was_in_old = 1 if d in old_diffs else 0
        # Available count = total count minus what this row currently contributes
        available = diff_count[d] - was_in_old
        if available > 0:
            return False
    
    return True

def _apply_move(rows, r_idx, pos, new_val, diff_count, max_val):
    """Apply a move and update diff_count. Returns old_diffs for potential rollback."""
    row = rows[r_idx]
    old_val = row[pos]
    old_diffs = _diffs_involving(row, pos)
    
    # Remove old diffs
    for d in old_diffs:
        if d <= max_val:
            diff_count[d] -= 1
    
    # Set new value
    row[pos] = new_val
    
    # Add new diffs
    new_diffs = _diffs_involving(row, pos)
    for d in new_diffs:
        if d <= max_val:
            diff_count[d] += 1
    
    return old_diffs

def _rollback(rows, r_idx, pos, old_val, diff_count, max_val):
    """Rollback a move."""
    row = rows[r_idx]
    current_val = row[pos]
    current_diffs = _diffs_involving(row, pos)
    
    for d in current_diffs:
        if d <= max_val:
            diff_count[d] -= 1
    
    row[pos] = old_val
    
    old_diffs = _diffs_involving(row, pos)
    for d in old_diffs:
        if d <= max_val:
            diff_count[d] += 1

def _local_search(rows, n_rows, k, deadline, rng, T_init=10.0, T_min=0.5, cool=0.99995, max_no_improve=20000):
    """Element-level local search with simulated annealing using diff_count for speed."""
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    max_val = current_val + 30  # Allow some room for increases
    diff_count = _build_diff_count(current_rows, max_val)
    row_maxes = [max(r) for r in current_rows]
    
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    T = T_init
    no_improve = 0
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Bias toward high-value positions in high-max rows
        if rng.random() < 0.5:
            max_rows = [i for i in range(n_rows) if row_maxes[i] >= current_val - 3]
            ri = rng.choice(max_rows) if max_rows else rng.randrange(n_rows)
        else:
            ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates: mostly decreases, some increases
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        if rng.random() < 0.1:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 15
            if low <= high:
                candidates.append(rng.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        rng.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            if not _is_valid_move(current_rows, ri, pi, new_val, diff_count, max_val):
                continue
            
            # Compute delta
            old_row_max = row_maxes[ri]
            cand_row_max = max(current_rows[ri][:pi] + [new_val] + current_rows[ri][pi+1:])
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.15:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                _apply_move(current_rows, ri, pi, new_val, diff_count, max_val)
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cool)
        
        if no_improve > max_no_improve // 2:
            # Restart from best
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            max_val = current_val + 30
            diff_count = _build_diff_count(current_rows, max_val)
            row_maxes = [max(r) for r in current_rows]
            T = T_init
            no_improve = 0
    
    return best_rows, best_val

def _greedy_descent(rows, n_rows, k, deadline, rng):
    """Aggressive greedy descent: repeatedly try to pull the max element down."""
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    max_val = current_val + 10
    diff_count = _build_diff_count(current_rows, max_val)
    row_maxes = [max(r) for r in current_rows]
    
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Find the row(s) with the global max
        max_val_cur = current_val
        max_rows = [i for i in range(n_rows) if row_maxes[i] == max_val_cur]
        
        if not max_rows:
            break
        
        ri = rng.choice(max_rows)
        # Focus on the last element (the max) or nearby
        if rng.random() < 0.7:
            pi = k  # The last element
        else:
            pi = rng.randrange(max(1, k - 2), k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Try to decrease
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(25, old_val - min_allowed)
        
        candidates = list(range(1, max_decrease + 1))
        rng.shuffle(candidates)
        
        moved = False
        for d in candidates:
            new_val = old_val - d
            if new_val <= min_allowed:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            if not _is_valid_move(current_rows, ri, pi, new_val, diff_count, max_val):
                continue
            
            # Accept any valid decrease
            _apply_move(current_rows, ri, pi, new_val, diff_count, max_val)
            row_maxes[ri] = max(current_rows[ri])
            current_val = max(row_maxes)
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
            
            no_improve = 0
            moved = True
            break
        
        if not moved:
            no_improve += 1
    
    return best_rows, best_val

def _perturb(rows, n_rows, k, rng, n_moves=3):
    """Perform random valid moves to perturb the solution for ILS."""
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    max_val = current_val + 20
    diff_count = _build_diff_count(current_rows, max_val)
    
    for _ in range(n_moves):
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 15
        
        # Try a random range of values
        range_size = max(1, min(20, max_allowed - min_allowed))
        for _attempt in range(10):
            new_val = rng.randint(min_allowed, min_allowed + range_size)
            if new_val == old_val:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            if _is_valid_move(current_rows, ri, pi, new_val, diff_count, max_val):
                _apply_move(current_rows, ri, pi, new_val, diff_count, max_val)
                break
    
    return current_rows

def _construct_single_start(n_rows, k, rng, deadline):
    """Single optimization run with ILS structure."""
    rows = _greedy_init(n_rows, k, rng)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Phase 1: Local search
    t1_deadline = min(deadline, time.time() + (deadline - time.time()) * 0.4)
    rows, val = _local_search(rows, n_rows, k, t1_deadline, rng, T_init=15.0, T_min=0.5, cool=0.99995, max_no_improve=15000)
    if val < best_val:
        best_val = val
        best_rows = rows
    
    # Phase 2: ILS with perturbation
    t2_deadline = min(deadline, time.time() + (deadline - time.time()) * 0.45)
    while time.time() < t2_deadline:
        # Perturb
        perturbed = _perturb(best_rows, n_rows, k, rng, n_moves=rng.randint(2, 5))
        
        # Local search on perturbed
        remaining = t2_deadline - time.time()
        if remaining < 2.0:
            break
        ls_deadline = min(t2_deadline, time.time() + remaining * 0.8)
        rows, val = _local_search(perturbed, n_rows, k, ls_deadline, rng, T_init=8.0, T_min=0.3, cool=0.99997, max_no_improve=8000)
        
        if val < best_val:
            best_val = val
            best_rows = rows
    
    # Phase 3: Greedy descent polish
    if time.time() < deadline - 1.0:
        gd_deadline = deadline
        rows, val = _greedy_descent(best_rows, n_rows, k, gd_deadline, rng)
        if val < best_val:
            best_val = val
            best_rows = rows
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s)
    
    # Multi-start
    num_starts = 4
    time_per_start = (deadline - start_time) / num_starts
    
    best_rows = None
    best_val = float('inf')
    
    for s in range(num_starts):
        start_deadline = start_time + (s + 1) * time_per_start
        if s < num_starts - 1:
            start_deadline = min(start_deadline, deadline - 10.0)
        
        seed = s * 7919 + 13
        rng = random.Random(seed)
        
        rows, val = _construct_single_start(n_rows, k, rng, start_deadline)
        
        if val < best_val:
            best_val = val
            best_rows = rows
    
    # Final polish
    if best_rows is not None and time.time() < deadline - 2.0:
        rng = random.Random(99999)
        polished_rows, polished_val = _greedy_descent(best_rows, n_rows, k, deadline - 1.0, rng)
        if polished_val < best_val:
            best_val = polished_val
            best_rows = polished_rows
    
    if best_rows is None:
        best_rows = _greedy_init(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
