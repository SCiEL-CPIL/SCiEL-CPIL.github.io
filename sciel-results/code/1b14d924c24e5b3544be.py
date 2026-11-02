import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
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

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _build_static_diffs(row, k):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                s.add(row[j] - row[i])
        static[pi] = s
    return static

def _compute_new_diffs_fast(row, pi, new_val, static_diffs_pi):
    """
    Given a row with position pi replaced by new_val, compute the full diff set
    using the precomputed static diffs plus the k diffs involving pi.
    Returns (new_diffs_set, valid_internal).
    """
    n = len(row)
    new_diffs = set(static_diffs_pi)
    valid = True
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)
    return new_diffs, valid

def _check_conflict_fast(new_val, row, pi, static_diffs_pi, old_diffs, all_used):
    """
    Fast conflict check: only check the k new differences (involving pi)
    against (all_used - old_diffs). Also check internal validity.
    Returns (valid, new_diffs_set) if valid, else (False, None).
    """
    n = len(row)
    static_set = static_diffs_pi
    # The new diffs are: static_set union {differences involving pi}
    # We need to check:
    # 1. No internal collision: new diffs don't collide with each other or with static_set
    # 2. No external collision: new diffs not in (all_used - old_diffs)
    
    # Build set of new diffs involving pi
    pi_diffs = set()
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in pi_diffs:
            return False, None
        if d in static_set:
            return False, None
        pi_diffs.add(d)
    
    # Check external conflict: pi_diffs should not intersect (all_used - old_diffs)
    for d in pi_diffs:
        if d in all_used and d not in old_diffs:
            return False, None
    
    # Build full new_diffs set
    new_diffs = set(static_set)
    new_diffs |= pi_diffs
    return True, new_diffs

def _random_valid_row(k, used_diffs, max_val):
    """
    Try to build a random valid row of k+1 entries starting at 0,
    with all entries <= max_val, and no differences in used_diffs.
    Returns None if no valid row found.
    """
    for attempt in range(50):
        row = [0]
        valid = True
        for _ in range(k):
            low = row[-1] + 1
            high = max_val
            if low > high:
                valid = False
                break
            x = random.randint(low, high)
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                valid = False
                break
            row.append(x)
        if valid:
            return row
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing with fast diff computation
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    
    # Precompute row diffs and static diffs
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    # Phase 1: Main SA loop
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates - try more of these
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Increasing candidates (for exploration) - fewer
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        # Random jump (large perturbation)
        if random.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        # Pre-fetch static diffs for this position
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            # Basic validity check for this position
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast conflict check
            valid_internal, new_diffs = _check_conflict_fast(
                new_val, current_rows[ri], pi, static_pi, old_diffs, all_used)
            
            if not valid_internal:
                continue
                
            # Move is valid - compute new global max efficiently
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            # Update row_maxes temporarily
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max  # restore
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                # Rebuild static diffs for this row
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
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
        
        # Periodic restart if stuck
        if no_improve > 12000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # Phase 2: Random row replacement to escape local minima
    polish_deadline = min(deadline, start_time + 145.0)
    T = 1.0
    T_min = 0.1
    cool = 0.99999
    no_improve = 0
    max_no_improve_polish = 10000
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Occasionally try a full row replacement
        if random.random() < 0.1:
            ri = random.randrange(n_rows)
            other_used = set()
            for j in range(n_rows):
                if j != ri:
                    other_used |= row_diffs[j]
            
            new_row = _random_valid_row(k, other_used, best_val)
            if new_row is not None:
                new_row_max = max(new_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = new_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                if delta < 0 or (delta == 0 and random.random() < 0.1):
                    all_used -= row_diffs[ri]
                    new_diffs = _row_diffs_set(new_row)
                    all_used |= new_diffs
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    static_diffs[ri] = _build_static_diffs(new_row, k)
                    row_maxes[ri] = new_row_max
                    current_val = max(row_maxes)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    continue
        
        # Standard position-based moves
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            valid_internal, new_diffs = _check_conflict_fast(
                new_val, current_rows[ri], pi, static_pi, old_diffs, all_used)
            
            if not valid_internal:
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.1:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.05))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
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
    
    # Phase 3: Final deterministic polish - try all possible decreases exhaustively
    final_deadline = start_time + 147.0
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    improved = True
    while improved and time.time() < final_deadline:
        improved = False
        for ri in range(n_rows):
            if time.time() >= final_deadline:
                break
            for pi in range(1, k + 1):
                if time.time() >= final_deadline:
                    break
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                # Try all values from min_allowed up to old_val - 1 (decreasing)
                for new_val in range(min_allowed, old_val):
                    static_pi = static_diffs[ri][pi]
                    old_diffs = row_diffs[ri]
                    valid_internal, new_diffs = _check_conflict_fast(
                        new_val, current_rows[ri], pi, static_pi, old_diffs, all_used)
                    if not valid_internal:
                        continue
                    
                    # Check if this reduces global max
                    if pi == k:
                        cand_row_max = new_val
                    else:
                        cand_row_max = current_rows[ri][k]
                    
                    old_row_max = row_maxes[ri]
                    row_maxes[ri] = cand_row_max
                    new_global_max = max(row_maxes)
                    row_maxes[ri] = old_row_max
                    
                    if new_global_max < current_val:
                        # Accept
                        all_used -= old_diffs
                        all_used |= new_diffs
                        current_rows[ri][pi] = new_val
                        row_diffs[ri] = new_diffs
                        static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                        row_maxes[ri] = cand_row_max
                        current_val = new_global_max
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        improved = True
                        break
                    elif new_global_max == current_val and pi == k:
                        # Same global max but smaller row max - still try
                        # Only accept if it doesn't hurt
                        pass
    
    return best_rows

# EVOLVE-BLOCK-END
