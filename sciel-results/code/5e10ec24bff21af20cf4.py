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

def _randomized_greedy_row(k, forbidden_diffs, max_limit, max_candidates=5):
    """
    Builds a row of k+1 elements starting at 0, strictly increasing,
    such that no difference is in forbidden_diffs and all are <= max_limit.
    At each step, it finds the first `max_candidates` valid integers and picks one at random.
    Returns None if no such row can be constructed.
    """
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        x = last_val + 1
        # We stop looking if we exceed max_limit or if we've looked far enough without finding candidates
        # Usually valid candidates are close, but sometimes we need to jump.
        # However, for "tightening" we want to stay low.
        limit_check = max_limit + 100 if max_limit is not None else 10000
        while x < limit_check:
            new_diffs = {x - a for a in row}
            # Check internal distinctness (new_diffs & used_in_row) and external (new_diffs & forbidden)
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
                if len(candidates) >= max_candidates:
                    break
            x += 1
            
        if not candidates:
            return None
        
        chosen_x = random.choice(candidates)
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Track state
    current_rows = [r[:] for r in rows]
    current_val = best_val
    row_maxes = [max(r) for r in rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    # 2. Main Phase: Simulated Annealing with Randomized Greedy Row Replacement
    T = 50.0
    T_min = 0.5
    cool = 0.9998
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Decide between local move (SA) and global move (Row Replacement)
        # As T drops, we rely more on local moves.
        # Early on, use row replacement to find good structures.
        if T > 5.0 and random.random() < 0.3:
            # --- Row Replacement Move ---
            ri = random.randrange(n_rows)
            
            # Compute forbidden diffs (all diffs except row ri)
            forbidden = set()
            for j in range(n_rows):
                if j != ri:
                    forbidden |= row_diffs[j]
            
            # Try to find a new row that fits within the current best_val
            # This is the "tightening" strategy from Inspiration 1
            new_row = _randomized_greedy_row(k, forbidden, best_val, max_candidates=10)
            
            if new_row is None:
                # If we can't fit within best_val, try to fit within current_val + some slack
                # or just find any valid row to explore
                new_row = _randomized_greedy_row(k, forbidden, current_val + 20, max_candidates=10)
            
            if new_row is not None:
                new_row_max = max(new_row)
                old_row_max = row_maxes[ri]
                
                # Calculate new global max
                row_maxes[ri] = new_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    # Accept equal max if it changes the row structure (might help later)
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Update state
                    all_used -= row_diffs[ri]
                    new_diffs = _row_diffs_set(new_row)
                    all_used |= new_diffs
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    static_diffs[ri] = _build_static_diffs(new_row, k)
                    row_maxes[ri] = new_row_max
                    current_val = new_global_max
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # --- Local SA Move ---
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(20, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
                
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
            max_increase = min(10, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
                
            if random.random() < 0.1:
                low = min_allowed
                high = max_allowed
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            random.shuffle(candidates)
            static_pi = static_diffs[ri][pi]
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                    
                new_diffs, valid_internal = _compute_new_diffs_fast(
                    current_rows[ri], pi, new_val, static_pi)
                
                if not valid_internal:
                    continue
                    
                old_diffs = row_diffs[ri]
                
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in old_diffs:
                        conflict = True
                        break
                
                if conflict:
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
                    if random.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
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
        
        # Restart if stuck
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 50.0
            no_improve = 0

    # 3. Final Polish Phase: Aggressive Tightening
    # Try to replace each row with a greedy version constrained to best_val
    # Then do a final local search to fix any issues or squeeze more.
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    polish_deadline = min(deadline, start_time + 118.0)
    
    # Try multiple passes of tightening
    for _ in range(10):
        if time.time() > polish_deadline:
            break
        improved = False
        for ri in range(n_rows):
            forbidden = set()
            for j in range(n_rows):
                if j != ri:
                    forbidden |= row_diffs[j]
            
            # Try to find a row strictly smaller than current row max, or at least <= best_val
            target_max = row_maxes[ri] - 1
            if target_max < 0:
                target_max = best_val
            
            new_row = _randomized_greedy_row(k, forbidden, target_max, max_candidates=20)
            
            if new_row is not None:
                new_row_max = max(new_row)
                # Check if it actually improves the global max or maintains it
                old_row_max = row_maxes[ri]
                row_maxes[ri] = new_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                if new_global_max < current_val:
                    # Accept
                    all_used -= row_diffs[ri]
                    new_diffs = _row_diffs_set(new_row)
                    all_used |= new_diffs
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    static_diffs[ri] = _build_static_diffs(new_row, k)
                    row_maxes[ri] = new_row_max
                    current_val = new_global_max
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    improved = True
                    break # Restart loop with new forbidden set
            else:
                # Try to maintain best_val but maybe change structure
                new_row = _randomized_greedy_row(k, forbidden, best_val, max_candidates=10)
                if new_row is not None and max(new_row) < row_maxes[ri]:
                     # This is a valid row that is smaller than the current one
                     # It might not reduce global max if another row is the max, but it frees up space
                     # Let's accept if it doesn't increase global max
                     new_row_max = max(new_row)
                     old_row_max = row_maxes[ri]
                     row_maxes[ri] = new_row_max
                     new_global_max = max(row_maxes)
                     row_maxes[ri] = old_row_max
                     
                     if new_global_max <= current_val:
                         all_used -= row_diffs[ri]
                         new_diffs = _row_diffs_set(new_row)
                         all_used |= new_diffs
                         current_rows[ri] = new_row
                         row_diffs[ri] = new_diffs
                         static_diffs[ri] = _build_static_diffs(new_row, k)
                         row_maxes[ri] = new_row_max
                         current_val = new_global_max
                         if current_val < best_val:
                             best_val = current_val
                             best_rows = [r[:] for r in current_rows]
                         improved = True
                         break
        if not improved:
            break

    # Final local search squeeze on the best solution
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    # Very low temp local search
    T = 1.0
    T_min = 0.1
    cool = 0.99
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < polish_deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(5, old_val - min_allowed)
        
        candidates = [old_val - d for d in range(1, max_decrease + 1)]
        random.shuffle(candidates)
        
        static_pi = static_diffs[ri][pi]
        moved = False
        
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
                
            old_diffs = row_diffs[ri]
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
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
                if random.random() < 0.05:
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

    return best_rows

# EVOLVE-BLOCK-END
