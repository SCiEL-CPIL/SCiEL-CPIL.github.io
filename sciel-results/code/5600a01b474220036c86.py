import time
import random
import math
import sys

sys.setrecursionlimit(10000)

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, seed=None):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rng = random.Random(seed) if seed is not None else random
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            for a in row:
                if (x - a) in used:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
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

def _randomized_greedy_row(forbidden_diffs, max_limit, k, max_candidates=5, min_gap=1):
    """
    Attempts to construct a valid row of length k+1 starting at 0,
    such that all its differences are not in forbidden_diffs,
    and the last element is <= max_limit.
    
    Returns the row if successful, None otherwise.
    """
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        x = last_val + min_gap
        
        search_limit = max_limit
        
        while x <= search_limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
                if len(candidates) >= max_candidates:
                    break
            x += 1
            
        if not candidates:
            return None
            
        chosen_x = random.choice(candidates)
        
        row.append(chosen_x)
        for a in row[:-1]:
            used_in_row.add(chosen_x - a)
        last_val = chosen_x
        
    return row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Multi-start Initialization
    best_rows = None
    best_val = float('inf')
    
    # Try several different greedy initializations with different seeds
    for seed in range(10):
        if time.time() > deadline - 60:
            break
        rows = _greedy_init(n_rows, k, seed=seed * 100 + 42)
        val = _global_max(rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
    
    # Fallback if all seeds failed (shouldn't happen)
    if best_rows is None:
        best_rows = _greedy_init(n_rows, k, seed=0)
        best_val = _global_max(best_rows)

    # --- Phase 1: Iterated Local Search with Row Replacement (Exploration) ---
    # This phase uses the structural idea from Inspiration 1.
    # We replace entire rows to escape local minima.
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    T_replace = 50.0
    cool_replace = 0.99
    no_improve_replace = 0
    max_no_improve_replace = 500
    
    while time.time() < deadline - 30.0 and no_improve_replace < max_no_improve_replace:
        # Pick a row to replace
        ri = random.randrange(n_rows)
        
        # Calculate forbidden diffs (union of diffs of all other rows)
        forbidden = set()
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs[j]
        
        # Try to generate a new row with max <= current_val
        new_row = _randomized_greedy_row(forbidden, current_val, k, max_candidates=10)
        
        if new_row is None:
            # Try to find a row with max <= current_val + 5 (exploration)
            new_row = _randomized_greedy_row(forbidden, current_val + 5, k, max_candidates=10)
            if new_row is None:
                no_improve_replace += 1
                T_replace = max(1.0, T_replace * cool_replace)
                continue
        
        new_row_diffs = _row_diffs_set(new_row)
        new_row_max = new_row[-1]
        
        # Check validity: new_row_diffs must be disjoint from forbidden
        if new_row_diffs & forbidden:
            no_improve_replace += 1
            continue
            
        # Calculate new global max
        old_row_max = current_rows[ri][-1]
        
        other_maxes = [current_rows[j][-1] for j in range(n_rows) if j != ri]
        new_global_max = max(max(other_maxes) if other_maxes else 0, new_row_max)
        
        delta = new_global_max - current_val
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if random.random() < 0.1:
                accept = True
        else:
            if T_replace > 1.0:
                prob = math.exp(-delta / T_replace)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current_rows[ri] = new_row
            row_diffs[ri] = new_row_diffs
            current_val = new_global_max
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
                no_improve_replace = 0
            else:
                no_improve_replace += 1
                
            T_replace = max(1.0, T_replace * cool_replace)
        else:
            no_improve_replace += 1
            T_replace = max(1.0, T_replace * cool_replace)

    # --- Phase 2: Standard Simulated Annealing with Single-Point Mutations (Exploitation) ---
    # This phase uses the refinement machinery from the current program.
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    T = 20.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline - 5.0 and no_improve < max_no_improve:
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            new_v = old_val - d
            candidates.append(new_v)
            
        # Increasing candidates (for exploration)
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            new_v = old_val + d
            candidates.append(new_v)
        
        # Random jump
        if random.random() < 0.1:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
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
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
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
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 20.0
            no_improve = 0

    # --- Phase 3: Final Polishing (Targeted Reduction) ---
    # Try to reduce scope by 1 or 2 units using a hard cap.
    
    current_best = best_val
    for target_val in range(current_best - 1, max(100, current_best - 5), -1):
        if time.time() > deadline - 2:
            break
            
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used(current_rows)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        T = 10.0
        T_min = 0.1
        cool = 0.999
        no_improve = 0
        max_steps = 3000
        
        for _ in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else target_val
            
            if min_allowed > max_allowed:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            new_val = random.randint(min_allowed, max_allowed)
            
            if new_val == old_val:
                continue
                
            static_pi = static_diffs[ri][pi]
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            old_diffs = row_diffs[ri]
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            if new_global_max > target_val:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if new_global_max <= target_val:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            T = max(T_min, T * cool)
            
            if no_improve > 500:
                break
                
        if best_val <= target_val:
            continue
        else:
            pass

    return best_rows

# EVOLVE-BLOCK-END
