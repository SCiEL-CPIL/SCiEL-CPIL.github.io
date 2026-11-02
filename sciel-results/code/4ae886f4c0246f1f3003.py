"""
Combined approach: Fast bitmask-based iterative deepening for finding a valid solution
at a target M, followed by aggressive simulated annealing polishing.

The core structural change is using bitmasks for O(1) difference conflict checks,
which enables much faster backtracking row construction. This allows the iterative
deepening phase to explore tighter M bounds within the time budget. The SA phase
then polishes the result by trying to reduce the max entry further.
"""
import time
import random
import math

# EVOLVE-BLOCK-START

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    """Check if all pairwise differences in the row are distinct."""
    n = len(row)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no diff collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = [x - a for a in row]
            if any(d in used for d in new_diffs):
                continue
            if len(new_diffs) != len(set(new_diffs)):
                continue
            row.append(x)
            for d in new_diffs:
                used.add(d)
        rows.append(row)
    return rows

def _backtrack_row_fast(prefix, local_diffs_mask, global_used_mask, k, max_val, rng, 
                        diff_to_bit, bit_to_diff, max_diff, time_limit=None, start_time=None):
    """
    Recursively complete a row using bitmask-based conflict checks.
    prefix: list of ints, starts with 0.
    local_diffs_mask: bitmask of differences already in this row.
    global_used_mask: bitmask of differences used by other rows.
    """
    if len(prefix) == k + 1:
        return prefix[:]
    
    if time_limit is not None and time.time() - start_time > time_limit:
        return None
    
    last = prefix[-1]
    remaining = k + 1 - len(prefix)
    min_span_needed = remaining * (remaining + 1) // 2
    if last + min_span_needed > max_val:
        return None
    
    start_val = last + 1
    end_val = min(start_val + 15, max_val + 1)
    
    valid_cands = []
    for v in range(start_val, end_val):
        new_diffs = [v - a for a in prefix]
        
        # Check for internal Sidon property and conflicts using bitmask
        new_mask = 0
        conflict = False
        for d in new_diffs:
            if d > max_diff:
                conflict = True
                break
            bit = 1 << d
            if new_mask & bit:
                conflict = True
                break
            new_mask |= bit
        
        if conflict:
            continue
        
        # Check against local and global used
        if (new_mask & local_diffs_mask) or (new_mask & global_used_mask):
            continue
        
        valid_cands.append((v, new_diffs, new_mask))
        
        if len(valid_cands) >= 8:
            break
    
    if not valid_cands:
        return None
    
    rng.shuffle(valid_cands)
    
    for v, new_diffs, new_mask in valid_cands:
        new_prefix = prefix + [v]
        new_local_mask = local_diffs_mask | new_mask
        
        res = _backtrack_row_fast(new_prefix, new_local_mask, global_used_mask, k, max_val, rng,
                                  diff_to_bit, bit_to_diff, max_diff, time_limit, start_time)
        if res is not None:
            return res
            
    return None

def _solve_at_M_fast(n_rows, k, M, rng, time_limit=3.0):
    """
    Try to construct n_rows rows of k+1 elements from {0,...,M} with all diffs distinct.
    Uses bitmask-based backtracking for speed.
    """
    start = time.time()
    
    # Precompute bit mappings
    max_diff = M
    diff_to_bit = {d: 1 << d for d in range(max_diff + 1)}
    
    global_used_mask = 0
    rows = []
    
    for ri in range(n_rows):
        if time.time() - start > time_limit:
            return None
        
        # Try to find a row with multiple random restarts
        row = None
        for attempt in range(30):
            if time.time() - start > time_limit:
                return None
            res = _backtrack_row_fast([0], 0, global_used_mask, k, M, rng,
                                      diff_to_bit, None, max_diff, 
                                      time_limit=time_limit - (time.time() - start),
                                      start_time=start)
            if res is not None:
                row = res
                break
        
        if row is None:
            return None
        
        rows.append(row)
        
        # Update global used mask
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                global_used_mask |= (1 << d)
        
    return rows

def _sa_move_single_pos(cur, cur_used, ri, pi, rng, k):
    """Try a single-position move on row ri at index pi."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else 10**6
    if hi < lo:
        return None, 0
    span = max(1, hi - lo + 1)
    candidates = []
    for _ in range(10):
        off = rng.randint(-span, span)
        nv = old_val + off
        if lo <= nv <= hi:
            candidates.append(nv)
    for d in (1, -1, 2, -2, 3, -3):
        nv = old_val + d
        if lo <= nv <= hi:
            candidates.append(nv)
    if not candidates:
        return None, 0
    nv = rng.choice(candidates)
    if nv == old_val:
        return None, 0
    cand_row = row[:]
    cand_row[pi] = nv
    if not _is_sidon(cand_row):
        return None, 0
    new_diffs = _row_diffs_set(cand_row)
    old_diffs = _row_diffs_set(row)
    other_diffs = cur_used - old_diffs
    if new_diffs & other_diffs:
        return None, 0
    old_max = _max_entry(cur)
    new_max = max(max(cand_row), max(max(r) for i, r in enumerate(cur) if i != ri))
    delta = new_max - old_max
    return cand_row, delta

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    T = 15.0
    decay = 0.999
    min_T = 0.02

    t0 = time.time()
    while time.time() - t0 < time_budget:
        r = rng.random()
        if r < 0.6:
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val = _max_entry(cur)
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
        else:
            # Re-solve a row with a tighter bound using fast backtracking
            ri = rng.randrange(n_rows)
            old_max = max(cur[ri])
            target = old_max - rng.randint(1, 8)
            if target > 0:
                other_used = set()
                for i, r in enumerate(cur):
                    if i != ri:
                        other_used |= _row_diffs_set(r)
                
                # Use fast bitmask-based solver
                max_diff = target
                diff_to_bit = {d: 1 << d for d in range(max_diff + 1)}
                global_used_mask = 0
                for d in other_used:
                    if d <= max_diff:
                        global_used_mask |= (1 << d)
                
                new_row = None
                for attempt in range(10):
                    res = _backtrack_row_fast([0], 0, global_used_mask, k, target, rng,
                                              diff_to_bit, None, max_diff,
                                              time_limit=0.5, start_time=time.time())
                    if res is not None:
                        new_row = res
                        break
                
                if new_row is not None:
                    new_diffs = _row_diffs_set(new_row)
                    if not (new_diffs & other_used):
                        old_diffs = _row_diffs_set(cur[ri])
                        cur_used -= old_diffs
                        cur_used |= new_diffs
                        cur[ri] = new_row
                        cur_val = _max_entry(cur)
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r2[:] for r2 in cur]

        T = max(min_T, T * decay)

    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Get initial upper bound via Greedy
    best = _build_greedy(n_rows, k)
    if best is None:
        best = [[0, 1, 3, 7, 15, 31]] * n_rows
        best_val = 31
    else:
        best_val = _max_entry(best)
        
    # 2. Calculate Lower Bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # 3. Iterative Deepening on M using fast bitmask-based backtracking
    # We try to find a solution with max entry <= M, starting from best_val-1 down to lb
    time_for_deepening = 70.0
    deepen_deadline = start_time + time_for_deepening
    
    current_best_val = best_val
    current_best = best
    
    # Try M values in descending order to improve the current best
    consecutive_fails = 0
    for M in range(current_best_val - 1, lb - 1, -1):
        if time.time() > deepen_deadline:
            break
        
        # Try to solve at this M with random restarts
        max_restarts = 15
        solved = False
        for restart in range(max_restarts):
            if time.time() > deepen_deadline:
                break
            rng = random.Random(54321 + M * 1000 + restart)
            res = _solve_at_M_fast(n_rows, k, M, rng, time_limit=2.0)
            if res is not None:
                if _valid(res, n_rows, k):
                    current_best = res
                    current_best_val = M
                    solved = True
                    consecutive_fails = 0
                    break
        
        if not solved:
            consecutive_fails += 1
            # If we've failed at many consecutive M values, stop
            if consecutive_fails >= 5:
                break
    
    # 4. SA Polish
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and current_best is not None:
        b, bv = _sa_run(n_rows, k, current_best, remaining_time - 2.0, seed=777)
        if bv < current_best_val:
            current_best_val = bv
            current_best = b

    # Final validation
    if not _valid(current_best, n_rows, k):
        current_best = _build_greedy(n_rows, k)
        if current_best is None:
            current_best = [[0, 1, 2, 3, 4, 5]] * n_rows

    return current_best

# EVOLVE-BLOCK-END
