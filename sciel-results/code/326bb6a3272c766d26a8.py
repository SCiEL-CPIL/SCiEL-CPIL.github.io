"""Optimized SA with strict time guards and faster local moves."""
import time
import random
import math
import sys

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
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s

def _is_sidon(row):
    """Check if all pairwise differences in the row are distinct."""
    n = len(row)
    seen = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            if d in seen:
                return False
            seen.add(d)
    return True

def _greedy_row_with_update(used, k, max_val=None, rng=None):
    """Build a single row greedily (optionally randomized), updating used as we go."""
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 10000
    while len(row) < k + 1:
        if rng is None:
            x += 1
            if x > limit:
                return None, used
            new_diffs = [x - a for a in row]
            if any(d in used for d in new_diffs):
                continue
            if len(new_diffs) != len(set(new_diffs)):
                continue
            row.append(x)
            for d in new_diffs:
                used.add(d)
        else:
            valid_xs = []
            scan = x + 1
            # Limit scan window for speed
            end_scan = min(x + 30, limit)
            while scan <= end_scan and len(valid_xs) < 5:
                new_diffs = [scan - a for a in row]
                if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                    valid_xs.append(scan)
                scan += 1
            if not valid_xs:
                # Fallback: scan further if we are close to limit
                scan = x + 1
                while scan <= limit:
                    new_diffs = [scan - a for a in row]
                    if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                        valid_xs.append(scan)
                        break
                    scan += 1
                if not valid_xs:
                    return None, used
            
            # Weighted pick biased toward smaller
            weights = [1.0 / (i + 1) for i in range(len(valid_xs))]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            chosen = valid_xs[-1]
            for v, w in zip(valid_xs, weights):
                cum += w
                if r <= cum:
                    chosen = v
                    break
            row.append(chosen)
            for a in row[:-1]:
                used.add(chosen - a)
            x = chosen
    return row, used

def _build_greedy(n_rows, k, rng=None):
    """Fast greedy: each row takes smallest (or random-window) next mark with no diff collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = _greedy_row_with_update(used, k, rng=rng)
        if row is None:
            # Deterministic fallback with no limit
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = [x - a for a in row]
                if not any(d in used for d in new) and len(new) == len(set(new)):
                    row.append(x)
                    used |= set(new)
        rows.append(row)
    return rows

def _sa_move_single_pos(cur, cur_used, ri, pi, rng, k, cur_max):
    """Try a single-position move on row ri at index pi.
    Returns (new_row, new_max, delta) or (None, 0, 0)."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else 10**6
    if hi < lo:
        return None, 0, 0
    
    # Try a small set of candidates
    candidates = set()
    # Near moves
    for d in range(-5, 6):
        nv = old_val + d
        if lo <= nv <= hi and nv != old_val:
            candidates.add(nv)
    # Random moves
    for _ in range(5):
        nv = rng.randint(lo, hi)
        if nv != old_val:
            candidates.add(nv)
            
    if not candidates:
        return None, 0, 0
        
    # Shuffle to ensure randomness
    cand_list = list(candidates)
    rng.shuffle(cand_list)
    
    for nv in cand_list:
        cand_row = row[:]
        cand_row[pi] = nv
        if not _is_sidon(cand_row):
            continue
        new_diffs = _row_diffs_set(cand_row)
        old_diffs = _row_diffs_set(row)
        other_diffs = cur_used - old_diffs
        if new_diffs & other_diffs:
            continue
        
        # Valid move found
        new_row_max = max(cand_row)
        # Calculate new global max
        # We need max of all other rows and this new row
        # Optimization: if cur[ri] was the unique max, we might lower it
        # But simply taking max of all is safe and O(N*K)
        # To speed up, we can precompute max of others? No, just compute.
        new_max = 0
        for i, r in enumerate(cur):
            if i == ri:
                if new_row_max > new_max:
                    new_max = new_row_max
            else:
                if max(r) > new_max:
                    new_max = max(r)
        
        delta = new_max - cur_max
        return cand_row, new_max, delta
        
    return None, 0, 0

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory starting from initial_rows.
    Returns the best valid rows seen."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    T0 = 10.0
    T = T0
    decay = 0.999
    min_T = 0.1
    no_improve_steps = 0
    
    steps = 0
    t0 = time.time()
    
    while time.time() - t0 < time_budget:
        steps += 1
        r = rng.random()
        
        if r < 0.6:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, new_max, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k, cur_val)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val = new_max
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
                        no_improve_steps = 0
                        T = min(T0, T * 1.2)
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
        elif r < 0.8:
            # Row swap
            if n_rows >= 2:
                ri = rng.randrange(n_rows)
                rj = rng.randrange(n_rows)
                if ri != rj:
                    # Swap
                    cur[ri], cur[rj] = cur[rj][:], cur[ri][:]
                    # Max might change? No, the set of rows is the same.
                    # So cur_val remains the same.
                    # However, we should check if this swap helps?
                    # Actually, swapping rows doesn't change the set of differences or the max.
                    # It only changes which row is which.
                    # So this move is neutral in objective.
                    # It might help if we have a specific structure in mind, but for pure max minimization, it's neutral.
                    # We can skip this to save time, or use it to break symmetry.
                    # Let's skip it to save time as it doesn't improve objective directly.
                    pass
        else:
            # Re-resolve a row (more expensive but can escape local minima)
            ri = rng.randrange(n_rows)
            other_used = set()
            for i, r in enumerate(cur):
                if i != ri:
                    other_used |= _row_diffs_set(r)
            
            fresh_used = set(other_used)
            # Try to find a better row for ri
            old_row = cur[ri]
            old_max_row = max(old_row)
            
            # Try greedy with a slightly smaller target if possible
            target = old_max_row - 1
            if target < old_row[-2] + 1:
                target = old_max_row
            
            new_row, _ = _greedy_row_with_update(fresh_used, k, max_val=target, rng=rng)
            if new_row is None:
                # Try with original max
                new_row, _ = _greedy_row_with_update(fresh_used, k, max_val=old_max_row, rng=rng)
            
            if new_row is not None:
                new_diffs = _row_diffs_set(new_row)
                if not (new_diffs & other_used):
                    new_row_max = max(new_row)
                    # Calculate new global max
                    new_max = 0
                    for i, r in enumerate(cur):
                        if i == ri:
                            if new_row_max > new_max:
                                new_max = new_row_max
                        else:
                            if max(r) > new_max:
                                new_max = max(r)
                    
                    delta = new_max - cur_val
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        old_diffs = _row_diffs_set(old_row)
                        cur_used -= old_diffs
                        cur_used |= new_diffs
                        cur[ri] = new_row
                        cur_val = new_max
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r2[:] for r2 in cur]
                            no_improve_steps = 0
                            T = min(T0, T * 1.2)
                        else:
                            no_improve_steps += 1
                    else:
                        no_improve_steps += 1
            else:
                no_improve_steps += 1

        T = max(min_T, T * decay)
        
        # Re-anchoring
        if steps % 1000 == 0 and no_improve_steps > 2000:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            no_improve_steps = 0
            T = T0

    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 120s limit, keep 5s buffer

    # Phase 1: Initialization
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(5):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    
    inits.sort(key=_max_entry)
    best_init = inits[0]
    best_val = _max_entry(best_init)
    
    # Phase 2: SA
    remaining = deadline - time.time()
    if remaining > 10:
        # Run one long SA from the best init
        b, bv = _sa_run(n_rows, k, best_init, remaining - 5.0, seed=42)
        if bv < best_val:
            best_val = bv
            best_init = b
    
    # Phase 3: Final Validation
    if not _valid(best_init, n_rows, k):
        # Fallback to a valid greedy
        best_init = _build_greedy(n_rows, k, rng=None)

    return best_init
# EVOLVE-BLOCK-END
