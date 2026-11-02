"""Hybrid: CSP/SA local search combined with a global row-replacement strategy
that treats each row independently against the union of other rows' diffs.
This structural change moves away from per-position mutations toward full-row
re-synthesis, which better exploits the row-level independence of the constraint."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Compute the set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    """Compute list of all pairwise differences."""
    lst = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            lst.append(row[j] - row[i])
    return lst

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

def _max_except(rows, exclude_idx):
    """Return the maximum entry across all rows except the excluded one."""
    m = 0
    for i, r in enumerate(rows):
        if i != exclude_idx:
            if r[-1] > m:
                m = r[-1]
    return m

def _greedy_init(n_rows, k, rng=None):
    """Build initial solution greedily with optional randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if rng is None:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            else:
                found = False
                scan = x + 1
                while scan <= x + 50:
                    new_diffs = {scan - a for a in row}
                    if not new_diffs & used and len(new_diffs) == len(set(new_diffs)):
                        row.append(scan)
                        used |= new_diffs
                        x = scan
                        found = True
                        break
                    scan += 1
                if not found:
                    scan = x + 1
                    while True:
                        new_diffs = {scan - a for a in row}
                        if not new_diffs & used and len(new_diffs) == len(set(new_diffs)):
                            row.append(scan)
                            used |= new_diffs
                            x = scan
                            found = True
                            break
                        scan += 1
        rows.append(row)
    return rows

def _random_valid_row(forbidden_diffs, max_limit, k, rng, max_candidates=8):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    row = [0]
    used_in_row = set()
    
    for _ in range(k):
        candidates = []
        start = row[-1] + 1
        end = max_limit if max_limit is not None else start + 300
        
        for x in range(start, end + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & forbidden_diffs:
                continue
            if new_diffs & used_in_row:
                continue
            candidates.append(x)
            if len(candidates) >= max_candidates:
                break
        
        if not candidates:
            return None
        
        # Weighted towards smaller values
        if rng.random() < 0.6:
            chosen_x = candidates[0]
        else:
            chosen_x = rng.choice(candidates)
        
        row.append(chosen_x)
        used_in_row |= {chosen_x - a for a in row[:-1]}
        
    return row

def _csp_build_row(used, k, deadline, rng=None, target=None):
    """Build a single row using CSP with forward checking, optionally bounded by target."""
    row = [0]
    current_pos = 1
    while current_pos <= k:
        if time.time() > deadline - 1:
            return None
        lo = row[-1] + 1
        hi = (target if target is not None else lo + 50)
        hi = max(hi, lo)
        
        candidates = []
        for val in range(lo, hi + 1):
            valid = True
            for i in range(len(row)):
                d = val - row[i]
                if d in used:
                    valid = False
                    break
            if not valid:
                continue
            diffs = set()
            for i in range(len(row)):
                d = val - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if valid:
                candidates.append(val)
            if len(candidates) >= 15:
                break
        
        if not candidates:
            # Expand search range
            for val in range(hi + 1, hi + 200):
                valid = True
                for i in range(len(row)):
                    d = val - row[i]
                    if d in used:
                        valid = False
                        break
                if not valid:
                    continue
                diffs = set()
                for i in range(len(row)):
                    d = val - row[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if valid:
                    candidates.append(val)
                    break
        
        if not candidates:
            return None
        
        if rng is not None and len(candidates) > 1:
            weights = [1.0 / (i + 1) for i in range(len(candidates))]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            chosen = candidates[-1]
            for v, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = v
                    break
        else:
            chosen = candidates[0]
        
        row.append(chosen)
        for i in range(len(row) - 1):
            used.add(row[-1] - row[i])
        current_pos += 1
    return row

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """
    SA with full-row replacement moves as the primary operator,
    combined with occasional per-position tweaks.
    """
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    cur_used = set()
    for dset in cur_row_diffs:
        cur_used |= dset
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    
    T = 8.0
    T_min = 0.05
    decay = 0.9997
    t0 = time.time()
    no_improve = 0
    
    while time.time() - t0 < time_budget:
        r = rng.random()
        
        if r < 0.70:
            # Full-row replacement
            ri = rng.randrange(n_rows)
            forbidden = cur_used - cur_row_diffs[ri]
            
            # Try to find a row with max <= cur_val first
            new_row = _random_valid_row(forbidden, cur_val, k, rng, max_candidates=8)
            
            if new_row is None:
                # Try slightly larger
                new_row = _random_valid_row(forbidden, cur_val + 5, k, rng, max_candidates=8)
            
            if new_row is None:
                # Try CSP-based
                fresh_used = set(forbidden)
                new_row = _csp_build_row(fresh_used, k, time.time() + 0.5, rng, target=cur_val + 10)
            
            if new_row is None:
                no_improve += 1
                T = max(T_min, T * decay)
                continue
            
            other_max = _max_except(cur, ri)
            new_max = max(other_max, new_row[-1])
            delta = new_max - cur_val
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                old_diffs = cur_row_diffs[ri]
                cur_used -= old_diffs
                new_diffs = _row_diffs_set(new_row)
                cur_used |= new_diffs
                cur_row_diffs[ri] = new_diffs
                cur[ri] = new_row
                cur_val = new_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r2[:] for r2 in cur]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        elif r < 0.85:
            # Per-position tweak
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            row = cur[ri]
            old_val = row[pi]
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else 10**6
            if hi < lo:
                T = max(T_min, T * decay)
                continue
            
            span = max(1, hi - lo + 1)
            candidates = []
            for _ in range(6):
                off = rng.randint(-span, span)
                nv = old_val + off
                if lo <= nv <= hi:
                    candidates.append(nv)
            for d in (1, -1, 2, -2):
                nv = old_val + d
                if lo <= nv <= hi:
                    candidates.append(nv)
            
            if not candidates:
                T = max(T_min, T * decay)
                continue
            
            nv = rng.choice(candidates)
            if nv == old_val:
                T = max(T_min, T * decay)
                continue
            
            cand_row = row[:]
            cand_row[pi] = nv
            if not _is_sidon(cand_row):
                T = max(T_min, T * decay)
                continue
            
            new_diffs = _row_diffs_set(cand_row)
            old_diffs = cur_row_diffs[ri]
            other_diffs = cur_used - old_diffs
            if new_diffs & other_diffs:
                T = max(T_min, T * decay)
                continue
            
            other_max = _max_except(cur, ri)
            new_max = max(other_max, max(cand_row))
            delta = new_max - cur_val
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur_row_diffs[ri] = new_diffs
                cur[ri] = cand_row
                cur_val = new_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r2[:] for r2 in cur]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        else:
            # Restart from best
            cur = [r2[:] for r2 in best]
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            cur_used = set()
            for dset in cur_row_diffs:
                cur_used |= dset
            cur_val = best_val
            T = max(T_min, 5.0)
            no_improve = 0
        
        T = max(T_min, T * decay)
        
        if no_improve > 20000:
            cur = [r2[:] for r2 in best]
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            cur_used = set()
            for dset in cur_row_diffs:
                cur_used |= dset
            cur_val = best_val
            T = 5.0
            no_improve = 0
    
    return best, best_val

def _polish_rows(rows, n_rows, k, deadline, seed):
    """Final polish: try to rebuild each row to minimize its max entry."""
    rng = random.Random(seed)
    cur = [r[:] for r in rows]
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    cur_used = set()
    for dset in cur_row_diffs:
        cur_used |= dset
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    
    for _ in range(3):
        for ri in range(n_rows):
            if time.time() > deadline - 2:
                break
            forbidden = cur_used - cur_row_diffs[ri]
            # Try to find a row with max < current best_val
            for target in range(best_val - 1, best_val - 15, -1):
                new_row = _random_valid_row(forbidden, target, k, rng, max_candidates=10)
                if new_row is not None:
                    old_diffs = cur_row_diffs[ri]
                    cur_used -= old_diffs
                    new_diffs = _row_diffs_set(new_row)
                    cur_used |= new_diffs
                    cur_row_diffs[ri] = new_diffs
                    cur[ri] = new_row
                    cur_val = _max_entry(cur)
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
                    break
    
    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Phase 1: Diverse initializations
    inits = []
    inits.append(_greedy_init(n_rows, k, rng=None))
    for s in range(5):
        inits.append(_greedy_init(n_rows, k, rng=random.Random(1000 + s)))
    
    if time.time() < deadline - 15:
        # CSP-based initialization
        used = set()
        csp_rows = []
        rng_csp = random.Random(42)
        ok = True
        for r in range(n_rows):
            if time.time() > deadline - 10:
                ok = False
                break
            row = _csp_build_row(used, k, deadline, rng_csp)
            if row is None:
                ok = False
                break
            csp_rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        if ok and csp_rows and _valid(csp_rows, n_rows, k):
            inits.append(csp_rows)
    
    inits.sort(key=_max_entry)
    top_inits = inits[:4]

    # Phase 2: SA trajectories with full-row replacement
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    n_sa = 3
    budget_per_sa = max(5.0, remaining / n_sa - 2.0)
    overall_best = None
    overall_best_val = float('inf')
    
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 8:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)

    # Phase 3: Polish
    if time.time() < deadline - 4:
        b, bv = _polish_rows(overall_best, n_rows, k, deadline, seed=999)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _greedy_init(n_rows, k, rng=None)

    return overall_best

# EVOLVE-BLOCK-END
