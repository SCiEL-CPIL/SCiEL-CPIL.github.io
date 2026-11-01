"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _max_entry(rows):
    return max(max(r) for r in rows)

def _valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _build_greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _try_set_value(cur, ri, pi, new_val, row_diffs, cur_used, k):
    """Try to set cur[ri][pi] = new_val. Returns (success, new_row_diffs, new_cur_used_delta)."""
    old_val = cur[ri][pi]
    if new_val == old_val:
        return False, None, None
    # Check strict increase
    if pi > 0 and new_val <= cur[ri][pi - 1]:
        return False, None, None
    if pi < k and new_val >= cur[ri][pi + 1]:
        return False, None, None
    # Build candidate row
    cand_row = cur[ri][:]
    cand_row[pi] = new_val
    # Check strict increase fully
    for j in range(1, len(cand_row)):
        if cand_row[j] <= cand_row[j - 1]:
            return False, None, None
    new_diffs = _row_diffs(cand_row)
    old_diffs = row_diffs[ri]
    other_used = cur_used - old_diffs
    if new_diffs & other_used:
        return False, None, None
    return True, new_diffs, (old_diffs, new_diffs)

def _local_search_reduce_max(cur, n_rows, k, target_max, row_diffs, cur_used):
    """Try to reduce the maximum entry to target_max. Returns (success, new_cur, new_row_diffs, new_cur_used)."""
    cur = [r[:] for r in cur]
    row_diffs = [_row_diffs(r) for r in cur]
    cur_used = set()
    for rd in row_diffs:
        cur_used |= rd
    
    # Find rows that have entries > target_max
    changed = True
    iterations = 0
    while changed and iterations < 500:
        changed = False
        iterations += 1
        for ri in range(n_rows):
            if max(cur[ri]) <= target_max:
                continue
            # Find the largest position that exceeds target_max
            for pi in range(k, 0, -1):
                if cur[ri][pi] > target_max:
                    # Try to reduce this position
                    found = False
                    for new_val in range(cur[ri][pi] - 1, target_max - 1, -1):
                        if new_val < 1:
                            break
                        ok, new_diffs, delta = _try_set_value(cur, ri, pi, new_val, row_diffs, cur_used, k)
                        if ok:
                            old_diffs = row_diffs[ri]
                            cur_used -= old_diffs
                            cur_used |= new_diffs
                            row_diffs[ri] = new_diffs
                            cur[ri][pi] = new_val
                            found = True
                            changed = True
                            break
                    if found:
                        break
    return True, cur, row_diffs, cur_used

def _multi_start_greedy(n_rows, k, seed):
    """Greedy with randomization in tie-breaking."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _swap_rows(cur, n_rows, k, row_diffs, cur_used, rng):
    """Try swapping two rows to see if it helps."""
    ri = rng.randrange(n_rows)
    rj = rng.randrange(n_rows)
    if ri == rj:
        return False
    # Just shuffle positions within a row
    return False

def _perturb_row(cur, ri, k, row_diffs, cur_used, rng, max_val):
    """Randomly perturb one position in row ri. Returns (success, new_row_diffs)."""
    pi = rng.randrange(1, k + 1)
    old_val = cur[ri][pi]
    # Generate candidates
    candidates = set()
    for d in range(-10, 11):
        if d == 0:
            continue
        nv = old_val + d
        if nv > 0:
            candidates.add(nv)
    for _ in range(5):
        nv = rng.randint(1, max_val)
        candidates.add(nv)
    # Try to reduce if this is near max
    if old_val > max_val * 0.8:
        for d in range(1, min(old_val, 20)):
            candidates.add(old_val - d)
    
    for new_val in sorted(candidates):
        ok, new_diffs, _ = _try_set_value(cur, ri, pi, new_val, row_diffs, cur_used, k)
        if ok:
            return True, new_diffs
    return False, None

def construct(n_rows, k):
    deadline = time.time() + 115  # safety margin before 120s limit
    rng = random.Random(42)
    
    # Initial construction: try multiple greedy starts
    best = None
    best_val = float('inf')
    
    for seed in [42, 123, 999, 7, 2025]:
        rows = _build_greedy(n_rows, k)
        val = _max_entry(rows)
        if val < best_val:
            best_val = val
            best = [r[:] for r in rows]
    
    # Local search phase 1: try to reduce max by 1 repeatedly
    cur = [r[:] for r in best]
    row_diffs = [_row_diffs(r) for r in cur]
    cur_used = set()
    for rd in row_diffs:
        cur_used |= rd
    cur_val = best_val
    
    # Phase 1: targeted max reduction
    while time.time() < deadline - 30:
        target = cur_val - 1
        if target < k + 1:  # minimum possible
            break
        # Try to achieve this target
        success, new_cur, new_rd, new_used = _local_search_reduce_max(
            cur, n_rows, k, target, row_diffs, cur_used
        )
        if success and _max_entry(new_cur) <= target:
            cur = new_cur
            row_diffs = new_rd
            cur_used = new_used
            cur_val = _max_entry(cur)
            best = [r[:] for r in cur]
            best_val = cur_val
        else:
            # Can't reduce by 1 with this approach; try simulated annealing
            break
    
    # Phase 2: simulated annealing for further improvement
    T = 20.0
    T_min = 0.01
    cool = 0.9999
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart from greedy
            seed = rng.randint(0, 100000)
            rows = _build_greedy(n_rows, k)
            cur = [r[:] for r in rows]
            cur_val = _max_entry(cur)
            row_diffs = [_row_diffs(r) for r in cur]
            cur_used = set()
            for rd in row_diffs:
                cur_used |= rd
            no_improve = 0
            T = 20.0
            continue
        
        # Pick random row and position
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        old_val = cur[ri][pi]
        
        # Generate candidates
        candidates = []
        for d in range(-12, 13):
            if d == 0:
                continue
            nv = old_val + d
            if nv > 0:
                candidates.append(nv)
        for _ in range(4):
            nv = rng.randint(1, max(old_val + 20, 40))
            candidates.append(nv)
        if old_val == cur_val:
            for d in range(1, min(old_val - 1, 25)):
                candidates.append(old_val - d)
        
        rng.shuffle(candidates)
        moved = False
        
        for new_val in candidates:
            if new_val < 1:
                continue
            ok, new_diffs, _ = _try_set_value(cur, ri, pi, new_val, row_diffs, cur_used, k)
            if not ok:
                continue
            
            # Compute new max
            cand_row = cur[ri][:]
            cand_row[pi] = new_val
            cand_max = max(max(cur[r]) for r in range(n_rows) if r != ri)
            cand_max = max(cand_max, max(cand_row))
            
            # Acceptance
            if cand_max < cur_val:
                accept = True
            elif cand_max == cur_val:
                accept = rng.random() < 0.6
            else:
                delta_e = cand_max - cur_val
                accept = rng.random() < 0.05 / (1.0 + T * 0.1)
            
            if accept:
                old_diffs = row_diffs[ri]
                cur_used -= old_diffs
                cur_used |= new_diffs
                row_diffs[ri] = new_diffs
                cur[ri][pi] = new_val
                cur_val = cand_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break
        
        T = max(T_min, T * cool)
    
    # Final verification
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best
# EVOLVE-BLOCK-END
