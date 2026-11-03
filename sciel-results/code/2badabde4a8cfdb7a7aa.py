import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = 0.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(k, n_rows, rng=None, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _row_rebuild(row, k, diff_counts, rng, max_scope):
    best_row = None
    best_score = float('inf')
    
    for attempt in range(10):
        new_row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.5:
                start = x + rng.randint(1, 5)
            # Limit search range to max_scope to keep it bounded
            for t in range(start, max_scope + 1):
                conflicts = 0
                new_diffs = []
                for a in new_row:
                    d = t - a
                    if d <= 0:
                        continue
                    new_diffs.append(d)
                    if diff_counts.get(d, 0) > 0:
                        conflicts += 1
                if conflicts == 0:
                    new_row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
                elif attempt < 5 and conflicts <= 1 and rng.random() < 0.2:
                    new_row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok and len(new_row) == k + 1:
            score = 0
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if diff_counts.get(d, 0) > 0:
                        score += 1
            if score < best_score:
                best_score = score
                best_row = list(new_row)
            if best_score == 0:
                break
    
    return best_row

def _sa_optimize(rows, k, iters, rng, deadline, phase="mixed"):
    """
    phase: 'collision' (high T, focus on 0 collisions), 'scope' (low T, focus on min scope)
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        if phase == "collision":
            # Heavily penalize collisions, scope is secondary
            return 1000 * collisions + scope
        elif phase == "scope":
            # Heavily penalize collisions, but scope is the primary driver
            return scope + 1000 * collisions
        else:
            return scope + 500 * collisions

    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    if phase == "collision":
        T = 15.0
        cooling = 0.999
        min_T = 1.0
    elif phase == "scope":
        T = 2.0
        cooling = 0.9995
        min_T = 0.05
    else:
        T = 8.0
        cooling = 0.9995
        min_T = 0.05
        
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.2:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            # Use current scope + margin as bound
            max_scope_bound = current_scope + 10
            new_row = _row_rebuild(old_row, k, diff_counts, rng, max_scope_bound)
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue
            
            _add_row_diffs(diff_counts, new_row)
            current_rows[i] = new_row
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                
        elif r < 0.5:
            # Scope compression: Left-most shift for the max element row
            # Find row with max value
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break
            
            if max_row_idx < 0:
                continue
                
            i = max_row_idx
            j = max_pos_idx
            old_val = current_rows[i][j]
            
            # Try to move this element to the smallest valid position > prev
            prev_val = current_rows[i][j-1]
            
            # Remove row diffs to test
            _remove_row_diffs(diff_counts, current_rows[i])
            
            found_new = False
            for new_val in range(prev_val + 1, old_val):
                # Check if new_val creates conflicts within the row or with other rows
                # We need to check diffs with all other elements in this row
                # And with diff_counts (other rows)
                conflict = False
                for t in range(k + 1):
                    if t == j:
                        continue
                    d = abs(new_val - current_rows[i][t])
                    if d == 0: continue
                    # Check internal conflict (diff already in this row)
                    # We removed the row, so we must check against the *other* elements in the row
                    # Actually, _remove_row_diffs removed ALL diffs of this row.
                    # So diff_counts now only has other rows.
                    # We must check if d is in diff_counts (other rows) OR if d is generated by other pairs in this row.
                    # Since we are only changing position j, the new diffs are (new_val - a_t) for t != j.
                    # The old diffs involving j are gone.
                    # We need to ensure the NEW diffs don't collide with:
                    # 1. Diffs from other rows (in diff_counts)
                    # 2. Diffs from other pairs in the SAME row (which are currently NOT in diff_counts)
                    
                    # This logic is complex. Let's simplify:
                    # Re-add the row with the new value to a temporary counter? No, too slow.
                    
                    # Alternative: Just check against diff_counts (other rows).
                    # If it passes, we accept it, but we must be careful about internal conflicts.
                    # Internal conflicts: d = new_val - current_rows[i][t].
                    # We need to ensure d is not equal to any other difference in the row.
                    # Let's precompute the set of diffs in the row excluding those involving j.
                    pass # This approach is getting messy.
                
                # Simpler approach for compression:
                # Just try random new values and let the cost function handle it.
                # But "left-most" is better.
                # Let's just try a few random values closer to prev_val.
                pass
                
            # Fallback: Random move
            new_val = rng.randint(prev_val + 1, old_val - 1) if old_val > prev_val + 1 else old_val
            if new_val == old_val:
                _add_row_diffs(diff_counts, current_rows[i])
                continue
                
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])

        else:
            # Single-element move (random)
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            if rng.random() < 0.7:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.2)))
                new_val = old_val + delta
            else:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
            
        T = max(T * cooling, min_T)
        
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 108.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    candidates = []
    
    # Generate diverse initial candidates
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Add a few randomized greedy variants
    for _ in range(3):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                # Randomized step size
                max_step = rng.randint(1, 10)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    candidates.sort(key=_scope)
    
    # Phase 1: Collision Resolution
    # Run SA with high T to get to 0 collisions
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE - 20:
            break
        sa_deadline = time.time() + 10
        iters = int((sa_deadline - time.time()) * 5000)
        iters = max(iters, 1000)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline, phase="collision")
        consider(rows)
        
    # Phase 2: Scope Optimization
    # If we have a valid solution, optimize its scope
    if _BEST is not None and time.time() < _DEADLINE - 5:
        remaining_time = _DEADLINE - time.time()
        # Use the best valid solution found so far
        best_rows_to_optimize = _BEST[1]
        
        # Split remaining time into a few passes
        passes = 3
        pass_time = remaining_time / passes
        
        for p in range(passes):
            if time.time() > _DEADLINE - 2:
                break
            sa_deadline = time.time() + pass_time
            iters = int((sa_deadline - time.time()) * 8000)
            iters = max(iters, 2000)
            s, rows = _sa_optimize(best_rows_to_optimize, k, iters, rng, sa_deadline, phase="scope")
            consider(rows)
            if _BEST is not None:
                best_rows_to_optimize = _BEST[1]
                
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
