import time
import random
import math
from collections import Counter

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

def _get_cost(rows, diff_counts, mode='mixed'):
    """
    mode='mixed': scope + 100 * collisions
    mode='scope': scope (used when collisions == 0)
    """
    collisions = sum(max(0, c - 1) for c in diff_counts.values())
    if mode == 'scope':
        return _scope(rows)
    return _scope(rows) + 100 * collisions

def _greedy_row(k, used_diffs, rng, max_scope, spread_factor=1.0):
    """
    Greedily construct a row of k+1 elements starting with 0,
    avoiding differences in used_diffs.
    """
    row = [0]
    x = 0
    for _ in range(k):
        found = False
        # Try to pick the smallest possible next value
        # Sometimes jump to avoid local minima
        min_step = 1
        max_step = int(5 * spread_factor)
        
        # Randomize the search window slightly
        search_start = x + min_step
        search_end = x + max_step + rng.randint(0, 5)
        
        for t in range(search_start, search_end + 1):
            if t > max_scope:
                break
            # Check if adding t creates any conflict
            conflict = False
            new_diffs = []
            for a in row:
                d = t - a
                if d > 0 and d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            if not conflict:
                row.append(t)
                # Update local used_diffs? No, we just check against global used_diffs.
                # The caller will update global used_diffs if the row is accepted.
                # However, for the row itself, we must ensure internal uniqueness.
                # Since we start with 0 and increase, and check against 'used_diffs' (which includes other rows),
                # we must also ensure that d is not in the *current row's* diffs.
                # The check `d in used_diffs` covers other rows.
                # We need to check `d in current_row_diffs`.
                pass
        
        # The above loop is a bit flawed because it doesn't track internal diffs properly 
        # if we just check against global `used_diffs`.
        # Let's do a proper check.
        
        row = [0]
        x = 0
        internal_diffs = set()
        for j in range(1, k + 1):
            found = False
            # Determine search range
            # Start from x+1
            # If spread is high, jump
            start = x + 1
            if j > 1 and rng.random() < 0.2:
                start = x + rng.randint(1, 3)
                
            for t in range(start, max_scope + 1):
                # Check internal diffs
                ok = True
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    if d in internal_diffs:
                        ok = False
                        break
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                
                if ok:
                    row.append(t)
                    internal_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
            
            if not found:
                return None
        return row

def _sa_optimize(rows, k, iters, rng, deadline, start_mode='mixed'):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc):
        collisions = sum(max(0, c - 1) for c in dc.values())
        scope = _scope(rows)
        # Heuristic: prioritize removing collisions first
        if collisions > 0:
            return scope + 500 * collisions
        else:
            return scope
    
    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    # Adaptive temperature
    T = 10.0
    cooling = 0.9998
    min_T = 0.01
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.2:
            # Row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            # Rebuild with some randomness
            new_row = _greedy_row(k, diff_counts, rng, current_scope + 10, spread_factor=1.0)
            if new_row is None:
                # Fallback: keep old row
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
                
        elif r < 0.4:
            # Targeted Scope Reduction: Find the row with the max element and try to shrink it
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
            
            # Try to decrease the value
            prev_val = current_rows[i][j-1]
            # Randomly choose a new value lower than old_val
            if old_val - prev_val <= 1:
                continue
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
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
                
        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            # Gaussian step
            span = hi - lo
            delta = int(rng.gauss(0, max(1, span * 0.3)))
            new_val = old_val + delta
            
            if new_val < lo or new_val > hi:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val:
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
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    
    def consider(rows):
        if not _valid(rows, k):
            return None
        return _scope(rows)
    
    rng = random.Random(123)
    
    # Generate multiple diverse initial candidates
    candidates = []
    
    # Candidate 1: Greedy with low spread
    rows1 = []
    used = set()
    for _ in range(n_rows):
        row = _greedy_row(k, used, rng, 150, spread_factor=0.5)
        if row is None:
            row = [0, 1, 2, 3, 4, 5] # Fallback
        rows1.append(row)
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j]-row[i])
    candidates.append(rows1)
    
    # Candidate 2: Greedy with high spread
    rows2 = []
    used = set()
    for _ in range(n_rows):
        row = _greedy_row(k, used, rng, 150, spread_factor=2.0)
        if row is None:
            row = [0, 5, 10, 15, 20, 25]
        rows2.append(row)
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j]-row[i])
    candidates.append(rows2)
    
    # Candidate 3: Randomized Greedy
    rows3 = []
    used = set()
    for _ in range(n_rows):
        row = _greedy_row(k, used, rng, 150, spread_factor=1.0)
        if row is None:
            row = [0, 1, 3, 6, 10, 15]
        rows3.append(row)
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j]-row[i])
    candidates.append(rows3)
    
    # Sort candidates by initial scope
    candidates.sort(key=lambda r: _scope(r))
    
    best_scope = float('inf')
    best_rows = None
    
    # Run SA on the best 2 candidates
    for idx, cand in enumerate(candidates[:2]):
        if time.time() > _DEADLINE:
            break
        
        # Allocate time based on remaining time
        remaining = _DEADLINE - time.time()
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.5
        else:
            sa_deadline = _DEADLINE - 5.0 # Leave time for polish
        
        iters = int((sa_deadline - time.time()) * 5000)
        iters = max(iters, 1000)
        
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        
        if _valid(rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
                
    # If we have a valid solution, try to polish it further
    if best_rows is not None and time.time() < _DEADLINE - 10:
        polish_deadline = _DEADLINE - 2.0
        polish_iters = int((polish_deadline - time.time()) * 8000)
        polish_iters = max(polish_iters, 500)
        
        # Run SA again with focus on scope reduction
        # We can reuse _sa_optimize, it already prioritizes scope if collisions are 0
        s, rows = _sa_optimize(best_rows, k, polish_iters, rng, polish_deadline)
        
        if _valid(rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
                
    if best_rows is not None:
        return best_rows
    
    # Fallback: return the best candidate if SA failed (should not happen)
    return candidates[0]
