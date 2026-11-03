import time
import random
import math
from collections import Counter

_DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _sum_entries(rows):
    return sum(sum(r) for r in rows)

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

def _row_diffs(row):
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

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

def _build_state(rows):
    """Build diff_counts dict and conflict_count from rows."""
    diff_counts = {}
    conflict_count = 0
    for row in rows:
        for d in _row_diffs(row):
            old_cnt = diff_counts.get(d, 0)
            if old_cnt >= 1:
                conflict_count += 1
            diff_counts[d] = old_cnt + 1
    return diff_counts, conflict_count

def _remove_row(row, diff_counts, conflict_count):
    """Remove a row's diffs, return updated conflict_count."""
    for d in _row_diffs(row):
        cnt = diff_counts[d]
        if cnt == 2:
            conflict_count -= 1
        elif cnt > 2:
            conflict_count -= 1
        if cnt == 1:
            del diff_counts[d]
        else:
            diff_counts[d] = cnt - 1
    return conflict_count

def _add_row(row, diff_counts, conflict_count):
    """Add a row's diffs, return updated conflict_count."""
    for d in _row_diffs(row):
        cnt = diff_counts.get(d, 0)
        if cnt >= 1:
            conflict_count += 1
        diff_counts[d] = cnt + 1
    return conflict_count

def _row_rebuild(row, k, diff_counts, rng, max_scope):
    """Rebuild a row greedily avoiding existing diffs where possible."""
    best_row = None
    best_score = float('inf')
    forbidden = set(diff_counts.keys())
    
    for attempt in range(10):
        row = [0]
        x = 0
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.5:
                start = x + rng.randint(1, 3)
            for t in range(start, max_scope + 1):
                conflicts = 0
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    new_diffs.append(d)
                    if d in forbidden:
                        conflicts += 1
                if conflicts == 0:
                    row.append(t)
                    x = t
                    found = True
                    break
                elif attempt < 4 and conflicts <= 1 and rng.random() < 0.2:
                    row.append(t)
                    x = t
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok and len(row) == k + 1:
            score = 0
            for d in _row_diffs(row):
                if d in forbidden:
                    score += 1
            if score < best_score:
                best_score = score
                best_row = list(row)
            if best_score == 0:
                break
    return best_row

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    
    diff_counts, conflict_count = _build_state(current_rows)
    current_scope = _scope(current_rows)
    current_sum = _sum_entries(current_rows)
    
    PENALTY = 500
    SUM_WEIGHT = 0.05
    
    def cost(scope, conflicts, sm):
        return scope + PENALTY * conflicts + SUM_WEIGHT * sm
    
    current_cost = cost(current_scope, conflict_count, current_sum)
    best_scope = current_scope
    best_conflicts = conflict_count
    best_sum = current_sum
    best_cost = current_cost
    best_rows = [list(r) for r in current_rows]
    
    if polish:
        T = 0.5
        cooling = 0.9999
        min_T = 0.01
    else:
        T = 8.0
        cooling = 0.9995
        min_T = 0.05
    
    for it in range(iters):
        if _elapsed() > deadline:
            break
        
        r = rng.random()
        
        if r < 0.15:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            conflict_count = _remove_row(old_row, diff_counts, conflict_count)
            
            new_row = _row_rebuild(old_row, k, diff_counts, rng, current_scope + 5)
            if new_row is None:
                conflict_count = _add_row(old_row, diff_counts, conflict_count)
                continue
            
            conflict_count = _add_row(new_row, diff_counts, conflict_count)
            current_rows[i] = new_row
            new_scope = _scope(current_rows)
            new_sum = _sum_entries(current_rows)
            new_cost = cost(new_scope, conflict_count, new_sum)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_sum = new_sum
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_conflicts = conflict_count
                    best_sum = new_sum
                    best_rows = [list(r) for r in current_rows]
            else:
                conflict_count = _remove_row(new_row, diff_counts, conflict_count)
                conflict_count = _add_row(old_row, diff_counts, conflict_count)
                current_rows[i] = old_row
                
        elif r < 0.35:
            # Scope compression: target the maximum entry
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
            prev_val = current_rows[i][j-1]
            
            # Try to compress: pick a new value between prev+1 and old-1
            if old_val - prev_val <= 1:
                continue
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
            if new_val <= prev_val or new_val >= old_val:
                continue
            
            conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
            current_rows[i][j] = new_val
            conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
            
            new_scope = _scope(current_rows)
            new_sum = _sum_entries(current_rows)
            new_cost = cost(new_scope, conflict_count, new_sum)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_sum = new_sum
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_conflicts = conflict_count
                    best_sum = new_sum
                    best_rows = [list(r) for r in current_rows]
            else:
                conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
                current_rows[i][j] = old_val
                conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
                
        else:
            # Single-element move with compression bias
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            # Bias toward compression (lower values)
            if rng.random() < 0.6:
                span = old_val - lo
                if span <= 0:
                    continue
                new_val = lo + int(rng.random() * span)
            else:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.25)))
                new_val = old_val + delta
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
            current_rows[i][j] = new_val
            conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
            
            new_scope = _scope(current_rows)
            new_sum = _sum_entries(current_rows)
            new_cost = cost(new_scope, conflict_count, new_sum)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_sum = new_sum
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_conflicts = conflict_count
                    best_sum = new_sum
                    best_rows = [list(r) for r in current_rows]
            else:
                conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
                current_rows[i][j] = old_val
                conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
            
        T = max(T * cooling, min_T)
        
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = _START + 143.0  # 7s safety margin
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
    
    rng = random.Random(42)
    
    # Generate initial candidates
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy
    for _ in range(4):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 8)
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
    
    # Run SA on top candidates
    for idx, cand in enumerate(candidates[:2]):
        if _elapsed() > _DEADLINE:
            break
        remaining = _DEADLINE - _elapsed()
        if idx == 0:
            sa_deadline = _START + _elapsed() + remaining * 0.45
        else:
            sa_deadline = _START + _DEADLINE
        iters = int((sa_deadline - _elapsed()) * 10000)
        iters = max(iters, 2000)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        consider(rows)
        
    # Polish phase
    if _BEST is not None and _elapsed() < _DEADLINE:
        remaining_time = _DEADLINE - _elapsed()
        extra_iters = int(remaining_time * 10000)
        if extra_iters > 500:
            s, rows = _sa_optimize(_BEST[1], k, extra_iters, rng, _START + _DEADLINE, polish=True)
            consider(rows)
            
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
