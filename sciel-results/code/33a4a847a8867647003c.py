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
    """Build a Counter of all positive differences across all rows."""
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
    """
    Rebuild a single row to minimize its contribution to collisions.
    Uses a greedy approach with forward checking against diff_counts.
    Returns a new row or None if it can't build one.
    """
    best_row = None
    best_score = float('inf')
    
    for attempt in range(8):
        r = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.3:
                start = x + rng.randint(1, 3)
            for t in range(start, max_scope + 1):
                conflicts = 0
                new_diffs = []
                for a in r:
                    d = t - a
                    if d <= 0:
                        continue
                    new_diffs.append(d)
                    if diff_counts[d] > 0:
                        conflicts += 1
                if conflicts == 0:
                    r.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
                elif attempt < 3 and conflicts <= 1 and rng.random() < 0.15:
                    r.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok and len(r) == k + 1:
            score = 0
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if diff_counts[d] > 0:
                        score += 1
            if score < best_score:
                best_score = score
                best_row = list(r)
            if best_score == 0:
                break
    
    return best_row

def _compress_rows(rows, k, diff_counts, rng):
    """Try to compress each row by shifting elements left where possible."""
    improved = True
    while improved:
        improved = False
        for i in range(len(rows)):
            row = rows[i]
            for j in range(k, 0, -1):
                # Try to decrease row[j] by 1
                new_val = row[j] - 1
                if new_val <= row[j-1]:
                    continue
                # Check if new_val creates any duplicate differences
                ok = True
                for t in range(k + 1):
                    if t == j:
                        continue
                    if t < j:
                        d = new_val - row[t]
                    else:
                        d = row[t] - new_val
                    if d <= 0:
                        ok = False
                        break
                    if diff_counts[d] >= 1:
                        # Would create a duplicate
                        ok = False
                        break
                if ok:
                    # Remove old diffs for position j
                    for t in range(k + 1):
                        if t == j:
                            continue
                        d = abs(row[j] - row[t])
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    row[j] = new_val
                    # Add new diffs
                    for t in range(k + 1):
                        if t == j:
                            continue
                        d = abs(new_val - row[t])
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    improved = True
                    break
    return rows

def _sa_optimize(rows, k, iters, rng, deadline):
    """
    Simulated Annealing with incremental diff counting and row-rebuild moves.
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    
    # Build initial diff counts
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(dc):
        scope = max(a for r in current_rows for a in r[1:])
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions

    current_cost = get_cost(diff_counts)
    best_cost = current_cost
    
    T = 8.0
    cooling = 0.9995
    min_T = 0.05
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.15:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            max_scope = max(a for row in current_rows for a in row[1:])
            new_row = _row_rebuild(old_row, k, diff_counts, rng, max_scope + 5)
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue
            
            _add_row_diffs(diff_counts, new_row)
            
            current_rows[i] = new_row
            new_cost = get_cost(diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                
        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else max(a for row in current_rows for a in row[1:]) + 10
            
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
            
            # Compute old diffs for this row element
            old_diffs = []
            for t in range(k + 1):
                if t == j:
                    continue
                d = abs(old_val - current_rows[i][t])
                if d > 0:
                    old_diffs.append(d)
            
            # Remove old diffs
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            current_rows[i][j] = new_val
            
            # Add new diffs
            new_diffs = []
            for t in range(k + 1):
                if t == j:
                    continue
                d = abs(new_val - current_rows[i][t])
                if d > 0:
                    new_diffs.append(d)
            
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_cost = get_cost(diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                current_rows[i][j] = old_val
            
        T = max(T * cooling, min_T)
    
    # Final compression pass
    _compress_rows(best_rows, k, diff_counts, rng)
    
    best_scope = _scope(best_rows)
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
    
    # 1. Generate several initial candidates using different greedy strategies
    candidates = []
    
    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    # Spread greedy
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
        
    # Sort candidates by scope
    candidates.sort(key=_scope)
    
    # 2. Run SA on the top 2 candidates with more iterations each
    for idx, cand in enumerate(candidates[:2]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        # Allocate time: first SA gets 45% of remaining, second gets rest
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.45
        else:
            sa_deadline = _DEADLINE
        iters = int((sa_deadline - time.time()) * 12000)
        iters = max(iters, 2000)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        consider(rows)
        
    # 3. If time remains, run SA on the best result found so far
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        extra_iters = int(remaining_time * 12000)
        if extra_iters > 500:
            s, rows = _sa_optimize(_BEST[1], k, extra_iters, rng, _DEADLINE)
            consider(rows)
            
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
