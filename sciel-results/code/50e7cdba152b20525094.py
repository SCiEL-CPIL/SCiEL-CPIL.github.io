import time
import random
import math

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

def _build_used(rows, k):
    """Build a set of all differences used across all rows."""
    used = set()
    for r in rows:
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                used.add(r[j] - r[i])
    return used

def _row_backtrack(k, used, max_scope, rng, deadline, spread_bias=0.0):
    """
    Backtracking search for a single row of k+1 increasing integers starting at 0,
    such that no difference appears in `used`.
    Returns the row (list) or None.
    Uses randomized ordering and pruning.
    """
    row = [0]
    
    def bt(pos, last_val):
        if time.time() > deadline:
            return None
        if pos == k + 1:
            return list(row)
        
        # Pruning: even minimum spacing won't fit
        min_remaining = pos - 1  # at least 1 per remaining element
        if last_val + min_remaining > max_scope:
            return None
        
        # Determine candidate range
        lo = last_val + 1
        hi = max_scope
        
        # For the last position, we can be tighter
        if pos == k:
            hi = max_scope
        
        # Generate candidates with some randomization
        candidates = list(range(lo, hi + 1))
        
        # Apply spread bias: prefer larger gaps
        if spread_bias > 0 and rng.random() < spread_bias:
            # Weight toward later candidates
            mid = (lo + hi) // 2
            candidates = candidates[len(candidates)//2:] + candidates[:len(candidates)//2]
        
        # Shuffle for diversity
        if rng.random() < 0.5:
            rng.shuffle(candidates)
        
        for t in candidates:
            # Check if t - a is in used for all a in row
            ok = True
            for a in row:
                d = t - a
                if d in used:
                    ok = False
                    break
            if not ok:
                continue
            
            row.append(t)
            # Add new diffs temporarily
            new_diffs = set()
            for a in row[:-1]:
                new_diffs.add(t - a)
            used.update(new_diffs)
            
            result = bt(pos + 1, t)
            
            used.difference_update(new_diffs)
            row.pop()
            
            if result is not None:
                return result
            if time.time() > deadline:
                return None
        
        return None
    
    return bt(1, 0)

def _constructive_search(n_rows, k, deadline, rng, initial_rows=None):
    """
    Constructive backtracking: build rows one by one.
    For the last few rows, do exhaustive search with restarts.
    """
    global _BEST
    
    if initial_rows is not None:
        rows = [list(r) for r in initial_rows]
        used = _build_used(rows, k)
        # Try to improve by rebuilding individual rows
        for i in range(n_rows - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            old_row = rows[i]
            # Remove row's diffs
            for a in old_row:
                for b in old_row:
                    if b > a:
                        used.discard(b - a)
            
            # Try to find a better row (smaller max)
            old_max = old_row[-1]
            new_max = old_max - 1
            found = None
            attempts = 0
            while new_max >= k and attempts < 50:
                if time.time() > deadline - 2:
                    break
                found = _row_backtrack(k, used, new_max, rng, deadline, spread_bias=0.3)
                if found is not None:
                    break
                new_max -= 1
                attempts += 1
            
            if found is not None:
                rows[i] = found
                for a in found:
                    for b in found:
                        if b > a:
                            used.add(b - a)
            else:
                # Restore original
                rows[i] = old_row
                for a in old_row:
                    for b in old_row:
                        if b > a:
                            used.add(b - a)
        
        if _valid(rows, k):
            s = _scope(rows)
            if _BEST is None or s < _BEST[0]:
                _BEST = (s, [list(r) for r in rows])
        return
    
    # Fresh construction
    rows = []
    used = set()
    
    # Build first n_rows-2 rows with greedy + backtracking
    for i in range(n_rows - 2):
        if time.time() > deadline - 5:
            # Fill remaining with greedy
            for _ in range(n_rows - len(rows)):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    ok = True
                    for a in row:
                        if x - a in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                rows.append(row)
            break
        
        # Try backtracking with increasing scope
        found = None
        min_scope = k * (k + 1) // 2  # minimum possible
        max_scope = 200  # upper bound for search
        
        # First try greedy to get a baseline
        greedy_row = [0]
        x = 0
        while len(greedy_row) < k + 1:
            x += 1
            ok = True
            for a in greedy_row:
                if x - a in used:
                    ok = False
                    break
            if ok:
                greedy_row.append(x)
                for a in greedy_row[:-1]:
                    used.add(x - a)
        
        # Now try to find a row with smaller max via backtracking
        best_row = greedy_row
        best_max = greedy_row[-1]
        
        # Try to reduce scope
        for target_max in range(best_max - 1, k, -1):
            if time.time() > deadline - 5:
                break
            found = _row_backtrack(k, used, target_max, rng, deadline, spread_bias=0.0)
            if found is not None:
                best_row = found
                best_max = target_max
                break
        
        rows.append(best_row)
        for a in best_row:
            for b in best_row:
                if b > a:
                    used.add(b - a)
        
        # Check if we have a valid partial solution worth recording
        if len(rows) == n_rows:
            if _valid(rows, k):
                s = _scope(rows)
                if _BEST is None or s < _BEST[0]:
                    _BEST = (s, [list(r) for r in rows])
    
    # Build the last 2 rows with more thorough search
    for i in range(len(rows), n_rows):
        if time.time() > deadline - 2:
            # Greedy fill
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                ok = True
                for a in row:
                    if x - a in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
            continue
        
        # For the last row(s), do more extensive search
        found = None
        # Try from small scope upward
        for target_max in range(k, 300):
            if time.time() > deadline - 1:
                break
            found = _row_backtrack(k, used, target_max, rng, deadline, spread_bias=0.2)
            if found is not None:
                break
        
        if found is None:
            # Fallback: greedy
            found = [0]
            x = 0
            while len(found) < k + 1:
                x += 1
                ok = True
                for a in found:
                    if x - a in used:
                        ok = False
                        break
                if ok:
                    found.append(x)
                    for a in found[:-1]:
                        used.add(x - a)
        
        rows.append(found)
        for a in found:
            for b in found:
                if b > a:
                    used.add(b - a)
    
    if _valid(rows, k):
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])

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

def _greedy_with_random(k, n_rows, rng):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            found = False
            max_step = rng.randint(1, 10)
            for t in range(x + 1, x + 1 + max_step * 3):
                ok = True
                for a in row:
                    if t - a in used:
                        ok = False
                        break
                if ok:
                    row.append(t)
                    for a in row[:-1]:
                        used.add(t - a)
                    x = t
                    found = True
                    break
            if not found:
                t = x + 1
                while True:
                    ok = True
                    for a in row:
                        if t - a in used:
                            ok = False
                            break
                    if ok:
                        break
                    t += 1
                row.append(t)
                for a in row[:-1]:
                    used.add(t - a)
                x = t
        rows.append(row)
    return rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Phase 1: Generate diverse initial solutions via different greedy strategies
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    for _ in range(6):
        candidates.append(_greedy_with_random(k, n_rows, rng))
    
    candidates.sort(key=_scope)
    
    # Phase 2: Constructive backtracking from scratch
    # Multiple restarts with different random seeds
    for restart in range(8):
        if time.time() > _DEADLINE - 10:
            break
        rng2 = random.Random(42 + restart * 7919)
        _constructive_search(n_rows, k, _DEADLINE - 5, rng2)
    
    # Phase 3: Use constructive search to improve best candidates
    if _BEST is not None:
        for restart in range(6):
            if time.time() > _DEADLINE - 5:
                break
            rng3 = random.Random(10000 + restart * 104729)
            # Try to improve the best solution by rebuilding rows
            _constructive_search(n_rows, k, _DEADLINE - 2, rng3, initial_rows=_BEST[1])
    
    # Phase 4: If we have a best, do targeted row-by-row backtracking improvement
    if _BEST is not None and time.time() < _DEADLINE - 3:
        best_rows = [list(r) for r in _BEST[1]]
        used = _build_used(best_rows, k)
        current_scope = _scope(best_rows)
        
        # Try to improve each row individually
        for i in range(n_rows):
            if time.time() > _DEADLINE - 3:
                break
            old_row = best_rows[i]
            old_max = old_row[-1]
            
            # Remove old row's diffs
            for a in old_row:
                for b in old_row:
                    if b > a:
                        used.discard(b - a)
            
            # Try to find a row with smaller max
            new_row = None
            for target in range(old_max - 1, k, -1):
                if time.time() > _DEADLINE - 3:
                    break
                new_row = _row_backtrack(k, used, target, rng, _DEADLINE - 3, spread_bias=0.3)
                if new_row is not None:
                    break
            
            if new_row is not None:
                best_rows[i] = new_row
                for a in new_row:
                    for b in new_row:
                        if b > a:
                            used.add(b - a)
                new_scope = _scope(best_rows)
                if _valid(best_rows, k) and new_scope < _BEST[0]:
                    _BEST = (new_scope, [list(r) for r in best_rows])
            else:
                # Restore
                best_rows[i] = old_row
                for a in old_row:
                    for b in old_row:
                        if b > a:
                            used.add(b - a)
    
    if _BEST is not None:
        return _BEST[1]
    
    # Fallback: return best candidate
    return candidates[0]
