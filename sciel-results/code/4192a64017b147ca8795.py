import time
import random
import math

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

def _diffs_for_row(row):
    """Return set of all pairwise differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return list of all differences across all rows."""
    diffs = []
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.append(r[j] - r[i])
    return diffs

def _greedy_random(n_rows, k, rng, aggressiveness=0.3):
    """
    Randomized greedy: for each position, consider a window of candidates
    and pick randomly from valid ones, biased toward smaller values.
    aggressiveness controls how far we look ahead.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            # Try candidates from last+1 upward, but with a window
            best_candidates = []
            x = last + 1
            window_size = int(5 + aggressiveness * 20)  # Look ahead a bit
            for _ in range(window_size):
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    best_candidates.append((x, new_diffs))
                    if len(best_candidates) >= 3:  # Enough candidates
                        break
                x += 1
            
            if not best_candidates:
                # Fallback: keep searching
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        best_candidates.append((x, new_diffs))
                        break
                    x += 1
            
            # Pick from candidates with bias toward smaller
            # Use weighted random: smaller values get higher weight
            weights = [1.0 / (c[0] - last + 1) ** 1.5 for c in best_candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = best_candidates[0]
            for c, w in zip(best_candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _local_search(rows, k, deadline, rng):
    """
    Local search to minimize scope while maintaining validity.
    Moves: change a single element in a row to a nearby value.
    """
    n_rows = len(rows)
    rows = [list(r) for r in rows]
    
    # Build initial diff set
    def get_diff_set():
        s = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    s.add(r[j] - r[i])
        return s
    
    diff_set = get_diff_set()
    scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = scope
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline:
        # Pick a random element to move
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)  # Don't move the first element (must be 0)
        
        old_val = rows[i][j]
        prev_val = rows[i][j - 1]
        next_val = rows[i][j + 1] if j < k else scope + 100
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        # Propose new value
        if rng.random() < 0.6:
            # Small perturbation
            span = hi - lo
            delta = int(rng.gauss(0, max(1, span * 0.15)))
            new_val = old_val + delta
        else:
            new_val = rng.randint(lo, hi)
        
        new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        # Compute cost change
        # Remove old diffs involving rows[i][j]
        old_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(rows[i][t] - old_val)
                old_diffs.add(d)
        
        # Check new diffs
        new_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(rows[i][t] - new_val)
                new_diffs.add(d)
        
        # Check if new_diffs collide with existing diff_set (excluding old_diffs)
        # Remove old_diffs from diff_set temporarily
        remaining = diff_set - old_diffs
        collisions = len(new_diffs & remaining)
        
        # New scope
        new_scope = max(scope, new_val)
        if old_val == scope and new_val < old_val:
            # Recalculate scope
            max_s = 0
            for r in rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            # Also check if any other row has the old scope value
            new_scope = max_s
        
        old_cost = scope + 1000 * 0  # No collisions currently (valid)
        new_cost = new_scope + 1000 * collisions
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0 or (T > min_T and rng.random() < math.exp(-delta_cost / T)):
            # Accept
            # Update diff_set
            diff_set = remaining | new_diffs
            rows[i][j] = new_val
            scope = new_scope
            
            if collisions == 0 and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
        else:
            # Reject, no state change needed
            pass
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + 110.0
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts
    best_rows = None
    best_scope = float('inf')
    
    n_restarts = 50
    deadline_phase1 = time.time() + 30.0
    
    for restart in range(n_restarts):
        if time.time() > deadline_phase1:
            break
        
        # Vary aggressiveness
        aggr = 0.1 + 0.8 * (restart % 10) / 10.0
        rows = _greedy_random(n_rows, k, rng, aggressiveness=aggr)
        
        if _valid(rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
        
        # Also try with different row orderings occasionally
        if restart % 5 == 0:
            # Shuffle row construction order conceptually by varying rng seed
            pass
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            for j in range(1, k + 1):
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        last = x
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
    
    # Phase 2: Local search to improve
    remaining = _DEADLINE - time.time()
    if remaining > 10 and best_rows is not None:
        # Run local search
        ls_scope, ls_rows = _local_search(best_rows, k, time.time() + remaining * 0.8, rng)
        
        if _valid(ls_rows, k):
            ls_scope = _scope(ls_rows)
            if ls_scope < best_scope:
                best_scope = ls_scope
                best_rows = ls_rows
    
    return best_rows
