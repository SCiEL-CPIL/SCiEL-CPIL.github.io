import time
import random
import math
from collections import Counter

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

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

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

def _get_cost(rows, diff_counts):
    """Compute cost: scope + penalty for collisions."""
    scope = _scope(rows)
    collisions = sum(max(0, c - 1) for c in diff_counts.values())
    return scope + 500 * collisions

def _fast_greedy(k, n_rows, rng=None, spread=False):
    """Bitmask-based greedy construction for fast initialization."""
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

def _csp_build_row(k, used_diffs, max_val, rng, time_limit):
    """
    Build a single row [0, a1, ..., ak] using CSP backtracking with forward checking.
    used_diffs: set of already-used differences (globally).
    Returns the row or None if not found within time_limit seconds.
    """
    start = time.time()
    best_row = None
    best_score = float('inf')
    
    for attempt in range(30):
        if time.time() - start > time_limit:
            break
        
        row = [0]
        local_used = set()
        ok = True
        
        for j in range(1, k + 1):
            found = False
            min_val = row[-1] + 1
            max_val_candidate = max_val - (k - j)
            
            if max_val_candidate < min_val:
                ok = False
                break
            
            if attempt < 5:
                candidates = range(min_val, max_val_candidate + 1)
            else:
                start_offset = rng.randint(0, max(0, max_val_candidate - min_val))
                candidates = list(range(min_val + start_offset, max_val_candidate + 1))
                if min_val + start_offset > min_val:
                    candidates += list(range(min_val, min_val + start_offset))
                rng.shuffle(candidates)
            
            for v in candidates:
                valid = True
                new_diffs = []
                for a in row:
                    d = v - a
                    if d in used_diffs or d in local_used:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(v)
                    local_used.update(new_diffs)
                    found = True
                    break
            
            if not found:
                ok = False
                break
        
        if ok:
            if _valid([row], k):
                score = row[-1]
                if score < best_score:
                    best_score = score
                    best_row = list(row)
                    if best_score <= k * (k + 1) // 2:
                        break
    
    return best_row

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    """
    Simulated annealing with efficient cost tracking.
    Key insight: track cost incrementally rather than recomputing from scratch.
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    current_cost = _get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    if polish:
        T = 0.5
        cooling = 0.9999
        min_T = 0.01
    else:
        T = 8.0
        cooling = 0.9995
        min_T = 0.05
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        r = rng.random()
        
        if r < 0.3:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            
            # Remove old row's diffs
            old_diffs = _row_diffs(old_row)
            old_scope = current_scope
            _remove_row_diffs(diff_counts, old_row)
            
            used_diffs = set(diff_counts.keys())
            
            # Try to rebuild with tight scope
            new_row = _csp_build_row(k, used_diffs, current_scope, rng, 0.03)
            if new_row is None:
                new_row = _csp_build_row(k, used_diffs, current_scope + 5, rng, 0.03)
            
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue
            
            # Add new row's diffs
            new_diffs = _row_diffs(new_row)
            _add_row_diffs(diff_counts, new_row)
            current_rows[i] = new_row
            
            new_scope = _scope(current_rows)
            new_cost = _get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                current_scope = old_scope
                
        elif r < 0.5:
            # Scope compression: target the max element
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
            
            if old_val <= prev_val + 1:
                continue
                
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_scope = _scope(current_rows)
            new_cost = _get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
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
            
            lo = current_rows[i][j-1] + 1
            hi = current_rows[i][j+1] - 1 if j < k else current_scope
            
            if hi <= lo:
                continue
            
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_scope = _scope(current_rows)
            new_cost = _get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _local_polish(rows, k, deadline, rng):
    """Aggressive local search to compress scope with fast convergence."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    best_scope = _scope(current_rows)
    
    diff_counts = _build_diff_counts(current_rows)
    current_cost = _get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    T = 0.3
    cooling = 0.99995
    min_T = 0.005
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        old_val = current_rows[i][j]
        
        lo = current_rows[i][j-1] + 1
        hi = current_rows[i][j+1] - 1 if j < k else best_scope
        
        if hi <= lo:
            continue
        
        # Bias toward decreasing values
        if old_val > lo:
            new_val = rng.randint(lo, old_val - 1)
        else:
            new_val = rng.randint(old_val + 1, hi)
        
        if new_val == old_val:
            continue
        
        _remove_row_diffs(diff_counts, current_rows[i])
        current_rows[i][j] = new_val
        _add_row_diffs(diff_counts, current_rows[i])
        
        new_scope = _scope(current_rows)
        new_cost = _get_cost(current_rows, diff_counts)
        
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = old_val
            _add_row_diffs(diff_counts, current_rows[i])
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions via fast greedy
    init_deadline = start_time + 5.0
    candidates = []
    
    for trial in range(25):
        if time.time() > init_deadline:
            break
        spread = rng.random() < 0.3
        rows = _fast_greedy(k, n_rows, rng, spread=spread)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if not candidates:
        # Last resort: simple sequential
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k+1):
                x = len(used) + j + 1
                while any((x - row[i]) in used for i in range(len(row))):
                    x += 1
                for i in range(len(row)):
                    used.add(x - row[i])
                row.append(x)
            rows.append(row)
        return rows
    
    # Phase 1.5: CSP-based reconstruction to tighten initial solution
    if best_rows and time.time() < deadline - 25:
        csp_rows = [list(r) for r in best_rows]
        diff_counts = _build_diff_counts(csp_rows)
        csp_scope = _scope(csp_rows)
        
        improved = True
        while improved and time.time() < deadline - 25:
            improved = False
            for i in range(n_rows):
                if time.time() > deadline - 25:
                    break
                old_row = list(csp_rows[i])
                _remove_row_diffs(diff_counts, old_row)
                used_diffs = set(diff_counts.keys())
                
                new_row = _csp_build_row(k, used_diffs, csp_scope, rng, 0.08)
                
                if new_row:
                    _add_row_diffs(diff_counts, new_row)
                    csp_rows[i] = new_row
                    new_scope = _scope(csp_rows)
                    if new_scope < csp_scope:
                        csp_scope = new_scope
                        improved = True
                else:
                    _add_row_diffs(diff_counts, old_row)
                    csp_rows[i] = old_row
        
        if _valid(csp_rows, k):
            actual = _scope(csp_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = csp_rows
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: SA optimization on top candidates
    remaining = deadline - time.time()
    top_n = min(4, len(candidates))
    time_per = remaining / max(1, top_n)
    
    for idx in range(top_n):
        if time.time() > deadline - 12:
            break
        s, rows = candidates[idx]
        sa_deadline = time.time() + time_per * 0.8
        sa_scope, sa_rows = _sa_optimize(rows, k, 8000, rng, sa_deadline)
        
        if sa_rows is not None and _valid(sa_rows, k):
            actual = _scope(sa_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = sa_rows
                candidates.append((actual, sa_rows))
                candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Local polish on the best solution
    if best_rows is not None and time.time() < deadline - 3:
        polish_deadline = time.time() + min(10, deadline - time.time() - 3)
        polish_scope, polish_rows = _local_polish(best_rows, k, polish_deadline, rng)
        
        if polish_rows is not None and _valid(polish_rows, k):
            actual = _scope(polish_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = polish_rows
    
    return best_rows
