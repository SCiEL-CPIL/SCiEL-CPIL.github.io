import time
import random
import sys
import math
from collections import Counter
from itertools import combinations

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k=None):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        if k is not None and len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, skip_prob=0.0, rng=None):
    """
    Greedy initialization with optional random skipping to break deterministic traps.
    """
    if rng is None:
        rng = random
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if skip_prob > 0 and rng.random() < skip_prob and x < 300:
                    skipped = False
                    for extra in range(rng.randint(1, 4)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        row.append(x)
                        used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

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
    
    for attempt in range(12):
        new_row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.4:
                start = x + rng.randint(1, 5)
            # Try to find a valid position
            candidates = []
            for t in range(start, min(max_scope + 1, start + 50)):
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
                    candidates.append((0, t, new_diffs))
                elif conflicts <= 1:
                    candidates.append((conflicts, t, new_diffs))
            
            if candidates:
                # Prefer zero-conflict, then lowest conflict, then smallest value
                candidates.sort(key=lambda x: (x[0], x[1]))
                if attempt < 4:
                    # Take the best candidate
                    _, t, new_diffs = candidates[0]
                else:
                    # Randomize among top candidates
                    top_n = min(3, len(candidates))
                    idx = rng.randint(0, top_n - 1)
                    _, t, new_diffs = candidates[idx]
                new_row.append(t)
                local_diffs.update(new_diffs)
                x = t
                found = True
            elif attempt < 6:
                # If no good candidate, try a random jump
                jump = rng.randint(1, 10)
                t = x + jump
                new_diffs = []
                for a in new_row:
                    d = t - a
                    if d > 0:
                        new_diffs.append(d)
                new_row.append(t)
                local_diffs.update(new_diffs)
                x = t
                found = True
            
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

def _sa_optimize(rows, n_rows, k, deadline, rng=None):
    """
    Optimized Simulated Annealing with Row Rebuilds, Scope Compression, and Randomized Restarts.
    Uses diff-counts for collision-aware cost.
    """
    if rng is None:
        rng = random
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    current_scope = best_scope
    
    diff_counts = _build_diff_counts(rows)
    
    def get_cost(rows, dc):
        scope = _get_scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions
    
    current_cost = get_cost(rows, diff_counts)
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.05
    
    restart_count = 0
    max_restarts = 3
    no_improve_steps = 0
    max_no_improve = 500
    
    while T > min_T and time.time() < deadline - 5:
        r = rng.random()
        
        if r < 0.25:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            new_row = _row_rebuild(old_row, k, diff_counts, rng, current_scope + 5)
            
            if new_row is not None:
                _add_row_diffs(diff_counts, new_row)
                rows[i] = new_row
                new_cost = get_cost(rows, diff_counts)
                
                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r2) for r2 in rows]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    _remove_row_diffs(diff_counts, new_row)
                    _add_row_diffs(diff_counts, old_row)
                    rows[i] = old_row
            else:
                _add_row_diffs(diff_counts, old_row)
                
        elif r < 0.50:
            # Scope compression: try to reduce the max element
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break
            
            if max_row_idx < 0:
                continue
                
            i = max_row_idx
            j = max_pos_idx
            old_val = rows[i][j]
            
            prev_val = rows[i][j-1]
            # Try several candidate values
            improved = False
            for _ in range(3):
                new_val = rng.randint(prev_val + 1, old_val - 1) if old_val > prev_val + 1 else prev_val + 1
                if new_val <= prev_val or new_val >= old_val:
                    continue
                
                _remove_row_diffs(diff_counts, rows[i])
                rows[i][j] = new_val
                _add_row_diffs(diff_counts, rows[i])
                
                new_cost = get_cost(rows, diff_counts)
                
                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r2) for r2 in rows]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                    improved = True
                    break
                else:
                    _remove_row_diffs(diff_counts, rows[i])
                    rows[i][j] = old_val
                    _add_row_diffs(diff_counts, rows[i])
            
            if not improved:
                no_improve_steps += 1
                
        else:
            # Single-element move
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]
            
            if rng.random() < 0.5:
                new_val = rng.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                new_val = rng.randint(1, current_scope + 5)
                
            if new_val == old_val:
                continue
                
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
                
            _remove_row_diffs(diff_counts, row)
            row[pos] = new_val
            _add_row_diffs(diff_counts, row)
            
            new_cost = get_cost(rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                new_scope = _get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r2) for r2 in rows]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                _remove_row_diffs(diff_counts, row)
                row[pos] = old_val
                _add_row_diffs(diff_counts, row)
                no_improve_steps += 1
                
        T *= cooling
        
        # Restart if stuck
        if no_improve_steps > max_no_improve and restart_count < max_restarts and time.time() < deadline - 15:
            restart_count += 1
            no_improve_steps = 0
            skip_prob = 0.1 * restart_count
            new_rows, new_used = _greedy_init(n_rows, k, skip_prob=skip_prob, rng=rng)
            
            if _is_valid(new_rows, k):
                new_scope = _get_scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r2) for r2 in new_rows]
                    rows = [list(r2) for r2 in new_rows]
                    diff_counts = _build_diff_counts(rows)
                    current_scope = new_scope
                    current_cost = get_cost(rows, diff_counts)
                    best_cost = current_cost
                    T = 50.0
                else:
                    # Even if not better, use as fresh start for SA
                    rows = [list(r2) for r2 in new_rows]
                    diff_counts = _build_diff_counts(rows)
                    current_scope = _get_scope(rows)
                    current_cost = get_cost(rows, diff_counts)
                    T = 30.0
            else:
                T = 20.0
    
    # Final polish: try to reduce scope by rebuilding rows that contain the max
    if time.time() < deadline - 10:
        final_scope = _get_scope(best_rows)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            max_val = final_scope
            found = False
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if best_rows[ri][pi] == max_val:
                        # Try to rebuild this row with a lower max
                        dc = _build_diff_counts(best_rows)
                        _remove_row_diffs(dc, best_rows[ri])
                        new_row = _row_rebuild(best_rows[ri], k, dc, rng, max_val - 1)
                        if new_row is not None:
                            _add_row_diffs(dc, new_row)
                            test_rows = [list(r) for r in best_rows]
                            test_rows[ri] = new_row
                            if _is_valid(test_rows, k):
                                new_scope = _get_scope(test_rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                    best_rows = test_rows
                                    final_scope = new_scope
                                    found = True
                                    break
                        break
                if found:
                    break
            if not found:
                break
    
    return best_rows

def _polish_local(rows, k, deadline, rng):
    """High-precision local polish: try small perturbations on each element of each row."""
    if not _is_valid(rows, k):
        return rows
    
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    diff_counts = _build_diff_counts(rows)
    
    def get_cost(rows, dc):
        scope = _get_scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions
    
    current_cost = get_cost(rows, diff_counts)
    
    T = 5.0
    cooling = 0.9998
    min_T = 0.01
    no_improve = 0
    max_no_improve = 200
    
    while T > min_T and time.time() < deadline - 2:
        r_idx = rng.randint(0, len(rows) - 1)
        row = rows[r_idx]
        pos = rng.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Try small perturbations: -2, -1, +1, +2
        deltas = [-2, -1, 1, 2]
        rng.shuffle(deltas)
        
        improved = False
        for delta in deltas:
            new_val = old_val + delta
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            _remove_row_diffs(diff_counts, row)
            row[pos] = new_val
            _add_row_diffs(diff_counts, row)
            
            new_cost = get_cost(rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                new_scope = _get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r2) for r2 in rows]
                    no_improve = 0
                else:
                    no_improve += 1
                improved = True
                break
            else:
                _remove_row_diffs(diff_counts, row)
                row[pos] = old_val
                _add_row_diffs(diff_counts, row)
        
        if not improved:
            no_improve += 1
        
        T *= cooling
        
        if no_improve > max_no_improve:
            break
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 148.0
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Pure greedy
    rows1, _ = _greedy_init(n_rows, k, skip_prob=0.0, rng=rng)
    if _is_valid(rows1, k):
        scope1 = _get_scope(rows1)
        if scope1 < best_scope:
            best_scope = scope1
            best_rows = [list(r) for r in rows1]
    
    # Strategy 2: Randomized greedy with small skip probability
    for attempt in range(5):
        if time.time() > deadline - 100:
            break
        rows2, _ = _greedy_init(n_rows, k, skip_prob=0.03 + 0.04 * attempt, rng=rng)
        if _is_valid(rows2, k):
            scope2 = _get_scope(rows2)
            if scope2 < best_scope:
                best_scope = scope2
                best_rows = [list(r) for r in rows2]
    
    if best_rows is None:
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0, rng=rng)
        return rows
    
    if not _is_valid(best_rows, k):
        return best_rows
        
    # Run SA optimization
    rows = _sa_optimize(best_rows, n_rows, k, deadline, rng)
    
    if not _is_valid(rows, k):
        return best_rows
    
    # Final check: if SA result is valid, use it
    if _is_valid(rows, k):
        final_scope = _get_scope(rows)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = rows
    
    # High-precision local polish
    if time.time() < deadline - 5:
        polished = _polish_local(best_rows, k, deadline, rng)
        if _is_valid(polished, k):
            polished_scope = _get_scope(polished)
            if polished_scope < best_scope:
                best_scope = polished_scope
                best_rows = polished
    
    return best_rows
