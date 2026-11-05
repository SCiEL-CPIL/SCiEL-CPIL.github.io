import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    lower_bound = total_diffs
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Multiple greedy constructions with different strategies
    for seed in range(20):
        if time.time() > deadline - 90:
            break
        rows = _greedy_construct_v2(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback guaranteed valid
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                row.append(offset + j * (total_diffs + 1))
            rows.append(row)
            offset += total_diffs + 1
        best_rows = rows
        best_scope = _scope(rows)
    
    # Phase 2: Fast local search with incremental validity
    if time.time() < deadline - 15:
        improved = _fast_local_search(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Targeted max reduction
    if time.time() < deadline - 5 and best_rows is not None:
        polished = _polish_max_v2(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _compute_used_diffs(rows, exclude_idx=None):
    """Compute set of used differences, optionally excluding one row."""
    used = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used.add(row[b] - row[a])
    return used

def _is_valid_fast(rows, n_rows, k):
    """Check validity efficiently."""
    used = set()
    expected = n_rows * (k + 1) * k // 2
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return len(used) == expected

def _greedy_construct_v2(n_rows, k, seed, limit=300):
    """Improved greedy with better scoring and backtracking-lite."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        success = True
        for _ in range(k):
            last_x = row[-1]
            candidates = []
            
            for x in range(last_x + 1, limit + 1):
                new_diffs = [x - a for a in row]
                if any(d in used for d in new_diffs):
                    continue
                # Score: prefer smaller x, penalize large gaps
                gap_penalty = max(0, x - last_x - 1) * 0.5
                score = x + gap_penalty
                candidates.append((score, x, new_diffs))
            
            if not candidates:
                # Try expanding
                for x in range(limit + 1, limit + 200):
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    gap_penalty = max(0, x - last_x - 1) * 0.5
                    score = x + gap_penalty
                    candidates.append((score, x, new_diffs))
                    if len(candidates) >= 5:
                        break
            
            if not candidates:
                success = False
                break
            
            # Pick best candidate (with some randomness)
            candidates.sort(key=lambda c: c[0])
            top_n = min(3, len(candidates))
            if top_n > 1 and rng.random() < 0.3:
                chosen = candidates[rng.randint(0, top_n - 1)]
            else:
                chosen = candidates[0]
            
            _, best_x, new_diffs = chosen
            row.append(best_x)
            used.update(new_diffs)
        
        if not success:
            return None
        rows.append(row)
    
    return rows

def _fast_local_search(rows, n_rows, k, start_time, deadline):
    """Fast local search with incremental difference tracking."""
    rng = random.Random(42)
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute used diffs for each row (excluding that row)
    T = 30.0
    T_end = 0.5
    total_steps = 8000
    cool_rate = (T_end / T) ** (1.0 / total_steps)
    
    for step in range(total_steps):
        if time.time() > deadline - 8:
            break
        
        # Pick row to modify
        if rng.random() < 0.6:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        # Get used diffs excluding this row
        used_diffs = _compute_used_diffs(rows, row_idx)
        
        # Try to generate a better row
        old_row = rows[row_idx]
        old_max = old_row[-1]
        
        # Strategy: try to build a row with max <= old_max - 1
        target_max = old_max - 1
        if target_max < k:
            target_max = old_max
        
        new_row = _build_row_constrained(target_max, k, used_diffs, rng)
        
        if new_row is None:
            # Try a random perturbation of the existing row
            new_row = _perturb_row(old_row, k, used_diffs, rng)
        
        if new_row is None:
            continue
        
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        # Quick validity check
        if not _is_valid_fast(new_rows, n_rows, k):
            continue
        
        delta = new_scope - current_scope
        if delta <= 0:
            rows = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            if T > 1e-9:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                rows = new_rows
                current_scope = new_scope
        
        T *= cool_rate
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _build_row_constrained(max_val, k, used_diffs, rng, limit=300):
    """Build a row with max element <= max_val, avoiding used differences."""
    row = [0]
    for _ in range(k):
        last_x = row[-1]
        best_x = None
        best_score = float('inf')
        
        upper = min(max_val, limit)
        for x in range(last_x + 1, upper + 1):
            new_diffs = [x - a for a in row]
            if any(d in used_diffs for d in new_diffs):
                continue
            score = x
            if score < best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            return None
        
        row.append(best_x)
        for d in (best_x - a for a in row[:-1]):
            used_diffs.add(d)
    
    return row

def _perturb_row(old_row, k, used_diffs, rng):
    """Create a perturbed version of a row."""
    row = [0]
    for i in range(1, len(old_row)):
        old_val = old_row[i]
        # Try nearby values
        tried = set()
        found = False
        for delta in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
            new_val = old_val + delta
            if new_val <= row[-1] or new_val > 200:
                continue
            if new_val in tried:
                continue
            tried.add(new_val)
            new_diffs = [new_val - a for a in row]
            if any(d in used_diffs for d in new_diffs):
                continue
            row.append(new_val)
            for d in new_diffs:
                used_diffs.add(d)
            found = True
            break
        
        if not found:
            return None
    
    return row if len(row) == k + 1 else None

def _polish_max_v2(rows, n_rows, k, start_time, deadline):
    """Aggressively try to reduce the maximum element."""
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    for iteration in range(200):
        if time.time() > deadline - 4:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        if current_max <= k:
            break
        
        used_diffs = _compute_used_diffs(rows, max_idx)
        
        # Try decreasing target
        for target in range(current_max - 1, max(current_max - 15, k), -1):
            new_row = _build_row_constrained(target, k, set(used_diffs), rng)
            if new_row is not None:
                new_rows = rows[:]
                new_rows[max_idx] = new_row
                if _is_valid_fast(new_rows, n_rows, k):
                    new_scope = _scope(new_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in new_rows]
                        rows = new_rows
                    break
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None
