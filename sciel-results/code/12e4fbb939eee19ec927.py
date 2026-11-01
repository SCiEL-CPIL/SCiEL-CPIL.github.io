"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
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

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _sidon_row_fast(row):
    """Fast Sidon check using a boolean array."""
    n = len(row)
    max_d = row[-1] - row[0]
    if max_d <= 0:
        return True
    seen = bytearray(max_d + 1)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            if seen[d]:
                return False
            seen[d] = 1
    return True

def _diffs_conflict(row, other_diffs):
    """Check if row's diffs conflict with other_diffs."""
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            if d in other_diffs:
                return True
    return False

def _random_sidon_row(k, max_val, used_diffs, rng, prefer_low=True):
    """Try to build a random Sidon row of k+1 elements starting at 0, within max_val,
    avoiding used_diffs."""
    row = [0]
    used = set(used_diffs)
    for _ in range(k):
        if prefer_low:
            candidates = list(range(row[-1] + 1, max_val + 1))
            rng.shuffle(candidates)
        else:
            candidates = list(range(max_val, row[-1], -1))
            rng.shuffle(candidates)
        placed = False
        for x in candidates:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                if len(new_diffs) == len(row):
                    row.append(x)
                    used |= new_diffs
                    placed = True
                    break
        if not placed:
            return None
    return row

def _try_resolve_row(k, max_val, other_diffs, rng, n_attempts=50):
    """Try multiple random attempts to resolve a row."""
    for _ in range(n_attempts):
        row = _random_sidon_row(k, max_val, other_diffs, rng, prefer_low=rng.random() < 0.5)
        if row is not None:
            return row
    return None

def _local_polish(rows, n_rows, k, deadline):
    """Final polish: iteratively try to reduce max entry by local moves."""
    row_diffs = [_row_diffs(r) for r in rows]
    global_diffs = set()
    for s in row_diffs:
        global_diffs |= s
    maxes = [max(r) for r in rows]
    cur_val = max(maxes)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Focus on rows with the current max
        max_max = max(maxes)
        for ri in range(n_rows):
            if maxes[ri] < max_max:
                continue
            old_row = rows[ri]
            old_diffs = row_diffs[ri]
            other_diffs = global_diffs - old_diffs
            
            # Try to reduce the last element
            if k >= 1:
                last_val = old_row[k]
                # Try smaller values for the last element
                min_last = old_row[k-1] + 1
                for new_last in range(min_last, last_val):
                    cand_row = old_row[:]
                    cand_row[k] = new_last
                    if not _sidon_row_fast(cand_row):
                        continue
                    if _diffs_conflict(cand_row, other_diffs):
                        continue
                    # Accept
                    global_diffs -= old_diffs
                    new_diffs = _row_diffs(cand_row)
                    global_diffs |= new_diffs
                    row_diffs[ri] = new_diffs
                    rows[ri] = cand_row
                    maxes[ri] = new_last
                    cur_val = max(maxes)
                    improved = True
                    break
            # Try to reduce middle elements
            for pi in range(k - 1, 0, -1):
                if time.time() >= deadline:
                    return rows
                old_val = old_row[pi]
                min_allowed = old_row[pi - 1] + 1
                max_allowed = old_row[pi + 1] - 1
                for new_val in range(min_allowed, old_val):
                    if new_val > max_allowed:
                        continue
                    cand_row = old_row[:]
                    cand_row[pi] = new_val
                    if not _sidon_row_fast(cand_row):
                        continue
                    if _diffs_conflict(cand_row, other_diffs):
                        continue
                    global_diffs -= old_diffs
                    new_diffs = _row_diffs(cand_row)
                    global_diffs |= new_diffs
                    row_diffs[ri] = new_diffs
                    rows[ri] = cand_row
                    maxes[ri] = max(cand_row)
                    cur_val = max(maxes)
                    improved = True
                    break
                if improved:
                    break
    
    return rows

def construct(n_rows, k):
    deadline = time.time() + 110
    random.seed(42)
    rng = random.Random(42)
    
    # Start from greedy
    rows = _build_greedy(n_rows, k)
    best = [r[:] for r in rows]
    best_val = _max_entry(best)
    
    # Precompute
    row_diffs = [_row_diffs(r) for r in rows]
    global_diffs = set()
    for s in row_diffs:
        global_diffs |= s
    
    cur = [r[:] for r in rows]
    cur_val = best_val
    
    # Strategy: Iterated Local Search with row-replacement moves
    T = 50.0
    T_min = 1.0
    cool = 0.998
    
    maxes = [max(r) for r in cur]
    
    stuck = 0
    max_stuck = 300
    
    step = 0
    while time.time() < deadline - 15:  # Reserve time for polish
        step += 1
        
        # Pick a row to modify: bias toward rows with large max entries
        max_max = max(maxes)
        if rng.random() < 0.7:
            candidates = [i for i in range(n_rows) if maxes[i] >= max_max - 5]
            ri = rng.choice(candidates)
        else:
            ri = rng.randrange(n_rows)
        
        old_row = cur[ri]
        old_diffs = row_diffs[ri]
        other_diffs = global_diffs - old_diffs
        
        improved = False
        
        # Try local moves: change one position
        for attempt in range(4):
            pi = rng.randrange(1, k + 1)
            old_val = old_row[pi]
            min_allowed = old_row[pi - 1] + 1
            max_allowed = old_row[pi + 1] - 1 if pi < k else old_val
            
            search_range = min(20, old_val - min_allowed + 1)
            if search_range <= 0:
                continue
            
            candidates = list(range(min_allowed, old_val))
            rng.shuffle(candidates)
            
            for new_val in candidates:
                if new_val > max_allowed:
                    continue
                cand_row = old_row[:]
                cand_row[pi] = new_val
                
                if not _sidon_row_fast(cand_row):
                    continue
                
                if _diffs_conflict(cand_row, other_diffs):
                    continue
                
                new_max = max(cand_row)
                new_global_max = max(maxes[:ri] + [new_max] + maxes[ri+1:])
                
                if new_global_max < cur_val:
                    accept = True
                elif new_global_max == cur_val:
                    accept = rng.random() < 0.4
                else:
                    delta = new_global_max - cur_val
                    accept = rng.random() < math.exp(-delta / max(T, 0.1))
                
                if not accept:
                    continue
                
                global_diffs -= old_diffs
                new_diffs = _row_diffs(cand_row)
                global_diffs |= new_diffs
                row_diffs[ri] = new_diffs
                cur[ri] = cand_row
                maxes[ri] = new_max
                cur_val = new_global_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    stuck = 0
                else:
                    stuck += 1
                
                improved = True
                break
            
            if improved:
                break
        
        if not improved:
            # Try full row replacement with multiple attempts
            target_max = maxes[ri] - 1
            if target_max < k:
                target_max = k
            
            new_row = _try_resolve_row(k, target_max, other_diffs, rng, n_attempts=20)
            if new_row is not None:
                new_diffs = _row_diffs(new_row)
                new_max = max(new_row)
                new_global_max = max(maxes[:ri] + [new_max] + maxes[ri+1:])
                
                if new_global_max < cur_val:
                    global_diffs -= old_diffs
                    global_diffs |= new_diffs
                    row_diffs[ri] = new_diffs
                    cur[ri] = new_row
                    maxes[ri] = new_max
                    cur_val = new_global_max
                    
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r[:] for r in cur]
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1
            else:
                stuck += 1
        
        T = max(T_min, T * cool)
        
        if stuck > max_stuck:
            cur = [r[:] for r in best]
            cur_val = best_val
            row_diffs = [_row_diffs(r) for r in cur]
            global_diffs = set()
            for s in row_diffs:
                global_diffs |= s
            maxes = [max(r) for r in cur]
            T = 50.0
            stuck = 0
    
    # Final polish phase
    if time.time() < deadline - 2:
        best = _local_polish([r[:] for r in best], n_rows, k, deadline - 2)
        if _valid(best, n_rows, k):
            new_val = _max_entry(best)
            if new_val < best_val:
                best_val = new_val
    
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best

# EVOLVE-BLOCK-END
