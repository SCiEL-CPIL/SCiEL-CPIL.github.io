import time
import random
import math
import sys

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

def _row_diffs(row):
    """Return set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

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
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _counting_bound(n_rows, k):
    """Simple counting bound: need n_rows * C(k+1,2) distinct differences, each <= M."""
    num_diffs = n_rows * (k * (k + 1) // 2)
    # Need M such that M >= num_diffs (since diffs are 1..M)
    return num_diffs

def _backtrack_construct(n_rows, k, M, deadline):
    """
    Try to construct n_rows rows of k+1 integers in [0, M] with all pairwise
    differences distinct globally. Uses backtracking with constraint propagation.
    Returns (success, rows) or (False, None).
    """
    rows = [None] * n_rows
    used_diffs = set()
    
    def backtrack(row_idx, pos, current_row, max_so_far):
        if time.time() > deadline:
            return False, None
        
        if row_idx == n_rows:
            return True, [r[:] for r in rows]
        
        if pos == k + 1:
            # Row complete
            rows[row_idx] = current_row[:]
            return backtrack(row_idx + 1, 0, [0], max_so_far)
        
        if pos == 0:
            # Start new row with 0
            return backtrack(row_idx, 1, [0], max_so_far)
        
        # Determine range for current position
        prev_val = current_row[pos - 1]
        min_val = prev_val + 1
        max_val = M
        
        # Pruning: if we've already used all values up to some point, 
        # and remaining positions need strictly increasing values
        remaining = k + 1 - pos
        if min_val + remaining - 1 > M:
            return False, None
        
        # Try values from min_val to max_val
        # Heuristic: try smaller values first to keep max small
        for val in range(min_val, max_val + 1):
            # Check if adding this value creates duplicate differences within this row
            new_diffs = set()
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in new_diffs or d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Check if we can still complete the row
            # Remaining positions need to fit in [val+1, M]
            rem = k + 1 - pos - 1
            if rem > 0 and val + rem > M:
                continue
            
            # Tentatively add
            current_row.append(val)
            old_used = used_diffs.copy()
            used_diffs.update(new_diffs)
            
            success, result = backtrack(row_idx, pos + 1, current_row, max(max_so_far, val))
            
            if success:
                return True, result
            
            # Backtrack
            current_row.pop()
            used_diffs.clear()
            used_diffs.update(old_used)
        
        return False, None
    
    return backtrack(0, 0, [], 0)

def _backtrack_with_restarts(n_rows, k, M, deadline, max_restarts=50):
    """
    Try backtracking with random restarts and different value orderings.
    """
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
        
        rows = [None] * n_rows
        used_diffs = set()
        
        def backtrack(row_idx, pos, current_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                rows[row_idx] = current_row[:]
                return backtrack(row_idx + 1, 0, [0])
            
            if pos == 0:
                return backtrack(row_idx, 1, [0])
            
            prev_val = current_row[pos - 1]
            min_val = prev_val + 1
            
            # Determine candidates
            candidates = list(range(min_val, M + 1))
            
            # Randomize order for diversity in restarts
            if restart > 0:
                random.shuffle(candidates)
            
            for val in candidates:
                new_diffs = set()
                valid = True
                for i in range(len(current_row)):
                    d = val - current_row[i]
                    if d in new_diffs or d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                rem = k + 1 - pos - 1
                if rem > 0 and val + rem > M:
                    continue
                
                current_row.append(val)
                old_used = used_diffs.copy()
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos + 1, current_row):
                    return True
                
                current_row.pop()
                used_diffs.clear()
                used_diffs.update(old_used)
            
            return False
        
        if backtrack(0, 0, []):
            return [r[:] for r in rows]
    
    return None

def _local_search_improve(rows, n_rows, k, deadline):
    """
    Local search to reduce the maximum entry.
    """
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs(r)
    cur_row_diffs = [_row_diffs(r) for r in cur]
    
    best = [r[:] for r in cur]
    best_val = cur_val
    
    T = 30.0
    T_min = 0.1
    cool = 0.9998
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        old_val = cur[ri][pi]
        
        # Generate candidates
        candidates = []
        # Try to decrease
        for d in range(1, 15):
            nv = old_val - d
            if nv <= cur[ri][pi - 1]:
                break
            candidates.append(nv)
        # Small increases
        for d in range(1, 8):
            nv = old_val + d
            if pi < k and nv >= cur[ri][pi + 1]:
                break
            candidates.append(nv)
        # Random perturbations
        for _ in range(10):
            d = random.randint(-10, 10)
            if d == 0:
                continue
            nv = old_val + d
            if nv > cur[ri][pi - 1] and (pi == k or nv < cur[ri][pi + 1]):
                candidates.append(nv)
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= cur[ri][pi - 1]:
                continue
            if pi < k and new_val >= cur[ri][pi + 1]:
                continue
            
            cand_row = cur[ri][:]
            cand_row[pi] = new_val
            
            # Check row is Sidon
            row_diffs = _row_diffs(cand_row)
            if len(row_diffs) != k * (k + 1) // 2:
                continue
            
            # Check no conflicts with other rows
            other_diffs = cur_used - cur_row_diffs[ri]
            if row_diffs & other_diffs:
                continue
            
            # Compute new max
            cand_max = max(max(r) for r in cur) if n_rows > 1 else 0
            cand_max = max(cand_max, max(cand_row))
            
            delta_e = cand_max - cur_val
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                accept = random.random() < 0.3
            else:
                if T > T_min:
                    prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                    accept = random.random() < prob
            
            if accept:
                cur_used -= cur_row_diffs[ri]
                cur_used |= row_diffs
                cur[ri] = cand_row
                cur_row_diffs[ri] = row_diffs
                cur_val = cand_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cool)
        
        if no_improve > 15000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set()
            for r in cur:
                cur_used |= _row_diffs(r)
            cur_row_diffs = [_row_diffs(r) for r in cur]
            T = 30.0
            no_improve = 0
    
    return best

def construct(n_rows, k):
    """
    Construct a difference triangle set for n_rows rows of k+1 integers.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    random.seed(42)
    
    # Get initial solution from greedy
    greedy_rows = _build_greedy(n_rows, k)
    greedy_max = _max_entry(greedy_rows)
    
    best = [r[:] for r in greedy_rows]
    best_val = greedy_max
    
    # Try to find a solution with smaller M using backtracking
    # Start from the counting bound and go up
    bound = _counting_bound(n_rows, k)
    
    # Try a few values around the bound
    for M in range(bound, min(bound + 15, greedy_max + 1)):
        if time.time() > deadline:
            break
        
        result = _backtrack_with_restarts(n_rows, k, M, deadline, max_restarts=10)
        if result is not None and _valid(result, n_rows, k):
            if _max_entry(result) < best_val:
                best = result
                best_val = _max_entry(result)
            # If we found one, we can stop trying larger M for this phase
            break
    
    # If backtracking didn't improve, or to further improve, do local search
    if time.time() < deadline - 5.0:
        improved = _local_search_improve(best, n_rows, k, deadline)
        if improved is not None and _valid(improved, n_rows, k):
            if _max_entry(improved) < best_val:
                best = improved
                best_val = _max_entry(improved)
    
    # Final validation
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best

# EVOLVE-BLOCK-END
