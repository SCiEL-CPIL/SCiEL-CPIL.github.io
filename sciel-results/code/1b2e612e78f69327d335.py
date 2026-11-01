import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Compute set of positive differences in a row."""
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

def _validate_solution(rows, n_rows, k):
    """Check if all rows are valid and all cross-row differences are distinct."""
    if len(rows) != n_rows:
        return False
    all_used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        rd = _row_diffs(row)
        if rd & all_used:
            return False
        all_used |= rd
    return True

def _greedy_init(n_rows, k):
    """
    Greedy: build each row with the smallest possible next element that avoids all used diffs.
    """
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_d = {x - a for a in row}
                if not (new_d & used) and len(new_d) == len(row):
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows

def _all_diffs(rows):
    s = set()
    for r in rows:
        s |= _row_diffs(r)
    return s

def _global_max(rows):
    return max(max(r) for r in rows)

def _try_backtrack(n_rows, k, max_scope, time_limit=3.0):
    """
    Try to construct n_rows rows, each a sorted list of k+1 ints from [0, max_scope],
    starting at 0, with all differences distinct across rows.
    Returns a valid list of rows or None.
    """
    start = time.time()
    deadline = start + time_limit
    rows = [None] * n_rows
    used = set()
    
    def fill_row(row_idx, depth, cur_row, cur_diffs):
        if time.time() > deadline:
            return None
        if depth == k + 1:
            return cur_row[:]
        
        lo = cur_row[-1] + 1 if depth > 0 else 1
        hi = max_scope
        # Pruning: need enough room for remaining entries
        remaining = k + 1 - depth
        if depth > 0:
            hi = min(hi, max_scope)
            # cur_row[-1] + remaining - 1 <= max_scope
            if cur_row[-1] + remaining - 1 > max_scope:
                return None
        
        # Try values from hi down to lo (reverse might find tighter solutions faster? 
        # Or just try all. Let's try increasing first for standard backtracking.)
        for v in range(lo, hi + 1):
            new_diffs = {v - a for a in cur_row}
            if len(new_diffs) != len(cur_row):
                continue
            if new_diffs & cur_diffs:
                continue
            if new_diffs & used:
                continue
            
            result = fill_row(row_idx, depth + 1, cur_row + [v], cur_diffs | new_diffs)
            if result is not None:
                return result
        return None
    
    def backtrack(row_idx):
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
        
        # Start row with [0]
        if row_idx == 0:
            res = fill_row(row_idx, 1, [0], set())
        else:
            res = fill_row(row_idx, 1, [0], set())
        
        if res is not None:
            rd = _row_diffs(res)
            # Check global collision again just in case
            if rd & used:
                return False
            rows[row_idx] = res
            used.update(rd)
            if backtrack(row_idx + 1):
                return True
            used.difference_update(rd)
            rows[row_idx] = None
        
        return False
    
    if backtrack(0):
        return rows
    return None

def _simulated_anneal(rows, n_rows, k, time_limit):
    """
    SA to minimize global max.
    """
    start = time.time()
    deadline = start + time_limit
    current = [r[:] for r in rows]
    best = [r[:] for r in current]
    cur_score = _global_max(current)
    best_score = cur_score
    
    T = 5.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline:
        # Pick a random row and a random position (not first)
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current[ri][pi]
        
        # Generate a new value
        # We want to potentially lower the global max, so bias towards smaller values if possible,
        # but also allow increases if it helps future moves.
        # Simple move: pick a random value between prev+1 and next-1 (or max_scope if last).
        lo = current[ri][pi - 1] + 1
        hi = current[ri][pi + 1] - 1 if pi < k else cur_score + 20
        
        if lo > hi:
            T = max(T_min, T * alpha)
            continue
        
        new_val = random.randint(lo, hi)
        if new_val == old_val:
            continue
        
        # Apply move
        current[ri][pi] = new_val
        
        # Check validity
        rd = _row_diffs(current[ri])
        # Check internal collisions (should be none if strictly increasing, but diff sets might overlap? No, strictly increasing implies unique positive diffs within row? 
        # Wait. If row is [0, 2, 4], diffs are 2, 4, 2. Collision! 
        # So we MUST check internal collisions.
        if len(rd) != k * (k + 1) // 2:
            current[ri][pi] = old_val
            T = max(T_min, T * alpha)
            continue
        
        # Check global collisions
        # Efficiently: compute all other diffs and check intersection.
        # For small n, k this is fine.
        other_diffs = set()
        for j in range(n_rows):
            if j != ri:
                other_diffs |= _row_diffs(current[j])
        
        if rd & other_diffs:
            # Invalid move
            current[ri][pi] = old_val
            # In SA, do we accept invalid states? No, usually we stay in valid space.
            # So we just revert.
            T = max(T_min, T * alpha)
            continue
        
        # Valid move. Compute new score.
        new_score = _global_max(current)
        delta = new_score - cur_score
        
        accept = False
        if delta <= 0:
            accept = True
        else:
            if T > T_min:
                p = math.exp(-delta / T)
                if random.random() < p:
                    accept = True
        
        if accept:
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best = [r[:] for r in current]
        else:
            # Revert
            current[ri][pi] = old_val
            
        T = max(T_min, T * alpha)
        
    return best

def _local_search(rows, n_rows, k, time_limit):
    """
    A more targeted local search: try to reduce the max entry of the row that currently has the global max.
    """
    start = time.time()
    deadline = start + time_limit
    current = [r[:] for r in rows]
    best = [r[:] for r in current]
    best_score = _global_max(current)
    
    while time.time() < deadline:
        # Identify rows with the current global max
        cur_max = _global_max(current)
        target_rows = [i for i in range(n_rows) if max(current[i]) == cur_max]
        
        if not target_rows:
            break
            
        # Pick one randomly
        ri = random.choice(target_rows)
        
        # Try to reduce the last element (which is the max)
        # But reducing the last element might require adjusting previous ones.
        # Let's try to replace the entire row with a "tighter" one.
        # Generate a candidate row that is <= current row.
        
        # Simple heuristic: try to shift the last few elements down.
        moved = False
        # Try moving the last element down
        lo = current[ri][-2] + 1
        hi = current[ri][-1] - 1
        if lo <= hi:
            # Try a few random candidates in this range
            for _ in range(10):
                new_val = random.randint(lo, hi)
                cand_row = current[ri][:]
                cand_row[-1] = new_val
                
                rd = _row_diffs(cand_row)
                if len(rd) != k * (k + 1) // 2:
                    continue
                
                other_diffs = set()
                for j in range(n_rows):
                    if j != ri:
                        other_diffs |= _row_diffs(current[j])
                
                if rd & other_diffs:
                    continue
                
                # Valid!
                current[ri] = cand_row
                new_max = _global_max(current)
                if new_max < best_score:
                    best_score = new_max
                    best = [r[:] for r in current]
                moved = True
                break
        
        if not moved:
            # Try moving a middle element to allow the last one to be smaller
            # This is complex. Just do a random SA step on this row instead.
            pass
            
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initial Greedy
    rows = _greedy_init(n_rows, k)
    best_val = _global_max(rows)
    
    # 2. Backtracking to find a solution with a smaller scope
    # We know the greedy result. Let's try to find one with scope < best_val.
    # We'll try scopes from (best_val - 1) down to (best_val - 10) or so.
    # Or better: try to find a solution with scope M where M is close to the theoretical bound.
    # For (7,5), bound is 105. Greedy is ~115.
    
    # Let's try backtracking with M = best_val - 1, best_val - 2, ...
    # But backtracking is exponential. We limit time per attempt.
    
    found_better = False
    # Try a few values
    for offset in range(1, 15):
        M = best_val - offset
        if M < 100:  # Don't go too low blindly
            break
            
        # Check time
        if time.time() > deadline - 20:
            break
            
        # Allocate time: 2s per attempt
        sol = _try_backtrack(n_rows, k, M, time_limit=2.0)
        if sol is not None:
            if _validate_solution(sol, n_rows, k):
                rows = sol
                best_val = M
                found_better = True
                # If we found one, we can try to go lower
                # But maybe just break and polish?
                # Let's continue trying to go lower if time permits.
    
    # 3. Simulated Annealing / Local Search to polish
    # This is the main driver for improvement if backtracking didn't hit the optimum.
    # SA can escape local minima and find tighter packings.
    
    # Run SA with remaining time
    if time.time() < deadline - 5:
        sa_time = deadline - 5 - time.time()
        polished = _simulated_anneal(rows, n_rows, k, sa_time)
        if polished and _validate_solution(polished, n_rows, k):
            if _global_max(polished) < best_val:
                rows = polished
                best_val = _global_max(polished)
    
    # 4. Final Local Search for any last minute gains
    if time.time() < deadline - 2:
        ls_time = deadline - 2 - time.time()
        polished2 = _local_search(rows, n_rows, k, ls_time)
        if polished2 and _validate_solution(polished2, n_rows, k):
            if _global_max(polished2) < best_val:
                rows = polished2
                best_val = _global_max(polished2)
    
    return rows

# EVOLVE-BLOCK-END
