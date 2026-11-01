import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid_row(row):
    # Check internal uniqueness
    diffs = _compute_diffs(row)
    return len(diffs) == len(row) * (len(row) - 1) // 2

def _greedy_build(n_rows, k, rng, randomize=False):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            # Try to find smallest x > row[-1]
            # such that x - a is not in used_diffs for all a in row
            # and x - a is not in internal diffs of current row (which are handled by used_diffs if we update it)
            # Actually, we need to check against existing internal diffs too?
            # No, used_diffs contains ALL diffs from ALL previous rows.
            # We need to ensure new diffs don't conflict with used_diffs.
            # And new diffs must be distinct among themselves (handled by checking against current row's diffs as we build? 
            # If we add x, new diffs are x - a for a in row.
            # If x - a1 == x - a2, then a1 == a2, impossible since row is strictly increasing.
            # So internal uniqueness is guaranteed if row is strictly increasing.
            # Wait, is it?
            # Diffs are x_j - x_i.
            # If we pick x, new diffs are x - x_0, x - x_1, ...
            # Are these distinct? Yes, because x_i are distinct.
            # Do they conflict with existing diffs in the row?
            # Existing diffs are x_j - x_i for j < current.
            # We need x - x_i != x_j - x_l.
            # So we must check against ALL used diffs (which includes previous rows AND current row's existing diffs).
            
            start_val = row[-1] + 1
            candidates = []
            # Search a window
            max_search = 20
            for x in range(start_val, start_val + max_search):
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            
            if not candidates:
                # Expand search if needed
                # For now, just fail or expand
                for x in range(start_val + max_search, start_val + 100):
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        if len(candidates) > 5: break
            
            if not candidates:
                return None # Failed to build
            
            if randomize and len(candidates) > 1:
                # Pick randomly from top few
                x = rng.choice(candidates[:3])
            else:
                x = candidates[0]
                
            row.append(x)
            for a in row[:-1]:
                used_diffs.add(x - a)
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, rng, time_budget):
    # Simulated Annealing
    best_rows = [list(r) for r in rows]
    best_score = max(max(r) for r in best_rows)
    current_rows = [list(r) for r in rows]
    current_score = best_score
    
    # Precompute diffs for other rows?
    # It's dynamic, so we recompute or update.
    # With n=4, recomputing is fast.
    
    start_time = time.time()
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while T > T_min and time.time() - start_time < time_budget and _time_left() > 2.0:
        # Pick random row and position
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k+1) # 1 to k
        
        old_val = current_rows[r][pos]
        
        # Determine range for new value
        min_val = current_rows[r][pos-1] + 1
        max_val = current_rows[r][pos+1] - 1 if pos < k else best_score + 10
        
        if min_val > max_val:
            T *= cooling
            continue
            
        # Try to pick a new value
        # Bias towards smaller values to reduce scope
        # Or random in range
        new_val = rng.randint(min_val, max_val)
        
        if new_val == old_val:
            T *= cooling
            continue
            
        # Create new row
        new_row = list(current_rows[r])
        new_row[pos] = new_val
        
        # Check validity
        new_diffs = _compute_diffs(new_row)
        
        # Check internal validity (distinct diffs)
        if len(new_diffs) != k * (k + 1) // 2:
            T *= cooling
            continue
            
        # Check conflict with other rows
        conflict = False
        for other_r in range(n_rows):
            if other_r == r:
                continue
            other_diffs = _compute_diffs(current_rows[other_r])
            if new_diffs & other_diffs:
                conflict = True
                break
                
        if conflict:
            T *= cooling
            continue
            
        # Calculate new score
        new_score = max(max(row) for row in current_rows) # Wait, only row r changed
        # Actually, max might come from another row.
        # new_score = max(max(new_row), max(max(current_rows[i]) for i in range(n_rows) if i != r))
        
        delta = new_score - current_score
        
        if delta < 0:
            # Accept
            current_rows[r] = new_row
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [list(row) for row in current_rows]
        else:
            # Accept with probability
            if rng.random() < pow(2.71828, -delta / T):
                current_rows[r] = new_row
                current_score = new_score
                
        T *= cooling
        
    return best_rows, best_score

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Greedy Initializations
    for i in range(20):
        if _time_left() < 10: break
        rows = _greedy_build(n_rows, k, rng, randomize=(i > 5))
        if rows:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _greedy_build(n_rows, k, rng, randomize=False)
        best_score = max(max(r) for r in best_rows)
        
    # Phase 2: Local Search / SA
    # Run SA on the best solution found so far
    if _time_left() > 5:
        budget = min(60.0, _time_left() - 5.0)
        ls_rows, ls_score = _local_search(best_rows, n_rows, k, rng, budget)
        if ls_score < best_score:
            best_score = ls_score
            best_rows = ls_rows
            
    # Phase 3: More SA with restarts if time allows
    if _time_left() > 10:
        budget = min(30.0, _time_left() - 5.0)
        # Restart from a slightly perturbed version of best
        perturbed = [list(r) for r in best_rows]
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k+1)
        old = perturbed[r][pos]
        new = old + rng.randint(-5, 5)
        if new > perturbed[r][pos-1] and (pos >= k or new < perturbed[r][pos+1]):
            perturbed[r][pos] = new
            # Check validity quickly? If invalid, SA will reject moves that increase conflict, 
            # but starting state should be valid or close.
            # Actually, if perturbation breaks validity, SA might struggle.
            # Let's just run SA on the best valid solution again with different seed/temp.
            
        ls_rows2, ls_score2 = _local_search(best_rows, n_rows, k, rng, budget)
        if ls_score2 < best_score:
            best_score = ls_score2
            best_rows = ls_rows2

    return best_rows
