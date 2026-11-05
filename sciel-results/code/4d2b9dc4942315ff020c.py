import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with multiple seeds
    seed = 0
    max_initial_seeds = 200
    while time.time() < deadline - 65 and seed < max_initial_seeds:
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
        seed += 1
    
    if best_rows is None:
        # Fallback - construct a valid (though potentially large) solution
        # For k=5, n_rows=7: we need 7*15=105 distinct differences
        # Simple construction: use powers or similar
        return [[0, 1, 3, 7, 15, 31]] * n_rows

    # Phase 2: Local search with targeted row replacement
    if time.time() < deadline - 15:
        improved = _local_search(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final polish - try to squeeze the max row
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
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

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i+1, n):
            diffs.add(row[j] - ri)
    return diffs

def _greedy_construct(n_rows, k, seed):
    """
    Greedy construction with lookahead.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = _generate_row_greedy(used, k, rng)
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs(row)
    
    return rows

def _generate_row_greedy(used, k, rng):
    """
    Generate a single row avoiding 'used' differences.
    Uses a scoring function that balances compactness and future flexibility.
    """
    row = [0]
    row_diffs = set()
    
    for step in range(k):
        last_x = row[-1]
        best_x = None
        best_score = float('inf')
        
        # Search in a window
        x = last_x + 1
        # Adaptive window size
        window = 80 + step * 20
        
        for _ in range(window):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & row_diffs):
                # Score: primarily minimize x, but penalize if it blocks future
                # Check if this choice leads to a dead end quickly
                if step < k - 1:
                    slack = _quick_slack_check(row + [x], used | row_diffs, 15)
                    if slack == 0:
                        score = x * 5  # Heavy penalty for dead end
                    else:
                        score = x - min(slack, 10) * 0.5
                else:
                    score = x
                
                if score < best_score:
                    best_score = score
                    best_x = x
            x += 1
        
        if best_x is None:
            # Expand search significantly
            for x in range(last_x + window + 1, last_x + window + 500):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and not (new_diffs & row_diffs):
                    if step < k - 1:
                        slack = _quick_slack_check(row + [x], used | row_diffs, 10)
                        if slack == 0:
                            continue
                    best_x = x
                    break
        
        if best_x is None:
            return None
        
        row.append(best_x)
        for a in row[:-1]:
            row_diffs.add(best_x - a)
    
    return row

def _quick_slack_check(row_prefix, forbidden_diffs, depth):
    """
    Count how many valid next elements exist after row_prefix.
    """
    count = 0
    last = row_prefix[-1]
    for y in range(last + 1, last + depth + 1):
        new_diffs = {y - a for a in row_prefix}
        if not (new_diffs & forbidden_diffs):
            count += 1
            if count >= 20:
                break
    return count

def _local_search(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search: iteratively replace rows with tighter alternatives.
    Focus on rows that have large max elements.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute row diffs
    row_diffs_list = [_row_diffs(r) for r in rows]
    
    def get_other_diffs(exclude_idx):
        other = set()
        for i in range(n_rows):
            if i != exclude_idx:
                other |= row_diffs_list[i]
        return other
    
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Choose which row to replace
        # Bias towards rows with larger max
        weights = [rows[i][-1] + 1 for i in range(n_rows)]
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        target_idx = 0
        for i in range(n_rows):
            cumulative += weights[i]
            if r <= cumulative:
                target_idx = i
                break
        
        other_diffs = get_other_diffs(target_idx)
        
        # Try to generate a tighter row
        # We want a row with max < current rows[target_idx][-1]
        current_max = rows[target_idx][-1]
        
        # Try multiple strategies
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Strategy: constrained generation
            # Try to fit within a slightly smaller max
            target_max = current_max - 1 - rng.randint(0, 5)
            if target_max < k + 1:  # Minimum possible max for k+1 elements is k (if 0..k)
                target_max = current_max
            
            new_row = _generate_constrained_row(other_diffs, k, target_max, rng)
            
            if new_row is not None:
                new_diffs = _row_diffs(new_row)
                if not (new_diffs & other_diffs):
                    # Valid replacement
                    new_rows = rows[:]
                    new_rows[target_idx] = new_row
                    new_scope = _scope(new_rows)
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in new_rows]
                        rows = new_rows
                        row_diffs_list = [_row_diffs(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
    
    return best_rows

def _generate_constrained_row(used_diffs, k, max_allowed, rng):
    """
    Generate a row with max element <= max_allowed, avoiding used_diffs.
    """
    # If max_allowed is too small, return None
    # Minimum possible max is k (for 0,1,2,...,k)
    if max_allowed < k:
        return None
    
    # Try backtracking with constraint
    row = [0]
    row_diffs = set()
    
    def backtrack(pos, current_max):
        if pos == k + 1:
            return True
        
        last_x = row[-1]
        start = last_x + 1
        end = min(max_allowed, last_x + 100)  # Limit search
        
        # Randomize order for diversity
        candidates = list(range(start, end + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs or new_diffs & row_diffs:
                continue
            
            row.append(x)
            new_internal = set(new_diffs)
            row_diffs |= new_internal
            
            if backtrack(pos + 1, max(x, current_max)):
                return True
            
            row.pop()
            row_diffs -= new_internal
        
        return False
    
    if backtrack(1, 0):
        return row
    return None

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: repeatedly try to reduce the max element of the row with the largest max.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    iteration = 0
    while time.time() < deadline - 2 and iteration < 100:
        iteration += 1
        
        # Find row with max element
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Get diffs of all other rows
        other_diffs = set()
        for i in range(n_rows):
            if i != max_idx:
                other_diffs |= _row_diffs(rows[i])
        
        # Try to generate a row with max < current_max
        for target_max in range(current_max - 1, current_max - 20, -1):
            if time.time() > deadline - 2:
                break
            
            if target_max < k:
                break
            
            # Try multiple attempts
            for _ in range(5):
                new_row = _generate_constrained_row(other_diffs, k, target_max, rng)
                if new_row is not None:
                    new_diffs = _row_diffs(new_row)
                    if not (new_diffs & other_diffs):
                        new_rows = rows[:]
                        new_rows[max_idx] = new_row
                        new_scope = _scope(new_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in new_rows]
                            rows = new_rows
                        break
            else:
                continue
            break  # Found improvement, restart the loop
    
    return best_rows
