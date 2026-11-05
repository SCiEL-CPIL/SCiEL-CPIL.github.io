import time
import random
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    seed = 0
    max_seeds = 30

    while time.time() < deadline - 5 and seed < max_seeds:
        rows = _backtracking_construct(n_rows, k, seed)
        
        if rows is not None:
            # Polish with local search
            if time.time() < deadline - 10:
                rows = _polish(rows, n_rows, k, time.time(), deadline - 5)
            
            scope = 0
            for row in rows:
                if row:
                    scope = max(scope, row[-1])
            
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
                # If we hit the lower bound, we can't do better
                if best_scope == lower_bound:
                    break
        
        seed += 1

    # If no solution found, return simple greedy
    if best_rows is None:
        best_rows = _simple_greedy(n_rows, k)
        
    return best_rows

def _backtracking_construct(n_rows, k, seed):
    """
    Backtracking search that selects the next row element by minimizing
    the 'degree' of future constraint violations. We prefer candidates that
    leave the most remaining valid difference slots for subsequent rows.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = [0]
        
        for pos in range(k):
            last_x = row[-1]
            candidates = []
            
            # Try candidates starting from last_x + 1
            # Limit search depth to keep it fast
            limit = last_x + 200
            x = last_x + 1
            
            while x <= limit and len(candidates) < 20:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # Score: how many future conflicts does this create?
                    # Approximate by counting how many used diffs are "close"
                    # to the new diffs (indicating tight packing)
                    score = 0
                    for d in new_diffs:
                        # Count used diffs within a small window
                        for delta in range(-3, 4):
                            if d + delta in used:
                                score += 1
                    candidates.append((score, x))
                x += 1
            
            if not candidates:
                # If no candidates found, expand limit
                x = limit + 1
                while len(candidates) < 20:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        score = 0
                        for d in new_diffs:
                            for delta in range(-3, 4):
                                if d + delta in used:
                                    score += 1
                        candidates.append((score, x))
                    x += 1
            
            if not candidates:
                return None  # Dead end
            
            # Sort by score (fewer conflicts first), then by x (smaller first)
            candidates.sort(key=lambda t: (t[0], t[1]))
            
            # Pick from top candidates with some randomness
            top_n = min(5, len(candidates))
            chosen_idx = rng.randint(0, top_n - 1)
            chosen_x = candidates[chosen_idx][1]
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _simple_greedy(n_rows, k):
    """Simple greedy fallback."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used.update(new_diffs)
        rows.append(row)
    
    return rows

def _polish(rows, n_rows, k, start_time, deadline):
    """Local search: try to reduce the scope by moving elements to smaller positions."""
    rng = random.Random(int(start_time * 1000) % 100000)
    
    def _valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    current_rows = [list(r) for r in rows]
    current_scope = max(max(r) for r in current_rows)
    
    for _ in range(30):  # max polish iterations
        if time.time() > deadline:
            break
        
        # Try to move a random element to a smaller position
        row_idx = rng.randint(0, n_rows - 1)
        pos_idx = rng.randint(1, k)  # positions 1..k (position 0 is always 0)
        
        old_val = current_rows[row_idx][pos_idx]
        
        # Try moving it to a smaller value
        lower = current_rows[row_idx][pos_idx - 1] + 1
        upper = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val - 1
        
        if upper < lower:
            continue
        
        # Try a few candidate values
        candidates = list(range(upper, lower - 1, -1))  # try smaller values first
        rng.shuffle(candidates)
        
        for new_val in candidates[:15]:
            if new_val >= old_val:
                continue
            
            # Temporarily update
            old = current_rows[row_idx][pos_idx]
            current_rows[row_idx][pos_idx] = new_val
            
            if _valid(current_rows):
                new_scope = max(max(r) for r in current_rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    break
                else:
                    # Revert if no improvement
                    current_rows[row_idx][pos_idx] = old
            else:
                current_rows[row_idx][pos_idx] = old
    
    return current_rows
