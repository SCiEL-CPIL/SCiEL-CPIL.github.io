import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used

def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _heuristic_backtrack(n_rows, k, deadline, rng, max_scope_hint=None):
    """
    Backtracking search with heuristic ordering.
    Heuristic: Prefer candidates that introduce differences with small 1/d sum.
    This tends to prefer larger differences, which are less likely to conflict
    with the "dense" small differences that are critical for packing.
    """
    start_time = time.time()
    best_rows = None
    best_scope = max_scope_hint if max_scope_hint is not None else float('inf')
    
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, current_row):
        nonlocal best_rows, best_scope, used_diffs, rows
        
        if time.time() > deadline:
            return False
        
        # If we have completed a row, move to the next
        if len(current_row) == k + 1:
            rows.append(current_row)
            if row_idx + 1 == n_rows:
                # All rows complete
                current_scope = max(r[-1] for r in rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
                rows.pop()
                return True
            else:
                # Recurse to next row
                if backtrack(row_idx + 1, [0]):
                    rows.pop()
                    return True
                rows.pop()
                return False
        
        # Determine candidates for the next element in current_row
        last_val = current_row[-1]
        
        # Upper bound for next element
        # We don't know the final scope, but we can limit it to best_scope - 1
        # to ensure we find a better solution if we are improving.
        # If best_scope is inf, we need a reasonable upper bound to avoid infinite search.
        # A safe upper bound for DTS(7,5) is around 200.
        upper_bound = min(best_scope - 1, 200) if best_scope < float('inf') else 200
        
        if last_val >= upper_bound:
            return False
            
        candidates = []
        
        # Generate candidates
        # To speed up, we might not iterate all the way to upper_bound if it's large
        # But for k=5, the branching factor is manageable if we prune well.
        
        for x in range(last_val + 1, upper_bound + 1):
            new_diffs = {x - a for a in current_row}
            
            # Check for conflicts with globally used differences
            if new_diffs & used_diffs:
                continue
            
            # Heuristic Score: Sum of 1/d for new differences
            # Lower score is better (prefers large differences)
            # We also add a small penalty for x to prefer smaller values if scores are close
            cost = sum(1.0 / d for d in new_diffs)
            
            candidates.append((cost, x, new_diffs))
            
            # Limit candidates to keep search space small
            # If we have too many, we can break, but sorting is better.
            # However, iterating to 200 is fast.
        
        if not candidates:
            return False
        
        # Sort candidates by cost, then by value
        candidates.sort(key=lambda item: (item[0], item[1]))
        
        # Randomization: To avoid deterministic failures, we can shuffle the top N candidates
        # or pick a random one from the top 5 with some probability.
        # For now, let's try the first few.
        # To add diversity across restarts, we can randomize the order of the top 10.
        
        top_n = min(10, len(candidates))
        top_candidates = candidates[:top_n]
        
        # Shuffle the top candidates to explore different branches in different runs
        rng.shuffle(top_candidates)
        
        for cost, x, new_diffs in top_candidates:
            # Apply choice
            current_row.append(x)
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, current_row):
                # Clean up before returning True
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                return True
            
            # Undo choice
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
            # Check time again
            if time.time() > deadline:
                return False
        
        return False
    
    # Start backtracking
    backtrack(0, [0])
    
    return best_rows

def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements.
    Works at the difference level: for each element position, try alternative values."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        # Try to reduce elements at or near the current maximum
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):  # from last to first
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope - 2:
                    continue
                
                # Try reducing this value
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                # Sample candidate values, preferring smaller ones
                n_candidates = min(20, high - low + 1)
                candidates = rng.sample(range(low, high + 1), n_candidates)
                candidates.sort()  # try smallest first for max reduction
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
                        continue
                    
                    # Compute old diffs involving pos
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    for i in range(pos + 1, len(row)):
                        old_diffs.add(row[i] - old_val)
                    
                    # Compute new diffs involving pos
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for i in range(pos + 1, len(row)):
                        new_diffs.add(row[i] - new_val)
                    
                    # Check for conflicts
                    conflict = new_diffs & (used - old_diffs)
                    if conflict:
                        continue
                    
                    # Accept this change
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break
    
    return rows

def _simulated_annealing(rows, k, deadline, rng):
    """Simulated annealing to reduce the maximum entry by coordinated row modifications."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    current_scope = _scope(rows)
    T = 20.0  # initial temperature
    cooling = 0.9995
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        # Pick a random row and position to modify
        ri = rng.randint(0, len(rows) - 1)
        row = rows[ri]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 15
        
        if low >= high:
            T *= cooling
            continue
        
        # Sample new value
        new_val = rng.randint(low, high)
        if new_val == old_val:
            T *= cooling
            continue
        
        # Compute diff changes
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for i in range(pos + 1, len(row)):
            old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for i in range(pos + 1, len(row)):
            new_diffs.add(row[i] - new_val)
        
        conflict = new_diffs & (used - old_diffs)
        if conflict:
            T *= cooling
            continue
        
        # Accept if improves or with probability exp(-delta/T)
        new_scope = _scope([row[:pos] + [new_val] + row[pos + 1:]] + rows[:ri] + rows[ri+1:])
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            # Accept move
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        
        T *= cooling
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Heuristic Backtracking with multiple restarts
    # We allocate 60% of the time to this phase to find a good initial solution
    bt_deadline = start_time + (deadline - start_time) * 0.6
    
    for restart in range(50):  # Max restarts, but controlled by time
        if time.time() > bt_deadline:
            break
            
        # Vary the seed for each restart to explore different search paths
        rng_restart = random.Random(42 + restart)
        
        # In the first few restarts, we don't have a hint.
        # Later, we use the best scope found to prune.
        hint = best_scope if best_scope < float('inf') else None
        
        rows = _heuristic_backtrack(n_rows, k, bt_deadline, rng_restart, max_scope_hint=hint)
        
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
            # If we found a solution, we can continue to try to improve it
            # or break if we hit a very low bound (e.g. 105)
            if best_scope <= 105:
                break

    # If no solution found via backtracking (unlikely), fall back to greedy
    if best_rows is None:
        # Simple greedy fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                row.append(x)
                used |= new_diffs
            best_rows.append(row)
        best_scope = _scope(best_rows)

    # Phase 2: Simulated Annealing
    if time.time() < deadline - 20.0:
        sa_rows = _simulated_annealing(best_rows, k, deadline - 20.0, rng)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope

    # Phase 3: Local Search
    if time.time() < deadline - 5.0:
        improved = _local_search_improve(best_rows, k, deadline - 5.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope

    return best_rows

# EVOLVE-BLOCK-END
