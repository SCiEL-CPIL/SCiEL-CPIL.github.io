"""Difference triangle set solver with backtracking, simulated annealing, and local search."""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return set of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used


def _is_valid(rows):
    """Check if all positive differences across all rows are distinct."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _greedy_row(used, k):
    """
    Greedily construct a row of k+1 integers starting at 0.
    Each new element is the smallest integer > last that doesn't create
    a difference already in 'used'.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs


def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= new_diffs
    return rows


def _randomized_greedy_row(used, k, rng, pool_size=5):
    """
    Randomized greedy: at each step, pick from the top 'pool_size' smallest
    valid candidates with some randomness.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.6:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, new_diffs


def _randomized_greedy(n_rows, k, rng, pool_size=5):
    """Randomized greedy construction with multiple restarts."""
    best_rows = None
    best_scope = float('inf')
    
    max_restarts = 50
    deadline = time.time() + 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row, new_diffs = _randomized_greedy_row(used, k, rng, pool_size)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= new_diffs
        if success:
            scope = _total_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    return best_rows if best_rows is not None else _greedy_construct(n_rows, k)


def _backtrack_row(used, k, max_val, rng, depth=0):
    """
    Backtracking search for a row of k+1 integers starting at 0.
    Tries values in random order to find diverse solutions.
    Returns (row, new_diffs) or (None, None) if no solution found.
    """
    if depth > 10000:
        return None, None
    
    row = [0]
    new_diffs = set()
    
    def _backtrack(pos, last_val):
        if pos == k + 1:
            return True
        
        # Generate candidate values in random order
        candidates = list(range(last_val + 1, max_val + 1))
        rng.shuffle(candidates)
        
        # Try smaller values first (with some randomness)
        for val in candidates:
            new_diffs_here = {val - a for a in row}
            if new_diffs_here & used or new_diffs_here & new_diffs:
                continue
            
            row.append(val)
            new_diffs.update(new_diffs_here)
            
            if _backtrack(pos + 1, val):
                return True
            
            row.pop()
            new_diffs -= new_diffs_here
        
        return False
    
    if _backtrack(1, 0):
        return row, new_diffs
    return None, None


def _backtrack_construct(n_rows, k, max_scope, rng, deadline):
    """
    Backtracking construction with time limit.
    Tries to build each row using backtracking search.
    """
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        # Estimate remaining scope needed
        remaining_rows = n_rows - row_idx
        # Each row needs at least k*(k+1)/2 differences, but they overlap
        # Use a heuristic estimate
        est_scope = max_scope
        
        # Try backtracking with increasing max values
        found = False
        for attempt_max in range(est_scope, est_scope + 50):
            if time.time() > deadline:
                return None
            
            row, new_diffs = _backtrack_row(used, k, attempt_max, rng)
            if row is not None:
                rows.append(row)
                used |= new_diffs
                found = True
                break
        
        if not found:
            return None
    
    return rows


def _local_search_shrink(rows, k, rng, deadline=None):
    """
    Local search that tries to reduce the scope by moving elements downward.
    Uses incremental difference tracking for speed.
    """
    n_rows = len(rows)
    current_rows = [r[:] for r in rows]
    current_scope = _total_scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Precompute all differences for quick validity checking
    all_diffs = set()
    for r in current_rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.add(r[j] - r[i])
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        # Try each row, each element (from end to start for shrinking)
        row_order = list(range(n_rows))
        rng.shuffle(row_order)  # Shuffle row order to break symmetry
        for row_idx in row_order:
            row = current_rows[row_idx]
            for elem_idx in range(len(row) - 1, 0, -1):
                old_val = row[elem_idx]
                # Try to decrease this element
                for new_val in range(old_val - 1, row[elem_idx - 1], -1):
                    # Check strict increasing
                    if new_val <= row[elem_idx - 1]:
                        break
                    if elem_idx < len(row) - 1 and new_val >= row[elem_idx + 1]:
                        continue
                    
                    # Remove old diffs involving elem_idx
                    old_diffs = set()
                    for j in range(elem_idx):
                        old_diffs.add(old_val - row[j])
                    for j in range(elem_idx + 1, len(row)):
                        old_diffs.add(row[j] - old_val)
                    
                    # Check new diffs
                    new_diffs = set()
                    for j in range(elem_idx):
                        new_diffs.add(new_val - row[j])
                    for j in range(elem_idx + 1, len(row)):
                        new_diffs.add(row[j] - new_val)
                    
                    # Check: new diffs must not conflict with remaining diffs
                    remaining = all_diffs - old_diffs
                    if new_diffs & remaining:
                        continue
                    
                    # Apply move
                    row[elem_idx] = new_val
                    all_diffs = remaining | new_diffs
                    new_scope = _total_scope(current_rows)
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                    current_scope = new_scope
                    improved = True
                    break  # Move on to next element
                
                if deadline is not None and time.time() > deadline:
                    return best_rows
        
        # If no improvement in a full pass, try random perturbations
        if not improved and deadline is not None and time.time() < deadline - 1:
            # Random perturbation: pick a random element and try to move it
            for _ in range(20):
                if time.time() > deadline:
                    break
                row_idx = rng.randint(0, n_rows - 1)
                row = current_rows[row_idx]
                elem_idx = rng.randint(1, len(row) - 1)
                old_val = row[elem_idx]
                step = rng.randint(-10, 10)
                if step == 0:
                    continue
                new_val = old_val + step
                if new_val <= row[elem_idx - 1] or (elem_idx < len(row) - 1 and new_val >= row[elem_idx + 1]):
                    continue
                
                # Check validity incrementally
                old_diffs = set()
                for j in range(elem_idx):
                    old_diffs.add(old_val - row[j])
                for j in range(elem_idx + 1, len(row)):
                    old_diffs.add(row[j] - old_val)
                
                new_diffs = set()
                for j in range(elem_idx):
                    new_diffs.add(new_val - row[j])
                for j in range(elem_idx + 1, len(row)):
                    new_diffs.add(row[j] - new_val)
                
                remaining = all_diffs - old_diffs
                if new_diffs & remaining:
                    continue
                
                # Accept
                row[elem_idx] = new_val
                all_diffs = remaining | new_diffs
                new_scope = _total_scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                current_scope = new_scope
                improved = True
    
    return best_rows


def _simulated_annealing(rows, k, rng, deadline):
    """
    Simulated annealing to minimize the scope.
    Tries random moves and accepts worse solutions with decreasing probability.
    """
    n_rows = len(rows)
    current_rows = [r[:] for r in rows]
    current_scope = _total_scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Precompute all differences
    all_diffs = set()
    for r in current_rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.add(r[j] - r[i])
    
    temp = 10.0
    cool_rate = 0.95
    
    while time.time() < deadline and temp > 0.1:
        # Pick a random element to move
        row_idx = rng.randint(0, n_rows - 1)
        row = current_rows[row_idx]
        elem_idx = rng.randint(1, len(row) - 1)
        old_val = row[elem_idx]
        
        # Try a random step
        max_step = min(20, old_val - row[elem_idx - 1] - 1, 
                      (row[elem_idx + 1] - old_val - 1) if elem_idx < len(row) - 1 else 20)
        if max_step <= 0:
            temp *= cool_rate
            continue
        
        step = rng.randint(-max_step, max_step)
        if step == 0:
            continue
        new_val = old_val + step
        
        if new_val <= row[elem_idx - 1]:
            continue
        if elem_idx < len(row) - 1 and new_val >= row[elem_idx + 1]:
            continue
        
        # Check validity
        old_diffs = set()
        for j in range(elem_idx):
            old_diffs.add(old_val - row[j])
        for j in range(elem_idx + 1, len(row)):
            old_diffs.add(row[j] - old_val)
        
        new_diffs = set()
        for j in range(elem_idx):
            new_diffs.add(new_val - row[j])
        for j in range(elem_idx + 1, len(row)):
            new_diffs.add(row[j] - new_val)
        
        remaining = all_diffs - old_diffs
        if new_diffs & remaining:
            temp *= cool_rate
            continue
        
        # Calculate new scope
        row[elem_idx] = new_val
        all_diffs = remaining | new_diffs
        new_scope = _total_scope(current_rows)
        
        delta = new_scope - current_scope
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            row[elem_idx] = old_val
            all_diffs = remaining | old_diffs
        
        temp *= cool_rate
    
    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = rows
    best_scope = _total_scope(rows)
    
    # Phase 1: Randomized greedy restarts
    if time.time() < deadline - 30:
        for restart in range(30):
            if time.time() > deadline - 30:
                break
            rand_rows = _randomized_greedy(n_rows, k, rng, pool_size=rng.choice([3, 5, 8]))
            if rand_rows is not None:
                scope = _total_scope(rand_rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rand_rows]
    
    # Phase 2: Local search shrink on best solution
    if time.time() < deadline - 20:
        ls_rows = _local_search_shrink(best_rows, k, rng, deadline=deadline - 15)
        scope = _total_scope(ls_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in ls_rows]
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        sa_rows = _simulated_annealing(best_rows, k, rng, deadline=deadline - 10)
        scope = _total_scope(sa_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in sa_rows]
    
    # Phase 4: More randomized restarts + local search if time remains
    if time.time() < deadline - 10:
        for restart in range(15):
            if time.time() > deadline - 5:
                break
            rand_rows = _randomized_greedy(n_rows, k, rng, pool_size=rng.choice([3, 5, 8]))
            if rand_rows is not None:
                # Apply local search to this candidate
                ls_candidate = _local_search_shrink(rand_rows, k, rng, deadline=deadline - 3)
                scope = _total_scope(ls_candidate)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in ls_candidate]
    
    # Phase 5: Final local search polish
    if time.time() < deadline - 2:
        final_rows = _local_search_shrink(best_rows, k, rng, deadline=deadline - 1)
        scope = _total_scope(final_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in final_rows]
    
    return best_rows

# EVOLVE-BLOCK-END
