import time
import random
import math
import sys
from collections import Counter

_DEADLINE = 135.0

def _row_diffs(row):
    """Return all positive differences a_j - a_i for i < j in a row."""
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d

def _build_counter(rows):
    """Build a Counter of all differences across all rows."""
    c = Counter()
    for row in rows:
        for d in _row_diffs(row):
            c[d] += 1
    return c

def _conflicts_from_counter(c):
    """Count total conflicts (differences that appear more than once)."""
    return sum(v - 1 for v in c.values() if v > 1)

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(r[-1] for r in rows)

def _valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _greedy_row(used, k, rng=None, rand_factor=0.0):
    """
    Build a single row greedily, choosing the smallest (or randomized small) 
    values that don't conflict with used differences.
    """
    row = [0]
    while len(row) < k + 1:
        x = row[-1] + 1
        found = False
        while True:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # This is valid
                if rng is not None and rand_factor > 0 and rng.random() < rand_factor:
                    # Randomly accept a slightly larger value sometimes
                    # to escape local traps
                    pass  # Actually, for greedy, we just take the smallest
                row.append(x)
                used |= new_diffs
                found = True
                break
            x += 1
        if not found:
            break
    return row

def _greedy_rows(n_rows, k, rng=None, rand_factor=0.0):
    """Build n_rows using greedy method."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, rand_factor)
        rows.append(row)
        # Update used with all diffs from this row
        for d in _row_diffs(row):
            used.add(d)
    return rows

def _il_kick(rows, k, rng, strength=1):
    """
    Perturb a solution by randomly modifying entries.
    """
    new_rows = [list(r) for r in rows]
    n_rows = len(rows)
    
    for _ in range(strength):
        # Pick a random row and a random position
        r_idx = rng.randrange(n_rows)
        pos = rng.randrange(1, k)  # Not the first element (0)
        
        # Get current value
        old_val = new_rows[r_idx][pos]
        
        # Propose a new value
        # Bias towards smaller values if possible
        min_allowed = new_rows[r_idx][pos - 1] + 1
        max_allowed = new_rows[r_idx][pos + 1] - 1 if pos + 1 < k + 1 else old_val + 20
        
        # Random step
        step = rng.randint(-10, 10)
        new_val = old_val + step
        new_val = max(min_allowed, new_val)
        new_val = min(max_allowed, new_val)
        
        if new_val != old_val:
            new_rows[r_idx][pos] = new_val
    
    return new_rows

def _local_search(rows, k, rng, time_limit=5.0):
    """
    Perform local search to minimize scope while maintaining validity.
    Returns the best solution found.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_scope = best_scope
    
    t0 = time.time()
    no_improve_time = 0.0
    last_improve = t0
    
    while time.time() - t0 < time_limit:
        # Try to improve
        r_idx = rng.randrange(len(current_rows))
        pos = rng.randrange(1, k)
        
        old_val = current_rows[r_idx][pos]
        min_allowed = current_rows[r_idx][pos - 1] + 1
        max_allowed = current_rows[r_idx][pos + 1] - 1 if pos + 1 < k + 1 else old_val + 5
        
        # Try to decrease the value
        new_val = old_val - 1
        if new_val < min_allowed:
            continue
        
        # Check if this move is valid
        # Remove old diffs involving pos
        row = current_rows[r_idx]
        old_diffs = []
        for j in range(k + 1):
            if j != pos:
                if j < pos:
                    old_diffs.append(old_val - row[j])
                else:
                    old_diffs.append(row[j] - old_val)
        
        # Check if new value creates conflicts
        new_diffs = []
        for j in range(k + 1):
            if j != pos:
                if j < pos:
                    new_diffs.append(new_val - row[j])
                else:
                    new_diffs.append(row[j] - new_val)
        
        # Check for conflicts with other rows
        # This is expensive, so only do it if scope would improve
        potential_scope = _scope([list(r) if i != r_idx else list(row[:pos]) + [new_val] + list(row[pos+1:]) for i, r in enumerate(current_rows)])
        
        if potential_scope < current_scope:
            # Check validity
            test_rows = [list(r) if i != r_idx else list(row[:pos]) + [new_val] + list(row[pos+1:]) for i, r in enumerate(current_rows)]
            if _valid(test_rows):
                current_rows = test_rows
                current_scope = potential_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    last_improve = time.time()
        
        no_improve_time = time.time() - last_improve
        if no_improve_time > 2.0:
            # Apply a kick
            current_rows = _il_kick(current_rows, k, rng, strength=1)
            current_scope = _scope(current_rows)
            last_improve = time.time()
    
    return best_rows

def _multi_start_il(n_rows, k, rng, total_time=100.0):
    """
    Run multiple starts with iterated local search.
    """
    t0 = time.time()
    best_rows = None
    best_scope = float('inf')
    
    # First, get a good initial solution
    initial_rows = _greedy_rows(n_rows, k, rng)
    if not _valid(initial_rows):
        # Fallback
        initial_rows = _greedy_rows(n_rows, k, rng, 0.0)
    
    best_rows = [list(r) for r in initial_rows]
    best_scope = _scope(best_rows)
    
    # Local search
    t0_ls = time.time()
    ls_rows = _local_search(initial_rows, k, rng, time_limit=10.0)
    ls_scope = _scope(ls_rows)
    if ls_scope < best_scope:
        best_scope = ls_scope
        best_rows = [list(r) for r in ls_rows]
    
    # Iterated Local Search
    while time.time() - t0 < total_time:
        # Kick
        kicked_rows = _il_kick(best_rows, k, rng, strength=2)
        
        # Local search
        t0_ls = time.time()
        remaining_ls = min(5.0, total_time - (time.time() - t0) - 5.0)
        if remaining_ls <= 0:
            break
        ls_rows = _local_search(kicked_rows, k, rng, time_limit=remaining_ls)
        ls_scope = _scope(ls_rows)
        
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = [list(r) for r in ls_rows]
    
    return best_rows

def _simulated_annealing(rows, k, rng, time_limit=30.0):
    """
    Simulated annealing to minimize scope while maintaining validity.
    """
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    t0 = time.time()
    T = 5.0
    cooling = 0.999
    
    while time.time() - t0 < time_limit:
        T *= cooling
        if T < 0.1:
            T = 0.1
        
        # Propose a move
        r_idx = rng.randrange(len(current_rows))
        pos = rng.randrange(1, k)
        
        old_val = current_rows[r_idx][pos]
        min_allowed = current_rows[r_idx][pos - 1] + 1
        max_allowed = current_rows[r_idx][pos + 1] - 1 if pos + 1 < k + 1 else old_val + 10
        
        new_val = old_val + rng.randint(-5, 5)
        new_val = max(min_allowed, new_val)
        new_val = min(max_allowed, new_val)
        
        if new_val == old_val:
            continue
        
        # Check validity
        test_rows = [list(r) if i != r_idx else list(current_rows[r_idx][:pos]) + [new_val] + list(current_rows[r_idx][pos+1:]) for i, r in enumerate(current_rows)]
        if not _valid(test_rows):
            continue
        
        new_scope = _scope(test_rows)
        delta = new_scope - current_scope
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_rows = test_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
    
    return best_rows

def construct(n_rows, k):
    """
    Main entry point.
    """
    t0 = time.time()
    rng = random.Random(42)
    
    # Generate initial solutions
    initial_rows = _greedy_rows(n_rows, k, rng)
    if not _valid(initial_rows):
        initial_rows = _greedy_rows(n_rows, k, rng, 0.0)
    
    # Apply local search
    ls_rows = _local_search(initial_rows, k, rng, time_limit=15.0)
    best_rows = [list(r) for r in ls_rows]
    best_scope = _scope(best_rows)
    
    # Check time
    elapsed = time.time() - t0
    remaining = _DEADLINE - elapsed
    
    # Apply simulated annealing if time permits
    if remaining > 10.0:
        sa_rows = _simulated_annealing(best_rows, k, rng, time_limit=min(30.0, remaining - 5.0))
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = [list(r) for r in sa_rows]
            best_scope = sa_scope
    
    # Apply multi-start ILS if time permits
    elapsed = time.time() - t0
    remaining = _DEADLINE - elapsed
    if remaining > 20.0:
        il_rows = _multi_start_il(n_rows, k, rng, total_time=min(60.0, remaining - 5.0))
        il_scope = _scope(il_rows)
        if il_scope < best_scope:
            best_rows = [list(r) for r in il_rows]
            best_scope = il_scope
    
    # Final check
    if not _valid(best_rows):
        # Fallback to greedy
        return _greedy_rows(n_rows, k, rng, 0.0)
    
    return best_rows

# Test harness for local debugging
if __name__ == "__main__":
    n_rows = 4
    k = 7
    start_time = time.time()
    result = construct(n_rows, k)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.2f}s")
    print(f"Valid: {_valid(result)}")
    print(f"Scope: {_scope(result)}")
    for row in result:
        print(row)
