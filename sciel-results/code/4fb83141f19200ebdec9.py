import time
import random
import math
import numpy as np

_DEADLINE = 0.0
_BEST = None

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _fast_greedy(k, n_rows, rng, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _row_to_bitset(row):
    """Convert a row to a difference bitset (Python int)."""
    mask = 0
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _row_scope(row):
    return row[-1] if row else 0

def _greedy_row_with_mask(k, used_mask, rng, max_scope_hint=150):
    """
    Greedily construct a row avoiding differences in used_mask.
    Returns (row, bitset) or (None, 0) if fails.
    """
    row = [0]
    x = 0
    row_mask = 0
    
    # Estimate max scope needed: at least k*(k+1)/2 differences, min gap 1
    # But we want small scope. Let's try up to max_scope_hint.
    # Actually, let's just try to build it. If it fails, return None.
    
    for _ in range(k):
        found = False
        # Try values in increasing order, but with some randomization for diversity
        # We want to keep scope small, so try smaller values first.
        # But pure greedy might get stuck. Let's try a few random starts or just greedy.
        # For speed, let's do greedy first, then random if fails.
        
        # Greedy: find smallest x' > x such that no diff conflict with used_mask or row_mask
        # Note: row_mask contains diffs within current row. used_mask contains diffs from other rows.
        # We need (new_diffs & used_mask) == 0 AND (new_diffs & row_mask) == 0
        
        # To speed up, we can iterate x from x+1 upwards.
        # How far? If we don't find one quickly, we might need to go further.
        # Let's limit the search range per step to avoid infinite loops, but expand if needed.
        
        # Actually, for k=5, the row length is 6.
        # The number of differences in a row is 15.
        # The maximum difference is the scope.
        # If scope is ~112, the bitset is ~112 bits.
        
        # Let's just scan from x+1 to a reasonable limit.
        # If we fail, we return None.
        
        limit = x + 100  # Arbitrary limit per step
        for t in range(x + 1, limit):
            new_diffs = 0
            conflict = False
            for a in row:
                d = t - a
                if d <= 0:
                    continue
                if (used_mask >> d) & 1:
                    conflict = True
                    break
                if (row_mask >> d) & 1:
                    conflict = True
                    break
                new_diffs |= (1 << d)
            
            if not conflict:
                row.append(t)
                row_mask |= new_diffs
                x = t
                found = True
                break
        
        if not found:
            return None, 0
            
    return row, row_mask

def _evolutionary_search(n_rows, k, deadline, rng):
    """
    Population-based evolutionary algorithm.
    Individuals are sets of rows.
    Fitness is scope (lower is better).
    Operations:
    1. Replace a row in an individual with a new greedy row (avoiding other rows' diffs).
    2. Crossover: take rows from two parents (if they don't conflict).
    3. Mutation: slightly modify a row (shift elements) and repair.
    """
    
    # Initialize population
    pop_size = 20
    population = []
    
    # Add some greedy solutions
    for i in range(pop_size):
        if i < 2:
            rows = _fast_greedy(k, n_rows, rng, spread=(i == 1))
        else:
            # Randomized greedy
            used = 0
            rows = []
            for _ in range(n_rows):
                row, mask = _greedy_row_with_mask(k, used, rng)
                if row is None:
                    # Fallback to simple greedy
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        x += 1
                        new = 0
                        for a in row:
                            new |= (1 << (x - a))
                        if (used & new) == 0:
                            row.append(x)
                            used |= new
                        else:
                            # Skip this x, try next
                            # Actually, _fast_greedy logic is better here
                            pass
                    # This fallback is weak. Let's just use _fast_greedy for all init if this fails
                    if len(row) != k + 1:
                        rows = _fast_greedy(k, n_rows, rng)
                        break
                else:
                    used |= mask
                    rows.append(row)
            if len(rows) != n_rows:
                rows = _fast_greedy(k, n_rows, rng)
        
        # Validate
        if _valid(rows, k):
            population.append(rows)
    
    if not population:
        return None
    
    # Sort by scope
    population.sort(key=_scope)
    
    best_rows = population[0]
    best_scope = _scope(best_rows)
    
    # Main loop
    start_time = time.time()
    stagnation_count = 0
    max_stagnation = 50
    
    while time.time() < deadline - 2:
        # Pick a parent (tournament selection)
        if len(population) > 2:
            i1 = rng.randint(0, len(population) - 1)
            i2 = rng.randint(0, len(population) - 1)
            i3 = rng.randint(0, len(population) - 1)
            # Select the best of 3
            best_idx = i1
            best_s = _scope(population[i1])
            if _scope(population[i2]) < best_s:
                best_idx = i2
                best_s = _scope(population[i2])
            if _scope(population[i3]) < best_s:
                best_idx = i3
                best_s = _scope(population[i3])
            parent = population[best_idx]
        else:
            parent = population[0]
        
        # Create child
        child = [list(r) for r in parent]
        
        # Mutation strategy
        strategy = rng.random()
        
        if strategy < 0.6:
            # Replace one row
            i = rng.randint(0, n_rows - 1)
            # Compute used_mask from other rows
            used_mask = 0
            for j in range(n_rows):
                if j != i:
                    used_mask |= _row_to_bitset(child[j])
            
            # Try to generate a new row for index i
            # To improve scope, we might want a row with smaller scope than current.
            # But greedy finds the smallest possible given the constraints.
            # So we just replace it.
            new_row, new_mask = _greedy_row_with_mask(k, used_mask, rng)
            
            if new_row is not None:
                # Check if this improves the overall scope
                old_scope = _scope(child)
                child[i] = new_row
                new_scope = _scope(child)
                
                # Accept if valid (it is, by construction) and scope is not worse?
                # Or accept with probability if worse (SA-like)?
                # For now, let's only accept if scope <= old_scope or random.
                if new_scope <= old_scope or rng.random() < 0.1:
                    # Add to population
                    population.append(child)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = child
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    child[i] = parent[i] # Revert
        elif strategy < 0.8:
            # Shift one row
            i = rng.randint(0, n_rows - 1)
            old_row = child[i]
            new_row = list(old_row)
            shift = rng.randint(-2, 2)
            if shift != 0:
                new_row[1:] = [x + shift for x in new_row[1:]]
                # Check validity: strictly increasing and positive
                if new_row[1] > 0 and all(new_row[j] < new_row[j+1] for j in range(len(new_row)-1)):
                    # Check global validity
                    if _valid(child[:i] + [new_row] + child[i+1:], k):
                        old_scope = _scope(child)
                        child[i] = new_row
                        new_scope = _scope(child)
                        if new_scope <= old_scope or rng.random() < 0.1:
                            population.append(child)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = child
                                stagnation_count = 0
                            else:
                                stagnation_count += 1
                        else:
                            child[i] = old_row
                    else:
                        child[i] = old_row
        else:
            # Crossover: try to combine rows from two different parents
            if len(population) > 1:
                other_idx = rng.randint(0, len(population) - 1)
                if other_idx != (best_idx if len(population) > 2 else 0):
                    other_parent = population[other_idx]
                    # Try to build a child by picking rows from parent and other_parent
                    # This is hard to do directly due to conflicts.
                    # Instead, let's just swap a row from other_parent into child if it fits.
                    i = rng.randint(0, n_rows - 1)
                    # Check if other_parent[i] conflicts with child's other rows
                    used_mask = 0
                    for j in range(n_rows):
                        if j != i:
                            used_mask |= _row_to_bitset(child[j])
                    
                    other_row = other_parent[i]
                    other_mask = _row_to_bitset(other_row)
                    
                    if (used_mask & other_mask) == 0:
                        old_scope = _scope(child)
                        child[i] = list(other_row)
                        new_scope = _scope(child)
                        if new_scope <= old_scope or rng.random() < 0.1:
                            population.append(child)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = child
                                stagnation_count = 0
                            else:
                                stagnation_count += 1
                        else:
                            child[i] = parent[i]
        
        # Elitism: keep best N individuals
        population.sort(key=_scope)
        population = population[:pop_size]
        
        if stagnation_count > max_stagnation:
            # Restart with random perturbation
            for _ in range(5):
                i = rng.randint(0, n_rows - 1)
                used_mask = 0
                for j in range(n_rows):
                    if j != i:
                        used_mask |= _row_to_bitset(best_rows[j])
                new_row, _ = _greedy_row_with_mask(k, used_mask, rng)
                if new_row:
                    best_rows[i] = new_row
            population = [best_rows] * 5 + [list(r) for r in population[1:6]]
            stagnation_count = 0
            
    return best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 108.0

    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])

    rng = random.Random(42)

    # 1. Initial greedy
    greedy_rows = _fast_greedy(k, n_rows, rng, spread=False)
    consider(greedy_rows)
    
    # 2. Evolutionary Search
    if time.time() < _DEADLINE - 10:
        evo_rows = _evolutionary_search(n_rows, k, _DEADLINE - 5, rng)
        if evo_rows:
            consider(evo_rows)

    # 3. Final polish: simple local search on best
    if _BEST is not None and time.time() < _DEADLINE - 2:
        best_rows = [list(r) for r in _BEST[1]]
        # Try shifting elements
        for _ in range(1000):
            if time.time() > _DEADLINE - 2:
                break
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = best_rows[i][j]
            prev_val = best_rows[i][j-1]
            next_val = best_rows[i][j+1] if j < k else best_rows[i][-1] + 100
            lo = prev_val + 1
            hi = next_val - 1
            if lo > hi:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            best_rows[i][j] = new_val
            if _valid(best_rows, k):
                s = _scope(best_rows)
                if s < _BEST[0]:
                    _BEST = (s, [list(r) for r in best_rows])
            else:
                best_rows[i][j] = old_val

    if _BEST is not None:
        return _BEST[1]
    return greedy_rows
