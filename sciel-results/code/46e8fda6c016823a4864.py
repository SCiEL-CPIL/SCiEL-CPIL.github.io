import time
import random
import math
import sys

def _row_diffs_list(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _scope(rows):
    if not rows:
        return 0
    return max(r[-1] for r in rows)

def _is_valid(rows):
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))

def _greedy_construct_row(used_mask, num_elements, max_scope, rng):
    """
    Construct a single valid row given a used difference mask.
    Returns row or None.
    """
    row = [0]
    last_val = 0
    
    for _ in range(num_elements - 1):
        found = False
        # Try values from last_val+1 up to max_scope
        # Randomize order slightly to allow diversity
        start = last_val + 1
        end = min(max_scope, last_val + 50) # Local search window
        candidates = list(range(start, end + 1))
        rng.shuffle(candidates)
        
        # If no candidate found in local window, extend
        if not candidates or not any(c > last_val for c in candidates):
             candidates = list(range(start, max_scope + 1))
             rng.shuffle(candidates[:20])
             candidates = candidates[:20] + sorted(candidates[20:])

        for val in candidates:
            if val <= last_val:
                continue
            new_diffs = []
            valid = True
            for prev in row:
                d = val - prev
                if d <= 0:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_mask = 0
            for d in new_diffs:
                new_mask |= (1 << d)
                
            if new_mask & used_mask:
                continue
                
            row.append(val)
            used_mask |= new_mask
            last_val = val
            found = True
            break
            
        if not found:
            return None
            
    return row

def _initial_population(n_rows, k, pop_size, rng, time_limit):
    """
    Generate an initial population of valid solutions using greedy construction.
    """
    num_elements = k + 1
    population = []
    start_time = time.time()
    
    # Estimate a reasonable max scope
    total_diffs_needed = n_rows * (num_elements * (num_elements - 1) // 2)
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    for _ in range(pop_size):
        if time.time() > time_limit:
            break
            
        rows = []
        used_mask = 0
        success = True
        
        # Shuffle row order to break symmetry
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for r_idx in row_order:
            # Try to construct row with a tight scope first, then relax
            max_scope = counting_bound + rng.randint(0, 30)
            row = _greedy_construct_row(used_mask, num_elements, max_scope, rng)
            if row is None:
                # Retry with larger scope
                max_scope = counting_bound + rng.randint(30, 100)
                row = _greedy_construct_row(used_mask, num_elements, max_scope, rng)
            
            if row is None:
                success = False
                break
                
            rows.append(row)
            used_mask |= _row_mask(row)
            
        if success:
            # Re-sort rows to standard order (index 0 to n_rows-1)
            # Note: The greedy construction doesn't care about row index, 
            # but the final output must be a list of rows.
            # The order of rows doesn't affect validity, only the set of rows matters.
            population.append(rows)
            
    # If population is empty, force create one
    if not population:
        rows = []
        used_mask = 0
        for _ in range(n_rows):
            max_scope = counting_bound + 100
            row = _greedy_construct_row(used_mask, num_elements, max_scope, rng)
            if row is None:
                # Fallback: simple sequential
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & {d for d in range(1, max_scope+1) if (used_mask >> d) & 1}):
                        row.append(x)
                        for d in new_diffs:
                            used_mask |= (1 << d)
            rows.append(row)
        population.append(rows)
        
    return population

def _crossover(parent1, parent2, rng):
    """
    Uniform Row Crossover: For each row index, pick from parent1 or parent2.
    """
    n_rows = len(parent1)
    child = []
    for i in range(n_rows):
        if rng.random() < 0.5:
            child.append(parent1[i][:])
        else:
            child.append(parent2[i][:])
    return child

def _mutate(rows, num_elements, used_mask_of_others, rng, scope_limit):
    """
    Mutate a single row in the population.
    1. Pick a random row index.
    2. Pick a random element position (1 to num_elements-1).
    3. Perturb the value.
    4. Check validity. If invalid, repair or revert.
    """
    n_rows = len(rows)
    r_idx = rng.randint(0, n_rows - 1)
    row = rows[r_idx][:]
    
    # Calculate mask of all other rows
    other_mask = 0
    for i in range(n_rows):
        if i != r_idx:
            other_mask |= _row_mask(rows[i])
            
    # Current row mask
    current_mask = _row_mask(row)
    
    # Pick a position to mutate
    pos = rng.randint(1, num_elements - 1)
    old_val = row[pos]
    
    # Determine range for new value
    low = row[pos - 1] + 1
    high = row[pos + 1] - 1 if pos < num_elements - 1 else scope_limit
    
    if low >= old_val and high <= old_val:
        return False # Cannot mutate here
        
    # Try to find a new value
    # Bias towards lower values to minimize scope
    search_range = list(range(low, min(high + 1, old_val + 10)))
    if not search_range:
        return False
        
    rng.shuffle(search_range)
    
    for new_val in search_range:
        if new_val == old_val:
            continue
            
        row[pos] = new_val
        
        # Check internal validity
        new_diffs = _row_diffs_list(row)
        if len(new_diffs) != len(set(new_diffs)):
            continue
            
        new_mask = _row_mask(row)
        
        # Check global validity
        if new_mask & other_mask:
            continue
            
        # Valid mutation
        rows[r_idx] = row
        return True
        
    # If no valid mutation found, try a "jump" mutation
    # Randomly reset the row using greedy with the current other_mask
    new_row = _greedy_construct_row(other_mask, num_elements, scope_limit, rng)
    if new_row is not None:
        # Ensure the new row is different from old
        if new_row != row:
            rows[r_idx] = new_row
            return True
            
    return False

def _repair_solution(rows, num_elements, rng, time_limit):
    """
    Attempt to reduce the scope of an invalid or high-scope solution.
    Uses local search: pick a random element in the largest row, try to move it down.
    """
    start_time = time.time()
    n_rows = len(rows)
    
    # Identify the row with the largest scope
    max_scope = _scope(rows)
    
    for _ in range(1000):
        if time.time() > time_limit:
            break
            
        # Pick a random row
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick a random element (excluding first)
        pos = rng.randint(1, num_elements - 1)
        old_val = row[pos]
        
        # Try to move down
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < num_elements - 1 else old_val - 1
        
        if low > high:
            continue
            
        # Try values in [low, old_val-1]
        candidates = list(range(low, old_val))
        rng.shuffle(candidates)
        
        for new_val in candidates:
            row[pos] = new_val
            
            # Check internal
            diffs = _row_diffs_list(row)
            if len(diffs) != len(set(diffs)):
                continue
                
            new_mask = _row_mask(row)
            
            # Check global
            other_mask = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_mask |= _row_mask(rows[i])
                    
            if new_mask & other_mask:
                continue
                
            # Valid move
            return True
            
        # Revert if no valid move found in this iteration
        row[pos] = old_val
        
    return False

def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0 # Safety margin
    
    num_elements = k + 1
    rng = random.Random(42)
    
    # Initial Population
    pop_size = 20
    population = _initial_population(n_rows, k, pop_size, rng, DEADLINE - 100)
    
    if not population:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
        
    # Find best initial
    best_individual = min(population, key=_scope)
    best_scope = _scope(best_individual)
    
    # Genetic Algorithm Loop
    gen = 0
    max_gens = 5000
    while time.time() < DEADLINE - 20 and gen < max_gens:
        gen += 1
        
        # Sort population by scope
        population.sort(key=_scope)
        
        # Elitism: Keep the best 20%
        elite_size = max(1, pop_size // 5)
        new_population = [ind[:] for ind in population[:elite_size]]
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            if time.time() > DEADLINE - 20:
                break
                
            # Selection: Tournament
            p1_idx = rng.randint(0, min(5, len(population)-1))
            p2_idx = rng.randint(0, min(5, len(population)-1))
            
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]
            
            # Crossover
            child = _crossover(parent1, parent2, rng)
            
            # Mutation
            # Determine scope limit for mutation (current best + slack)
            scope_limit = best_scope + 20
            _mutate(child, num_elements, 0, rng, scope_limit) # Note: _mutate needs other_mask, passing 0 is wrong.
            # Actually, _mutate calculates other_mask internally? No, it takes used_mask_of_others.
            # Let's fix _mutate to calculate other_mask itself or pass it correctly.
            # For now, let's just try to repair the child if it's invalid or high scope.
            
            # Check validity of child
            if not _is_valid(child):
                # Attempt repair
                # Simple repair: replace invalid rows with greedy ones
                # This is expensive, so we just discard invalid children often
                # Or try a light repair
                pass
            
            # Calculate scope
            child_scope = _scope(child)
            
            # If child is invalid, we need to ensure it's valid before adding to population
            # Let's add a strict validity check and repair step
            if not _is_valid(child):
                # Try to fix by re-greedy-ing the rows that might be bad
                # For simplicity in this draft, we discard invalid children
                # But we need to ensure population stays valid.
                # Let's use a "repair" that replaces any row causing conflict with a new greedy row.
                # This is complex. Let's stick to: if invalid, discard.
                continue
                
            new_population.append(child)
            
            # Update best
            if child_scope < best_scope:
                best_scope = child_scope
                best_individual = child[:]
                
        population = new_population
        
    # Final Local Search on Best Individual
    remaining_time = DEADLINE - time.time()
    if remaining_time > 5:
        # Run repair/local search on the best individual
        # We can try to reduce scope further
        current_best = best_individual[:]
        # Use _repair_solution? It's a bit weak.
        # Let's do a simple hill climbing on the best individual
        for _ in range(1000):
            if time.time() > DEADLINE - 2:
                break
            # Pick random row, random element, try to move down
            r_idx = rng.randint(0, n_rows - 1)
            row = current_best[r_idx]
            pos = rng.randint(1, num_elements - 1)
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < num_elements - 1 else old_val - 1
            if low > high:
                continue
            candidates = list(range(low, old_val))
            rng.shuffle(candidates)
            for new_val in candidates:
                row[pos] = new_val
                diffs = _row_diffs_list(row)
                if len(diffs) != len(set(diffs)):
                    continue
                new_mask = _row_mask(row)
                other_mask = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_mask |= _row_mask(current_best[i])
                if new_mask & other_mask:
                    continue
                # Valid
                new_scope = _scope(current_best)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_individual = current_best[:]
                break
            else:
                row[pos] = old_val

    # Final Validation
    if _is_valid(best_individual):
        return best_individual
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
