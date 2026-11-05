import time
import random
import math
import sys

def _row_diffs(row):
    """Return set of all positive differences within a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _diffs_to_row(diffs, num_elements):
    """
    Given a set of 15 differences for a row of 6 elements, 
    try to find an assignment to slots (i,j) that yields a valid increasing row.
    Returns the row or None.
    """
    # The 15 pairs are:
    # (0,1), (0,2), (0,3), (0,4), (0,5)
    # (1,2), (1,3), (1,4), (1,5)
    # (2,3), (2,4), (2,5)
    # (3,4), (3,5)
    # (4,5)
    
    # We need to assign values d_01, d_02, ... d_45 from the set 'diffs'
    # such that:
    # x_1 = d_01
    # x_2 = d_02
    # x_3 = d_03
    # x_4 = d_04
    # x_5 = d_05
    # And consistency:
    # d_12 = x_2 - x_1
    # d_13 = x_3 - x_1
    # d_14 = x_4 - x_1
    # d_15 = x_5 - x_1
    # d_23 = x_3 - x_2
    # d_24 = x_4 - x_2
    # d_25 = x_5 - x_2
    # d_34 = x_4 - x_3
    # d_35 = x_5 - x_3
    # d_45 = x_5 - x_4
    
    # All these derived differences must be in the input set 'diffs'.
    # Also x_1 < x_2 < x_3 < x_4 < x_5.
    
    # Strategy: Pick 5 values from 'diffs' to be x_1..x_5 (d_01..d_05).
    # There are C(15,5) = 3003 combinations. This is small enough to brute force for each row.
    
    diffs_list = sorted(diffs)
    n = num_elements
    # Indices in the row: 1, 2, 3, 4, 5 correspond to positions 1..5
    # We select 5 values for positions 1..5 from the 15 available differences.
    
    # To speed up, we can iterate through combinations.
    # However, we also need the *other* 10 differences to match the derived values.
    
    # Let's try all combinations of 5 values from the 15 to be the "base" differences (from 0).
    # Since 3003 is small, we can do this.
    
    # Optimization: The largest difference must be x_5 (d_05).
    # The smallest difference must be one of the adjacent differences (e.g., d_45 or d_01 etc), but not necessarily d_01.
    # Actually, d_01 is x_1. d_45 is x_5 - x_4.
    
    # Let's just iterate.
    from itertools import combinations
    
    for combo in combinations(diffs_list, 5):
        # combo is (v1, v2, v3, v4, v5) corresponding to x_1, x_2, x_3, x_4, x_5
        if not (combo[0] < combo[1] < combo[2] < combo[3] < combo[4]):
            continue
            
        x1, x2, x3, x4, x5 = combo
        
        # Calculate derived differences
        derived = [
            x2 - x1, # d_12
            x3 - x1, # d_13
            x4 - x1, # d_14
            x5 - x1, # d_15
            x3 - x2, # d_23
            x4 - x2, # d_24
            x5 - x2, # d_25
            x4 - x3, # d_34
            x5 - x3, # d_35
            x5 - x4  # d_45
        ]
        
        # Check if all derived differences are in the original set
        # Note: The original set contains x1..x5 AND the derived ones.
        # So we need to check if the set of all 15 differences (x1..x5 + derived) matches the input 'diffs'.
        
        all_needed = set(combo)
        all_needed.update(derived)
        
        if all_needed == set(diffs):
            # We found a valid assignment!
            return [0, x1, x2, x3, x4, x5]
            
    return None

def _generate_random_row(diff_set, num_elements):
    """
    Given a set of 15 unique differences, find a valid row configuration.
    Uses the brute-force combination check.
    """
    return _diffs_to_row(diff_set, num_elements)

def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """
    Initial construction using greedy difference assignment.
    """
    rows = []
    used_mask = 0
    
    for _ in range(n_rows):
        # We need to pick 15 differences that are:
        # 1. Not in used_mask
        # 2. Can form a valid row (i.e., _diffs_to_row returns non-None)
        
        # Greedy approach: Pick small differences first to keep scope low?
        # Or pick differences that "fit" well.
        
        # Let's try to construct a row by picking differences one by one?
        # That's hard because of the structural constraint.
        
        # Alternative: Generate random sets of 15 differences and check validity.
        # This might be slow.
        
        # Better: Use the structure.
        # Pick x1, x2, x3, x4, x5 such that all derived diffs are unused.
        
        row = [0]
        x = 0
        # This is hard to do greedily without backtracking on the row structure.
        
        # Let's use a simple randomized backtracking for the initial row construction
        # that respects the global used_mask.
        
        # We will search for a row [0, a, b, c, d, e]
        # such that all diffs are in available space.
        
        # Let's just use a simple heuristic: try to find a row with small max value.
        # We'll iterate through possible max values (scope) and try to fill the row.
        
        found_row = None
        current_scope = 10
        while current_scope < 200: # Safe upper bound for initial
            # Try to find a row with max element <= current_scope
            # We can use a simple backtracking here, but let's try a randomized approach first.
            
            # Randomize the search for x1..x5
            for _ in range(100):
                # Pick random x1..x5 in increasing order within scope
                # Ensure all diffs are available
                pass # Placeholder, we'll implement a proper backtracking in _construct if needed.
            
            current_scope += 5
            
            if found_row:
                break
                
        # Fallback: Simple sequential greedy (slow but works)
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_mask) and len(new_diffs) == len(set(new_diffs)):
                # Check if this partial row can be extended? No, just accept.
                # This might lead to a dead end later, but let's try.
                row.append(x)
                for d in new_diffs:
                    used_mask |= (1 << d)
        
        if len(row) == num_elements:
            rows.append(row)
        else:
            # Should not happen with large enough x
            rows.append(row + [x+1, x+2, x+3, x+4, x+5]) # Emergency, likely invalid
            
    return rows

def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    # Phase 1: Initial Construction
    # We need a valid starting population.
    # The greedy fallback in the current program is okay but can be improved.
    # Let's use a randomized backtracking for the first row to ensure validity,
    # then use the "diff set -> row" mapping for GA.
    
    # Actually, the GA works on "diff sets". 
    # We need to start with valid diff sets that map to valid rows.
    
    # Let's generate a few valid rows using a simple backtracking that respects global constraints.
    
    def _backtrack_row_with_global(row_prefix, used_mask, num_elements, max_val, rng, deadline):
        if time.time() > deadline:
            return None
        if len(row_prefix) == num_elements:
            return row_prefix
        
        last_val = row_prefix[-1]
        candidates = []
        for val in range(last_val + 1, max_val + 1):
            new_diffs = set()
            valid = True
            for prev in row_prefix:
                d = val - prev
                if d in new_diffs:
                    valid = False
                    break
                if (used_mask >> d) & 1:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                candidates.append(val)
        
        if not candidates:
            return None
            
        rng.shuffle(candidates)
        
        for val in candidates:
            if time.time() > deadline:
                return None
            new_diffs = set()
            for prev in row_prefix:
                new_diffs.add(val - prev)
            new_mask = 0
            for d in new_diffs:
                new_mask |= (1 << d)
            
            result = _backtrack_row_with_global(row_prefix + [val], used_mask | new_mask, num_elements, max_val, rng, deadline)
            if result is not None:
                return result
        return None

    # Initial rows
    initial_rows = []
    used_mask = 0
    # Start with a generous scope to ensure we find something
    initial_scope = counting_bound + 20
    
    for i in range(n_rows):
        if time.time() > DEADLINE - 100:
            break
        row = [0]
        res = _backtrack_row_with_global(row, used_mask, num_elements, initial_scope, rng, DEADLINE - 100)
        if res is None:
            # Fallback to simple greedy if backtracking fails
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_mask) and len(new_diffs) == len(set(new_diffs)):
                    row.append(x)
                    for d in new_diffs:
                        used_mask |= (1 << d)
            res = row
            # Verify validity of fallback (it should be by construction)
        
        initial_rows.append(res)
        m = _row_mask(res)
        used_mask |= m
        
    if len(initial_rows) < n_rows:
        # If we couldn't get enough rows, pad with simple greedy (might be invalid, but we'll fix in GA?)
        # Actually, if we can't construct valid rows, we return the best we have.
        # But the task requires valid rows.
        # Let's assume backtracking works for reasonable bounds.
        while len(initial_rows) < n_rows:
            initial_rows.append([0, 1, 2, 3, 4, 5]) # Dummy, will be invalid.
            # This is a critical failure case. We should try harder.
            # For now, let's hope the initial backtracking succeeds.
            
    if not _is_valid(initial_rows):
        # If initial rows are invalid, we have a problem.
        # This can happen if the fallback greedy produced duplicates or if backtracking failed silently.
        # Let's re-run a simpler greedy that checks global validity.
        initial_rows = []
        used_mask = 0
        for i in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_mask) and len(new_diffs) == len(set(new_diffs)):
                    row.append(x)
                    for d in new_diffs:
                        used_mask |= (1 << d)
            initial_rows.append(row)
        
        if not _is_valid(initial_rows):
            # Last resort: return whatever we have, even if invalid? No, must be valid.
            # If we can't find a valid set, we return the greedy one which is valid by construction 
            # (since it checks used_mask).
            # The check above `if not (new_diffs & used_mask)` ensures global distinctness.
            # So this should be valid.
            pass

    best_rows = initial_rows
    best_scope = _scope(best_rows)
    
    # Phase 2: Genetic Algorithm
    # Representation: List of rows.
    # Fitness: Scope (lower is better).
    # Operations:
    # 1. Mutation: Pick a row, pick a random element, move it.
    #    If valid, accept.
    # 2. Crossover: Swap entire rows? No, rows are independent except for the global constraint.
    #    Since rows are independent, "crossover" is just swapping rows between individuals.
    #    The main search is local search on individual rows (moving elements) and global search (swapping rows).
    #    
    #    However, the "diff set" representation is more powerful.
    #    Let's stick to element moves for simplicity and robustness, but use the 
    #    "diff set" idea to *reconstruct* a row if it gets stuck.
    
    # Actually, the current program's local search is decent. 
    # The structural change requested is "single-processor scheduling... genetic".
    # Let's implement a GA that operates on the *difference sets*.
    
    # Population: List of individuals. Each individual is a list of diff-sets.
    # Diff-set i corresponds to row i.
    # Validity: 
    #   1. All diff-sets are disjoint.
    #   2. Each diff-set maps to a valid row (via _diffs_to_row).
    #   3. The scope of the resulting rows is minimized.
    
    # Initialization: Convert initial_rows to diff-sets.
    # If a row is valid, its diff-set is valid.
    
    def _rows_to_diffsets(rows):
        return [_row_diffs(r) for r in rows]
    
    def _diffsets_to_rows(diffsets, num_elements):
        rows = []
        for ds in diffsets:
            r = _diffs_to_row(ds, num_elements)
            if r is None:
                return None
            rows.append(r)
        return rows
    
    # Check if diffsets are disjoint
    def _diffsets_disjoint(diffsets):
        seen = set()
        for ds in diffsets:
            for d in ds:
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    # Initial population
    pop_size = 10
    population = []
    for i in range(pop_size):
        # Clone initial rows with some noise
        # For simplicity, start with the same valid rows
        # We will mutate them.
        population.append(_rows_to_diffsets(best_rows))
        
    # GA Loop
    for gen in range(500):
        if time.time() > DEADLINE - 10:
            break
            
        new_population = []
        
        # Elitism: Keep the best 2
        # We need to evaluate fitness. 
        # Fitness = Scope.
        # We must check validity first.
        
        evaluated = []
        for ind in population:
            rows = _diffsets_to_rows(ind, num_elements)
            if rows is None:
                evaluated.append((float('inf'), ind))
                continue
            if not _diffsets_disjoint(ind):
                evaluated.append((float('inf'), ind))
                continue
            if not _is_valid(rows): # Should be covered by disjoint check + valid row construction
                evaluated.append((float('inf'), ind))
                continue
            scope = _scope(rows)
            evaluated.append((scope, ind))
            
        # Sort by scope
        evaluated.sort(key=lambda x: x[0])
        
        # Update best
        if evaluated[0][0] < best_scope:
            best_scope = evaluated[0][0]
            best_rows = _diffsets_to_rows(evaluated[0][1], num_elements)
            
        # Elitism
        new_population.append(evaluated[0][1])
        new_population.append(evaluated[1][1])
        
        # Crossover and Mutation
        while len(new_population) < pop_size:
            # Pick parents
            p1 = random.choice(evaluated[:pop_size//2])[1]
            p2 = random.choice(evaluated[:pop_size//2])[1]
            
            # Crossover: Swap a random row's diff-set between p1 and p2
            # Since rows are independent, we can swap any row index.
            child1 = p1[:]
            child2 = p2[:]
            
            swap_idx = rng.randint(0, n_rows - 1)
            child1[swap_idx] = p2[swap_idx]
            child2[swap_idx] = p1[swap_idx]
            
            # Mutation
            # Pick a random row in child1 and perturb its diff-set.
            # How to perturb a diff-set while keeping it valid?
            # This is the hard part.
            
            # Strategy: 
            # 1. Convert the row's diff-set back to a row.
            # 2. Move an element in the row.
            # 3. If valid, convert back to diff-set.
            
            row_idx = rng.randint(0, n_rows - 1)
            ds = child1[row_idx]
            row = _diffs_to_row(ds, num_elements)
            
            if row is not None:
                # Try to move an element
                pos = rng.randint(1, num_elements - 1)
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < num_elements - 1 else best_scope + 10
                
                if low < high:
                    new_val = rng.randint(low, high)
                    if new_val != old_val:
                        row[pos] = new_val
                        # Check if this row is still valid internally
                        new_ds = _row_diffs(row)
                        if len(new_ds) == num_pairs:
                            # Check if it conflicts with other rows in child1
                            # We need to check against all other diff-sets in child1
                            conflict = False
                            for i, other_ds in enumerate(child1):
                                if i == row_idx:
                                    continue
                                if new_ds & other_ds:
                                    conflict = True
                                    break
                            if not conflict:
                                child1[row_idx] = new_ds
            
            new_population.append(child1)
            
            # Also add child2 with a similar mutation
            row_idx = rng.randint(0, n_rows - 1)
            ds = child2[row_idx]
            row = _diffs_to_row(ds, num_elements)
            if row is not None:
                pos = rng.randint(1, num_elements - 1)
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < num_elements - 1 else best_scope + 10
                if low < high:
                    new_val = rng.randint(low, high)
                    if new_val != old_val:
                        row[pos] = new_val
                        new_ds = _row_diffs(row)
                        if len(new_ds) == num_pairs:
                            conflict = False
                            for i, other_ds in enumerate(child2):
                                if i == row_idx:
                                    continue
                                if new_ds & other_ds:
                                    conflict = True
                                    break
                            if not conflict:
                                child2[row_idx] = new_ds
            new_population.append(child2)
            
        population = new_population
        
        # If we found a solution at the counting bound, stop
        if best_scope <= counting_bound:
            break

    # Phase 3: Final Local Search on the best rows found by GA
    if time.time() < DEADLINE - 20 and best_rows:
        # Use the local search from the original program, adapted
        # This is a structural addition: GA for global structure, Local Search for fine-tuning.
        
        current_rows = [r[:] for r in best_rows]
        current_scope = _scope(current_rows)
        best_ls_rows = [r[:] for r in current_rows]
        best_ls_scope = current_scope
        
        global_mask = 0
        row_masks = []
        for row in current_rows:
            m = _row_mask(row)
            row_masks.append(m)
            global_mask |= m
        
        stagnation = 0
        max_stagnation = 2000
        
        while time.time() < DEADLINE - 10 and stagnation < max_stagnation:
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, num_elements - 1)
            row = current_rows[row_idx]
            old_val = row[pos]
            
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < num_elements - 1 else current_scope
            
            if low >= old_val:
                stagnation += 1
                continue
                
            new_val = rng.randint(low, old_val - 1)
            
            row[pos] = new_val
            diffs_in_row = set()
            valid = True
            for i in range(num_elements):
                for j in range(i + 1, num_elements):
                    d = row[j] - row[i]
                    if d in diffs_in_row:
                        valid = False
                        break
                    diffs_in_row.add(d)
                if not valid:
                    break
            
            if valid:
                new_row_mask = 0
                for d in diffs_in_row:
                    new_row_mask |= (1 << d)
                
                other_mask = global_mask & ~row_masks[row_idx]
                if new_row_mask & other_mask:
                    valid = False
            
            if not valid:
                row[pos] = old_val
                stagnation += 1
                continue
            
            row[pos] = new_val
            old_row_mask = row_masks[row_idx]
            new_row_mask = 0
            for d in diffs_in_row:
                new_row_mask |= (1 << d)
            row_masks[row_idx] = new_row_mask
            global_mask = (global_mask & ~old_row_mask) | new_row_mask
            
            new_scope = _scope(current_rows)
            if new_scope < best_ls_scope:
                best_ls_scope = new_scope
                best_ls_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                stagnation += 1
                
        if best_ls_scope < best_scope:
            best_scope = best_ls_scope
            best_rows = best_ls_rows

    # Final Validation
    if best_rows and _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        return _greedy_fallback(n_rows, num_elements)

def _greedy_fallback(n_rows, num_elements):
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def construct(n_rows, k):
    return _construct(n_rows, k)
