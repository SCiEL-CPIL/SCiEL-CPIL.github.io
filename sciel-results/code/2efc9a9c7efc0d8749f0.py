"""
Strategy: Greedy Set Packing with Forward Checking.

Instead of local search on element positions, we treat the problem as selecting
a set of rows (each defined by its set of internal differences) such that no
difference is repeated.
1. Pre-compute all valid difference sets (signatures) for rows of size k+1
   within a scope M. A signature is valid if all internal differences are distinct.
2. We want to pick n_rows signatures such that their union of differences is disjoint.
3. We use a greedy approach with forward checking (backtracking) to select these
   signatures. We iterate M (scope) from the lower bound upwards.
4. For a fixed M, we generate candidate rows. To avoid explosion, we prioritize
   rows that use "rare" differences or simply use a randomized greedy with restarts.
   
This is structurally different from local search because it operates on the 
discrete space of valid difference sets rather than continuous element positions.
"""

import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _generate_candidates(n_rows, k, max_scope, rng):
    """
    Generate a list of candidate rows (tuples of elements) that are internally valid.
    Each row starts with 0 and has k+1 elements.
    We limit the number of candidates to keep memory and time manageable.
    """
    num_elements = k + 1
    candidates = []
    
    # We want to generate diverse rows.
    # A row is defined by k-1 internal values x_1 < ... < x_{k-1} in 1..max_scope.
    # The last element is max_scope? No, the last element is just some value <= max_scope.
    # Actually, to minimize scope, we want to pack tightly.
    # Let's generate rows where the last element is exactly max_scope? 
    # Not necessarily. But for a fixed scope M, any valid row with max element <= M is a candidate.
    
    # To make it efficient, we will sample random subsets of {1, ..., max_scope} of size k-1.
    # We will also include rows that are "tight" (small differences).
    
    # Strategy: 
    # 1. Generate some random rows.
    # 2. Generate some greedy rows (smallest possible values).
    
    num_candidates = 2000  # Limit candidates to avoid memory issues
    
    # Greedy rows: try to fill with smallest possible numbers
    # This is hard to do globally, so we just do random sampling with bias.
    
    for _ in range(num_candidates):
        # Sample k-1 values from 1..max_scope
        if k - 1 > max_scope:
            continue
        vals = sorted(rng.sample(range(1, max_scope + 1), k - 1))
        row = [0] + vals
        # Check internal validity
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)):
            continue
        
        # Ensure max element <= max_scope (it is by construction)
        candidates.append(tuple(row))
        
        if len(candidates) >= num_candidates:
            break
            
    # Also add a few "tight" rows:
    # Try to construct rows with small max element
    # Simple heuristic: try to find rows with small last element
    # We can just run a small local search on a single row to minimize its max element?
    # Or just sample from a smaller range.
    
    # Let's add candidates from range 1..max_scope//2
    for _ in range(num_candidates // 2):
        if k - 1 > max_scope // 2:
            continue
        vals = sorted(rng.sample(range(1, max_scope // 2 + 1), k - 1))
        row = [0] + vals
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)):
            continue
        candidates.append(tuple(row))
        
    return candidates

def _greedy_set_packing(n_rows, k, max_scope, rng, max_restarts=50):
    """
    Try to find n_rows non-conflicting rows from the candidate pool.
    Uses randomized greedy with restarts.
    """
    num_elements = k + 1
    
    for restart in range(max_restarts):
        # Generate candidates for this restart (maybe different random seed or just reuse)
        # To save time, generate once per scope, but shuffle for each restart?
        # Let's generate a fresh pool for each restart to ensure diversity in selection.
        # But generation is expensive. 
        # Compromise: Generate a large pool once, then sample from it.
        
        # Actually, let's generate candidates inside the restart loop if it's not too slow.
        # Or generate a static pool and randomize selection order.
        
        # Let's generate a pool of candidates.
        candidates = _generate_candidates(n_rows, k, max_scope, rng)
        
        if not candidates:
            return None
            
        # Pre-compute difference masks for candidates
        # mask is an integer bitmask of differences
        # max difference is max_scope
        # We need a bitset of size max_scope+1
        # Python integers are arbitrary precision, so this is fine.
        
        cand_info = []
        for row in candidates:
            diffs = _row_diffs(row)
            # Check if diffs are distinct (already done in generation, but double check)
            if len(diffs) != len(set(diffs)):
                continue
            mask = 0
            for d in diffs:
                mask |= (1 << d)
            cand_info.append((row, mask))
            
        if len(cand_info) < n_rows:
            return None
            
        # Shuffle candidates to randomize greedy choice
        rng.shuffle(cand_info)
        
        selected_rows = []
        used_mask = 0
        
        for i in range(n_rows):
            found = False
            for row, mask in cand_info:
                if mask & used_mask:
                    continue
                # Check if this row can be added
                # It can if no conflict
                selected_rows.append(row)
                used_mask |= mask
                found = True
                break
            
            if not found:
                return None
                
        return [list(r) for r in selected_rows]

def _local_search(rows, num_elements, max_scope, rng, time_limit):
    """
    Local search to reduce scope.
    Same as before, but we might call it less often or with different params.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m

    n_rows = len(current_rows)
    k = num_elements - 1

    def _row_mask(row):
        m = 0
        for d in _row_diffs(row):
            m |= (1 << d)
        return m

    max_iter = 20000
    for it in range(max_iter):
        if time.time() > time_limit:
            break

        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, len(current_rows[row_idx]) - 1)
        old_val = current_rows[row_idx][pos]

        low = current_rows[row_idx][pos - 1] + 1
        high = current_rows[row_idx][pos + 1] - 1 if pos < len(current_rows[row_idx]) - 1 else max_scope

        if low >= high:
            continue

        if rng.random() < 0.7:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)

        if new_val == old_val:
            continue

        current_rows[row_idx][pos] = new_val
        row = current_rows[row_idx]
        diffs = _row_diffs(row)
        valid = len(diffs) == len(set(diffs))
        if valid:
            new_mask = _row_mask(row)
            other_mask = global_mask & ~row_masks[row_idx]
            valid = (new_mask & other_mask) == 0
        if valid:
            row_masks[row_idx] = _row_mask(row)
            global_mask = (global_mask & ~row_masks[row_idx]) | row_masks[row_idx]
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                max_scope = best_scope
        else:
            current_rows[row_idx][pos] = old_val

    return best_rows, best_scope

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs

    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    # Start from a reasonable bound
    # For (7,5), counting bound is ~105. Best known 112.
    # Let's start from counting_bound.
    initial_scope = max(counting_bound, 50)
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try to find a solution with increasing scope using Set Packing
    # We try scopes from initial_scope upwards.
    # If we find one, we try to improve it with local search.
    
    # To avoid getting stuck in a bad scope, we try a few scopes.
    # If we find a solution at scope M, we record it.
    # Then we try M-1? No, we iterate upwards.
    # But we want the MINIMUM scope.
    # So we should iterate from lower bound UP.
    # If we find a solution at M, we can stop the search for that M?
    # No, there might be a solution at M-1 we missed.
    # But we are iterating UP. So the first M where we find a solution is the best we can find with this method?
    # Not necessarily, because the search is stochastic.
    # So we should try multiple restarts for each M.
    
    # Strategy:
    # For M in range(initial_scope, initial_scope + 20):
    #   Try to find solution with scope M.
    #   If found, update best.
    #   If found, break? Or continue to see if we can find a better one?
    #   Since we are going up, the first one found is the smallest M for which we succeeded.
    #   However, we might fail at M=105 but succeed at M=106.
    #   We want the smallest M.
    #   So we should try M=105 with many restarts. If fail, try M=106.
    
    max_scope_limit = initial_scope + 30  # Don't go too far
    
    for M in range(initial_scope, max_scope_limit + 1):
        if time.time() > DEADLINE - 30:
            break
            
        # Try to find a solution with scope M
        # We use the set packing method
        rows = _greedy_set_packing(n_rows, k, M, rng, max_restarts=20)
        
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            
            # If we found a solution, we can try to improve it with local search
            # But first, let's see if we can find a solution with a smaller scope?
            # No, we are iterating upwards. So this is the smallest M we've tried so far.
            # Wait, if we found a solution at M, it means there EXISTS a solution with scope <= M.
            # The actual scope of the found rows might be less than M.
            # So we should record the actual scope.
            
            # Should we continue to larger M?
            # No, because we want to MINIMIZE scope.
            # If we found a solution with scope s <= M, and we are iterating M upwards,
            # any future M' > M will likely yield solutions with scope >= s (or we might find a better one?
            # Unlikely, because the search space is larger, but the constraint is tighter for smaller M).
            # Actually, if we find a solution at M=105, we should stop trying larger M.
            # We should then try to LOCAL SEARCH to reduce the scope of this solution.
            
            break
            
    # If no solution found in the range, try a larger range with more restarts
    if best_rows is None:
        for M in range(max_scope_limit, max_scope_limit + 50):
            if time.time() > DEADLINE - 30:
                break
            rows = _greedy_set_packing(n_rows, k, M, rng, max_restarts=100)
            if rows is not None:
                best_rows = rows
                best_scope = _scope(rows)
                break
                
    # Fallback
    if best_rows is None:
        # Sequential construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)

    # Phase 2: Local Search
    while time.time() < DEADLINE - 10:
        improved_rows, improved_scope = _local_search(
            best_rows, num_elements, best_scope, rng,
            time_limit=DEADLINE - 15
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        else:
            # If no improvement, break to save time?
            # Or try a few more times?
            break
            
    # Phase 3: Final Validation
    if _is_valid(best_rows):
        return best_rows
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
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
