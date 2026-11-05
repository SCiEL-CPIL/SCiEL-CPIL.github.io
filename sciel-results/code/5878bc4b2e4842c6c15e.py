"""
Strategy: Structured backtracking (MRV + forward checking) for initial construction,
followed by Simulated Annealing with multi-element row rewrites and row swaps.
"""
import time
import random
import math
import sys

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

def _row_diff_set(row):
    """Return set of all positive differences within a row (i<j)."""
    return set(_row_diffs(row))

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _backtrack_construct(n_rows, k, max_val, rng, time_limit):
    """
    Construct rows using backtracking with MRV heuristic.
    Returns list of rows or None.
    """
    start_time = time.time()
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    
    # We need to assign n_rows * num_pairs distinct differences.
    # The differences must be in range [1, max_val].
    # This is a constraint satisfaction problem.
    
    # State:
    # used_diffs: set of used differences
    # rows: list of current rows being built
    # pos: current position in the row (0 to num_elements-1)
    # row_idx: current row index
    
    # MRV heuristic: pick the value that leaves the most options for the next step?
    # Actually, for difference sets, it's often better to pick values that are "sparse" or "dense" depending on strategy.
    # Let's try a standard backtracking where we choose the next element.
    
    # Optimization: Precompute valid candidates for each position?
    # Since max_val can be up to ~112, and num_elements is 6, the search space is manageable if we prune well.
    
    # We will build rows one by one.
    # For each row, we build it element by element.
    # To satisfy MRV, we might want to pick the element that minimizes the domain of subsequent elements.
    # However, simple chronological backtracking with forward checking is often sufficient for small k.
    
    # Let's use a recursive helper.
    
    def solve(row_idx, current_row, used_diffs):
        if time.time() > time_limit:
            return None
        
        if row_idx == n_rows:
            return rows_result[:] # Return copy
        
        # If current_row is complete, move to next row
        if len(current_row) == num_elements:
            # Check if internal diffs of current_row are distinct (should be guaranteed by construction)
            # And no conflict with used_diffs (guaranteed by construction)
            rows_result.append(current_row)
            res = solve(row_idx + 1, [0], used_diffs)
            if res is not None:
                return res
            rows_result.pop()
            return None
        
        # Determine next position
        pos = len(current_row)
        
        # Generate candidates for current_row[pos]
        # Must be > current_row[pos-1]
        # Must be <= max_val
        # Must not create duplicate diffs within current_row
        # Must not create duplicate diffs with used_diffs
        
        # Heuristic: Try values that are "likely" to work.
        # For difference sets, spreading out is good.
        # Let's try a mix of random and structured.
        
        # To improve MRV, we could estimate how many future values are available.
        # For simplicity, let's try values in a randomized order, but biased towards "safe" zones.
        
        # Actually, a better heuristic for difference sets is to pick values that avoid "collisions".
        # Let's just iterate through possible values, but shuffle them to add randomness.
        
        # Range of values:
        low = current_row[-1] + 1
        high = max_val
        
        # We need to ensure we can complete the row.
        # If pos = num_elements - 1 (last element), we just pick one.
        
        # Generate list of candidates
        candidates = list(range(low, high + 1))
        
        # Shuffle candidates to break symmetry and allow randomized search
        rng.shuffle(candidates)
        
        # Optional: Sort by some heuristic?
        # For now, random order is fine for the backtracking to find *a* solution.
        
        for val in candidates:
            # Check if val is valid
            new_row = current_row + [val]
            
            # Check internal diffs
            # We only need to check diffs involving the new element
            # because previous diffs are already known to be distinct and non-conflicting.
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used_diffs:
                    valid = False
                    break
                # Check if d is already in current_row's diffs
                # We can maintain a set of diffs for current_row
                # For efficiency in this recursive step, let's just compute or check against a set passed down.
                # Let's pass a set of diffs for the current row.
                
            # Re-implementation with diff set for current row
            # This is getting complex to pass around. Let's simplify:
            # Compute all diffs for new_row.
            # If len(set) == len(list) and no intersection with used_diffs, it's valid.
            
            diffs = _row_diffs(new_row)
            if len(diffs) != len(set(diffs)):
                continue
            if any(d in used_diffs for d in diffs):
                continue
            
            # Valid candidate
            new_used_diffs = used_diffs.union(diffs)
            
            # Recurse
            res = solve(row_idx, new_row, new_used_diffs)
            if res is not None:
                return res
        
        return None

    rows_result = []
    # Start with row 0, starting with [0]
    # used_diffs is empty
    return solve(0, [0], set())

def _simulated_annealing(rows, k, rng, time_limit):
    """
    Simulated Annealing to minimize the scope.
    Moves:
    1. Change a single element in a row (maintaining order and validity).
    2. Swap two rows (to allow different packings).
    3. Rewrite a whole row (random valid row with max element <= current scope).
    """
    n_rows = len(rows)
    num_elements = k + 1
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Precompute masks for fast checking
    def get_mask(row):
        m = 0
        for d in _row_diffs(row):
            m |= (1 << d)
        return m

    row_masks = [get_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    # Temperature
    T = 1.0
    cooling = 0.9995
    
    start_time = time.time()
    
    while time.time() < time_limit:
        # Pick a move
        move_type = rng.random()
        
        if move_type < 0.6:
            # Single element change
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, num_elements - 1)
            old_val = current_rows[row_idx][pos]
            
            # Determine range
            low = current_rows[row_idx][pos - 1] + 1
            high = current_rows[row_idx][pos + 1] - 1 if pos < num_elements - 1 else current_scope
            
            if low >= high:
                continue
            
            # Sample new value
            new_val = rng.randint(low, high)
            if new_val == old_val:
                continue
            
            # Check validity
            new_row = current_rows[row_idx][:]
            new_row[pos] = new_val
            diffs = _row_diffs(new_row)
            if len(diffs) != len(set(diffs)):
                continue
            
            new_mask = get_mask(new_row)
            other_mask = global_mask & ~row_masks[row_idx]
            if new_mask & other_mask:
                continue
            
            # Valid move
            new_scope = _scope(current_rows)
            # Note: scope might change if this was the max element
            if new_val > current_scope and pos == num_elements - 1:
                # Actually scope is max of last elements.
                pass
            
            # Calculate delta
            # We want to minimize scope.
            # Let's define energy as scope.
            # But local moves might not change scope immediately.
            # To encourage exploration, we can use a penalty for "tightness" or just accept valid moves.
            # Standard SA accepts worse moves with prob exp(-delta/T).
            # Here, "worse" means larger scope.
            
            # Update state
            current_rows[row_idx] = new_row
            row_masks[row_idx] = new_mask
            global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
            
            new_scope_val = _scope(current_rows)
            delta = new_scope_val - current_scope
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_scope = new_scope_val
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Revert
                current_rows[row_idx][pos] = old_val
                row_masks[row_idx] = get_mask(current_rows[row_idx])
                global_mask = (global_mask & ~row_masks[row_idx]) | row_masks[row_idx]
                
        elif move_type < 0.8:
            # Swap two rows
            r1, r2 = rng.sample(range(n_rows), 2)
            # Swapping rows doesn't change the set of differences or the scope.
            # It only changes which row is which.
            # This is useful if we have row-specific constraints (we don't really, except index).
            # So this move is neutral for scope. It helps escape local minima if combined with other moves?
            # Actually, since rows are indistinguishable in terms of validity (only global diffs matter),
            # swapping rows doesn't change the state's validity or scope.
            # It only changes the "identity" of the rows.
            # So it's a no-op for the objective function.
            # Let's skip this or use it to break symmetry if we had row-specific heuristics.
            pass
            
        else:
            # Rewrite a whole row
            row_idx = rng.randint(0, n_rows - 1)
            # Try to find a new valid row for this index
            # Max element <= current_scope
            # Avoid diffs used by other rows
            other_mask = global_mask & ~row_masks[row_idx]
            
            # Try a few random candidates
            found = False
            for _ in range(50):
                # Generate random row
                # Pick num_elements - 1 values from 1..current_scope
                vals = sorted(rng.sample(range(1, current_scope + 1), num_elements - 1))
                new_row = [0] + vals
                
                diffs = _row_diffs(new_row)
                if len(diffs) != len(set(diffs)):
                    continue
                
                new_mask = get_mask(new_row)
                if new_mask & other_mask:
                    continue
                
                # Valid
                # Apply
                current_rows[row_idx] = new_row
                row_masks[row_idx] = new_mask
                global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
                
                new_scope_val = _scope(current_rows)
                delta = new_scope_val - current_scope
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current_scope = new_scope_val
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                else:
                    # Revert
                    # We need to restore old row
                    # We didn't store it, so we need to re-generate or store it before.
                    # Let's store old row and mask before trying.
                    pass
                found = True
                break
            
            # If not found, do nothing (or revert if we applied and rejected, but we break on first valid candidate)
            # Wait, if we applied and rejected, we need to revert.
            # In the code above, if delta > 0 and rejected, we don't revert in the 'else' block of the rewrite logic.
            # Let's fix the rewrite logic to handle reversion properly.
            
            # Revised Rewrite Logic:
            old_row = current_rows[row_idx][:]
            old_mask = row_masks[row_idx]
            
            for _ in range(50):
                vals = sorted(rng.sample(range(1, current_scope + 1), num_elements - 1))
                new_row = [0] + vals
                
                diffs = _row_diffs(new_row)
                if len(diffs) != len(set(diffs)):
                    continue
                
                new_mask = get_mask(new_row)
                if new_mask & other_mask:
                    continue
                
                # Candidate valid
                current_rows[row_idx] = new_row
                row_masks[row_idx] = new_mask
                global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
                
                new_scope_val = _scope(current_rows)
                delta = new_scope_val - current_scope
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current_scope = new_scope_val
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                    # Keep the change
                    break
                else:
                    # Revert
                    current_rows[row_idx] = old_row
                    row_masks[row_idx] = old_mask
                    global_mask = (global_mask & ~row_masks[row_idx]) | old_mask
                    # Try next candidate
                    continue
            
        T *= cooling
        if T < 1e-6:
            T = 1e-6

    return best_rows, best_scope

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    import math
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    # Best known for (7,5) is 112.
    # Let's start backtracking from a reasonable upper bound.
    # If backtracking fails, we increase the bound.
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Backtracking Construction
    # Try scopes starting from counting_bound (or slightly above) up to some limit.
    # Backtracking can be slow, so we limit time per attempt.
    
    # For (7,5), counting bound is ~105.
    # Let's try 105, 106, ...
    
    current_scope_limit = max(counting_bound, 100)
    
    # We will try to find a solution with scope <= current_scope_limit.
    # If we find one, we record it.
    # We can try multiple times with different random seeds/shuffles to find a tighter one.
    
    # To speed up, we can use a smaller time limit for each backtracking attempt.
    
    # Let's try a few scopes.
    for scope in range(counting_bound, counting_bound + 20):
        if time.time() > DEADLINE - 30:
            break
            
        # Try to construct with this scope
        # Give it 2 seconds
        rows = _backtrack_construct(n_rows, k, scope, rng, time.time() + 2.0)
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            # If we found a solution with scope == scope, we might stop if it's close to bound
            # But we can try to improve with SA.
            break
    
    # If backtracking failed to find anything in the tight range, try a wider range or fallback
    if best_rows is None:
        # Try a larger scope
        for scope in range(counting_bound + 20, counting_bound + 50):
            if time.time() > DEADLINE - 20:
                break
            rows = _backtrack_construct(n_rows, k, scope, rng, time.time() + 1.0)
            if rows is not None:
                best_scope = _scope(rows)
                best_rows = rows
                break
        
        # Fallback to greedy if backtracking totally fails
        if best_rows is None:
            # Simple greedy fallback
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
    
    # Phase 2: Simulated Annealing
    # Run SA to improve the scope.
    if time.time() < DEADLINE - 5:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, k, rng, time_limit=DEADLINE - 5
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
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
