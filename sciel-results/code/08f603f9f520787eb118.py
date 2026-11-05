import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    lower_bound = total_diffs_needed  # minimum scope needed
    
    best_scope = float('inf')
    best_rows = None
    
    # Precompute: for each possible row (tuple of k+1 increasing ints starting at 0),
    # store the set of differences it produces.
    # We'll use a different approach: generate rows on the fly but use bitmask for used diffs.
    
    # For n_rows=7, k=5: each row has 6 elements, 15 differences per row, 105 total.
    # Scope is at least 105 (counting bound). Best known is 112.
    
    # Strategy: Use backtracking with constraint propagation.
    # Represent used differences as a set. For each row, we need to find a valid row
    # whose differences don't conflict with used diffs.
    
    # Key insight: Instead of building row by row greedily, use a smarter search.
    # We'll try multiple strategies:
    # 1. Backtracking with MRV (minimum remaining values)
    # 2. Local search / simulated annealing on complete solutions
    # 3. Hybrid: greedy initialization + local search refinement
    
    # For speed, we'll represent the "used" set as a Python set of ints.
    # Each row is a tuple of k+1 integers starting with 0.
    
    def _row_diffs(row):
        """Return the set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _is_valid_rows(rows):
        """Check if all rows have distinct differences across all rows."""
        used = set()
        for row in rows:
            diffs = _row_diffs(row)
            if used & diffs:
                return False
            used |= diffs
        return True
    
    def _scope_of_rows(rows):
        """Return the maximum entry across all rows."""
        return max(max(r) for r in rows)
    
    # Strategy 1: Backtracking with randomized restarts
    def _backtrack_search(n_rows, k, max_scope, rng, time_limit):
        """
        Backtracking search for a valid set of n_rows rows with scope <= max_scope.
        Uses randomized ordering for restarts.
        """
        used = set()
        rows = []
        
        def _gen_candidates(used, max_scope, rng):
            """Generate candidate rows that don't conflict with used diffs."""
            candidates = []
            # Generate random rows within scope
            attempts = 0
            max_attempts = 5000
            while len(candidates) < 50 and attempts < max_attempts:
                # Generate a random increasing sequence of k+1 elements starting at 0
                # within [0, max_scope]
                if k == 0:
                    row = (0,)
                else:
                    # Choose k values from 1..max_scope
                    vals = sorted(rng.sample(range(1, max_scope + 1), k))
                    row = (0,) + tuple(vals)
                
                diffs = _row_diffs(row)
                if not (diffs & used):
                    candidates.append((row, diffs))
                attempts += 1
            
            return candidates
        
        def _search(depth, used, rows):
            if time.time() > time_limit:
                return False
            
            if depth == n_rows:
                return True
            
            # Get candidates
            candidates = _gen_candidates(used, max_scope, rng)
            
            if not candidates:
                return False
            
            # Shuffle for randomness
            rng.shuffle(candidates)
            
            # Try each candidate
            for row, diffs in candidates:
                rows.append(row)
                used.update(diffs)
                
                if _search(depth + 1, used, rows):
                    return True
                
                rows.pop()
                used.difference_update(diffs)
            
            return False
        
        return _search(0, used, rows)
    
    # Strategy 2: Greedy construction with local search
    def _greedy_construct(n_rows, k, rng, c=20):
        """Greedy construction: pick each row to minimize future conflicts."""
        used = set()
        rows = []
        
        for _ in range(n_rows):
            best_row = None
            best_score = float('inf')
            
            # Try many random rows, pick the one that minimizes "damage"
            # (i.e., the one whose diffs are most "spread out" or cause least future conflict)
            for _ in range(200):
                if k == 0:
                    row = (0,)
                else:
                    # Generate random row with some structure
                    # Bias toward smaller values but allow some spread
                    max_val = rng.randint(k + 1, 150)
                    vals = sorted(rng.sample(range(1, max_val + 1), k))
                    row = (0,) + tuple(vals)
                
                diffs = _row_diffs(row)
                if used & diffs:
                    continue
                
                # Score: prefer rows with smaller max value, and diffs that are "well-spread"
                score = max(row)  # Primary: minimize scope
                # Secondary: prefer diffs that are not clustered
                diff_list = sorted(diffs)
                if len(diff_list) > 1:
                    # Penalize clustering
                    gaps = [diff_list[i+1] - diff_list[i] for i in range(len(diff_list)-1)]
                    score -= 0.01 * sum(gaps)
                
                if score < best_score:
                    best_score = score
                    best_row = (row, diffs)
            
            if best_row is None:
                # Fallback: find any valid row
                max_val = 200
                while True:
                    if k == 0:
                        row = (0,)
                    else:
                        vals = sorted(rng.sample(range(1, max_val + 1), k))
                        row = (0,) + tuple(vals)
                    diffs = _row_diffs(row)
                    if not (used & diffs):
                        best_row = (row, diffs)
                        break
                    max_val += 10
            
            row, diffs = best_row
            rows.append(row)
            used.update(diffs)
        
        return rows
    
    def _local_search(rows, n_rows, k, rng, time_limit):
        """Local search: try to reduce scope by modifying individual elements."""
        current_rows = [list(r) for r in rows]
        current_scope = _scope_of_rows(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_scope = current_scope
        
        t = 5.0
        cooling = 0.995
        no_improve = 0
        
        while time.time() < time_limit and t > 0.01:
            # Pick a random element to modify
            row_idx = rng.randint(0, n_rows - 1)
            pos_idx = rng.randint(1, k)  # Don't modify the first element (always 0)
            
            old_val = current_rows[row_idx][pos_idx]
            
            # Determine valid range
            lower = current_rows[row_idx][pos_idx - 1] + 1
            upper = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val - 1
            
            if upper <= lower:
                t *= cooling
                continue
            
            # Try to move to a smaller value (bias toward reduction)
            if rng.random() < 0.7:
                search_range = min(30, old_val - lower)
                if search_range > 0:
                    new_val = rng.randint(lower, min(upper, old_val - 1))
                else:
                    t *= cooling
                    continue
            else:
                new_val = rng.randint(lower, upper)
            
            if new_val == old_val:
                t *= cooling
                continue
            
            # Check validity
            current_rows[row_idx][pos_idx] = new_val
            
            # Check if still valid
            valid = True
            used = set()
            for row in current_rows:
                diffs = _row_diffs(row)
                if used & diffs:
                    valid = False
                    break
                used |= diffs
            
            if not valid:
                current_rows[row_idx][pos_idx] = old_val
                t *= cooling
                continue
            
            new_scope = _scope_of_rows(current_rows)
            delta = new_scope - current_scope
            
            if delta < 0 or (delta == 0 and rng.random() < 0.5) or (delta > 0 and rng.random() < math.exp(-delta / t)):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_rows[row_idx][pos_idx] = old_val
                no_improve += 1
            
            t *= cooling
            
            if no_improve > 200:
                break
        
        return best_rows, best_scope
    
    # Strategy 3: Iterative deepening on scope
    def _iterative_deepening(n_rows, k, rng, time_limit):
        """Try increasing scopes until we find a valid solution."""
        # Start from the lower bound
        lb = n_rows * (k + 1) * k // 2
        scope = lb
        
        while time.time() < time_limit:
            # Try to find a solution with this scope
            found = False
            for attempt in range(20):  # Multiple attempts per scope
                if time.time() > time_limit:
                    break
                
                rows = _greedy_construct(n_rows, k, rng, c=30)
                if rows and _is_valid_rows(rows):
                    s = _scope_of_rows(rows)
                    if s <= scope:
                        found = True
                        break
                    else:
                        # Got a valid solution but scope too high, keep trying
                        pass
            
            if found:
                # Polish it
                polished, polished_scope = _local_search(rows, n_rows, k, rng, 
                                                         min(time_limit, time.time() + 10))
                return polished, polished_scope
            
            scope += 1
        
        return None, float('inf')
    
    # Main search loop
    # First, try greedy + local search with multiple restarts
    seed = 0
    while time.time() < deadline - 10:
        rng = random.Random(seed)
        
        # Greedy construction
        rows = _greedy_construct(n_rows, k, rng, c=25)
        
        if rows is None:
            seed += 1
            continue
        
        # Check validity
        if not _is_valid_rows(rows):
            seed += 1
            continue
        
        scope = _scope_of_rows(rows)
        
        # Local search to improve
        if time.time() < deadline - 5:
            polished, polished_scope = _local_search(rows, n_rows, k, rng, 
                                                     min(deadline - 2, time.time() + 15))
            if polished_scope < scope:
                rows = polished
                scope = polished_scope
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
            # If we hit the lower bound, stop
            if best_scope <= lower_bound:
                break
        
        seed += 1
    
    # If we haven't found anything, try backtracking
    if best_rows is None and time.time() < deadline - 5:
        for scope in range(lower_bound, lower_bound + 30):
            if time.time() > deadline - 5:
                break
            rng = random.Random(scope)
            found = _backtrack_search(n_rows, k, scope, rng, time.time() + 10)
            if found:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                break
    
    # Final fallback
    if best_rows is None:
        # Simple greedy
        used = set()
        best_rows = []
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
            best_rows.append(row)
        best_scope = _scope_of_rows(best_rows)
    
    # Ensure we return lists of ints
    return [[int(x) for x in row] for row in best_rows]
