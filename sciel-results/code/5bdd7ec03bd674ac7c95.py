import time
import random
import heapq

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin before 150s kill
    row_len = k + 1
    
    # Precompute all possible row configurations up to a reasonable scope
    # Scope 112 is the known best, so we search up to maybe 150 to be safe, 
    # but we can prune based on the counting bound logic or just try to minimize.
    # Let's generate candidates for rows with max entry up to S_max.
    # S_max should be high enough to allow valid solutions, low enough to keep search space manageable.
    # The counting bound is 105. Best known 112. Let's set S_max to 130.
    S_max = 130
    
    def gen_rows(scope_limit):
        """Generate all valid rows (starting at 0, strictly increasing, length row_len) 
        with max entry <= scope_limit. Return list of (max_entry, bitmask, row_tuple)."""
        rows = []
        # We need to generate subsets of size row_len-1 from {1, ..., scope_limit}
        # such that 0 is included.
        # Actually, just iterate through combinations? 
        # C(130, 5) is huge (~150 million). Too slow to precompute all.
        
        # Instead, we generate rows on the fly or use a smarter generation.
        # For A*, we need to know the "cost" of adding a row.
        # The state is the used differences bitmask.
        # The transition is adding a row.
        # We need to enumerate possible next rows given the current used mask.
        
        pass

    # Since precomputing all rows is too expensive, we will generate candidate rows
    # that are "compatible" with the current used mask during the search.
    # However, generating compatible rows on the fly requires checking intersections,
    # which is fast if we use bitmasks.
    
    # Let's refine the A* approach:
    # State: (used_diffs_mask, num_rows_placed, current_max_scope)
    # Goal: num_rows_placed == n_rows
    # Cost: current_max_scope (we want to minimize the max scope of the entire set)
    # Actually, the cost is the max scope of the last row added, but we track the global max.
    # Since we want to minimize the global max, we can sort rows by their max scope.
    # If we add rows in increasing order of scope, the cost of a state is just the scope of the last added row.
    # This makes the heuristic admissible and consistent?
    # Heuristic h(n) = min_scope_needed_for_remaining_rows.
    # A simple admissible heuristic is 0 (Dijkstra), or based on counting bound.
    # Counting bound: total diffs needed = n_rows * C(k+1, 2).
    # Available diffs in scope S is S(S-1)/2.
    # So S must satisfy S(S-1)/2 >= n_rows * C(k+1, 2).
    # This gives a lower bound on S.
    
    # Let's implement a DFS with pruning and branch-and-bound, guided by the counting bound.
    # Or A* where we expand states with lowest "potential max scope".
    
    # Given the complexity of generating compatible rows dynamically, a hybrid approach might be better:
    # 1. Generate a pool of "good" rows (low scope) using the existing greedy/DFS logic.
    # 2. Use A* / Backtracking to select a subset of n_rows rows from this pool with disjoint diffs, minimizing max scope.
    
    # This is a Set Packing problem.
    # Let's stick to the directive: "Formulate the problem as a shortest path search in a state space defined by the bitmask of used differences... use A*".
    
    # To make A* feasible, we must limit the branching factor.
    # We can generate candidate rows for the next step only up to a certain scope limit, 
    # or prioritize rows with smaller scope.
    
    # Let's try a Beam Search / A* hybrid.
    # State: (used_mask, rows_placed, max_scope_so_far)
    # We expand by adding a row.
    # To avoid infinite loops and huge branching, we only consider rows with scope <= current_best_max_scope.
    # And we generate these rows on the fly, sorted by scope.
    
    def row_diffs_mask(row):
        mask = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                # d can be up to S_max. We need a bit for each d.
                # S_max=130 -> 130 bits. Python ints handle this fine.
                mask |= (1 << d)
        return mask

    def gen_rows_for_scope(scope, used_mask):
        """Generate all rows of length row_len, starting at 0, max <= scope,
        such that their diff mask is disjoint from used_mask.
        Returns list of (scope, mask, row_tuple)."""
        results = []
        # We need to choose row_len-1 numbers from 1..scope
        # such that all differences are not in used_mask.
        
        # Recursive generation
        def dfs(current_row, current_mask, last_val):
            if len(current_row) == row_len:
                results.append((current_row[-1], current_mask, tuple(current_row)))
                return
            # Next value must be > last_val and <= scope
            # Also, to ensure we don't exceed scope, we can prune if not enough room?
            # Not strictly necessary if we just check mask.
            
            # Optimization: if the number of remaining slots to fill is R, 
            # and we are at index i, the smallest possible max value is last_val + R.
            # The largest possible max value is scope.
            
            # We iterate x from last_val + 1 to scope
            # But we can prune if x - current_row[0] > scope? No, x <= scope is enforced.
            
            # To keep it fast, we can limit the number of candidates or use a heuristic order.
            # For A*, we want low scope rows first.
            
            rem = row_len - len(current_row)
            # Minimum next value is last_val + 1
            # Maximum next value is scope - (rem - 1) to allow strictly increasing sequence to fit in scope
            max_x = scope - (rem - 1)
            
            for x in range(last_val + 1, max_x + 1):
                # Check differences with all previous elements
                valid = True
                new_mask = current_mask
                for a in current_row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                if valid:
                    # Also check internal differences? 
                    # The loop above checks x against all a in current_row.
                    # Since current_row already has disjoint internal diffs (by induction), 
                    # and x's diffs with previous are disjoint from used_mask, 
                    # we must also ensure x's diffs with previous are disjoint from each other?
                    # No, diffs of x with a1 and a2 are x-a1 and x-a2. 
                    # They are distinct if a1 != a2. Since row is strictly increasing, a1 != a2.
                    # So we just need to check against used_mask.
                    # Wait, do we need to check against diffs already in current_mask?
                    # Yes! The new diffs must also be disjoint from the diffs already in this row.
                    # My loop `if (used_mask >> d) & 1` only checks global used.
                    # I need to check `if ((used_mask | current_mask) >> d) & 1`.
                    
                    pass # placeholder for correct logic below
            
        # Corrected DFS
        def dfs_correct(current_row, current_mask, last_val):
            if len(current_row) == row_len:
                results.append((current_row[-1], current_mask, tuple(current_row)))
                return
            
            rem = row_len - len(current_row)
            max_x = scope - (rem - 1)
            
            for x in range(last_val + 1, max_x + 1):
                valid = True
                new_mask = current_mask
                for a in current_row:
                    d = x - a
                    # Check against global used AND current row's mask
                    if ((used_mask | current_mask) >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if valid:
                    current_row.append(x)
                    dfs_correct(current_row, new_mask, x)
                    current_row.pop()
                    
        dfs_correct([0], 0, 0)
        return results

    # A* Search
    # Priority queue items: (max_scope, rows_placed, used_mask, rows_list)
    # We want to minimize max_scope.
    # Since we add rows, max_scope is non-decreasing.
    # If we always add the row with the smallest possible scope that is valid, 
    # we are doing a greedy/Dijkstra-like search.
    
    # However, generating `gen_rows_for_scope` is expensive if done for every state.
    # We can memoize or limit the search.
    
    # Alternative: Pre-compute a list of all valid rows up to S_max?
    # C(130, 5) is too big.
    
    # Let's restrict the search to a Beam Search with width W.
    # At each step, keep the W best states (lowest max_scope, then fewest used diffs? or just lowest max_scope).
    
    def beam_search(beam_width=100):
        # Initial state
        # We can start with any row. To be systematic, we can start with the smallest scope row.
        # Or just start with empty state and add first row.
        
        # State: (used_mask, rows_placed, max_scope, rows_list)
        # Start: (0, 0, 0, [])
        
        current_states = [(0, 0, 0, [])]
        
        for step in range(n_rows):
            next_states = []
            for used_mask, placed, max_scope, rows_list in current_states:
                # Generate candidate rows
                # We only care about rows with scope <= current_best_bound
                # What is current_best_bound? 
                # We can use the counting bound for the remaining rows.
                # Or just use S_max.
                
                # To reduce branching, we can limit the scope of new rows to some value.
                # If we are at step `step`, we have `n_rows - step - 1` rows left.
                # Let's try generating rows up to scope 112 (known best) or slightly higher.
                
                # Optimization: If we find a complete solution, we can update the global best bound.
                
                candidates = gen_rows_for_scope(S_max, used_mask)
                
                # Sort candidates by scope to prioritize smaller max_scope
                candidates.sort(key=lambda x: x[0])
                
                # Add each candidate as a new state
                for scope, mask, row in candidates:
                    new_max_scope = max(max_scope, scope)
                    # Prune: if new_max_scope >= current_best_global, skip?
                    # We don't have a global best yet in this function.
                    
                    next_states.append((mask | used_mask, placed + 1, new_max_scope, rows_list + [row]))
            
            # Prune next_states: keep only the best `beam_width` states
            # Sort by max_scope, then by number of set bits in mask (fewer used diffs is better for flexibility?)
            next_states.sort(key=lambda s: (s[2], bin(s[0]).count('1')))
            
            current_states = next_states[:beam_width]
            
            if not current_states:
                return None
            
            # Optional: Early termination if we found a solution with max_scope == lower_bound
            # Lower bound from counting:
            total_diffs_needed = n_rows * (row_len * (row_len - 1) // 2)
            # S(S-1)/2 >= total_diffs_needed
            # S approx sqrt(2 * total_diffs_needed)
            
        # Return the best state from the final beam
        if current_states:
            # The first one is the best
            return current_states[0][3]
        return None

    # The above beam search might be too slow if gen_rows_for_scope is called many times.
    # And gen_rows_for_scope itself is a DFS that can be slow.
    
    # Let's combine with the existing robust fallback.
    
    def greedy_row(used_mask):
        row = [0]; x = 0
        while len(row) < row_len:
            x += 1
            valid = True
            for a in row:
                d = x - a
                if (used_mask >> d) & 1:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]: # exclude x itself
                    used_mask |= (1 << (x - a))
        return row, used_mask

    def run_beam(beam_width):
        # Use beam search
        result = beam_search(beam_width)
        if result:
            return result
        return None

    # Fallback to the original strategy if beam search fails or is too slow
    def run_original():
        t_start = time.time()
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            if time.time() - t_start > 110:
                break
            row, used_mask = greedy_row(used_mask)
            rows.append(row)
        while len(rows) < n_rows:
            row, used_mask = greedy_row(used_mask)
            rows.append(row)
        return rows

    # Try beam search with different widths
    best_rows = None
    best_scope = float('inf')
    
    # Quick beam search attempt
    try:
        # Use a small beam width first to get a quick result
        res = run_beam(50)
        if res:
            scope = max(r[-1] for r in res)
            if scope < best_scope:
                best_scope = scope
                best_rows = res
    except Exception:
        pass

    # If no result or timeout, use original
    if best_rows is None or time.time() - t0 > 100:
        rows = run_original()
        if rows:
            scope = max(r[-1] for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows

    # If we have time, try to improve with a larger beam or local search
    if best_rows and time.time() - t0 < 110:
        # Try beam search with larger width
        try:
            res = run_beam(200)
            if res:
                scope = max(r[-1] for r in res)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = res
        except Exception:
            pass

    if best_rows:
        return best_rows
    else:
        # Last resort
        return run_original()
