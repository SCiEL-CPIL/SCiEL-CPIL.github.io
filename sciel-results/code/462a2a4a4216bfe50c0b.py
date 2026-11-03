import time
import random
import sys

def _greedy_fallback(n_rows, k):
    """Greedy construction as fallback."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _scope(rows):
    return max(max(row) for row in rows)

def _validate(rows):
    """Check if all differences across all rows are distinct."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _compute_used_diffs(rows):
    """Compute the set of all used differences."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(row[j] - row[i])
    return used

def _randomized_greedy(n_rows, k, rng, look_ahead=5):
    """
    Randomized greedy construction with look-ahead.
    Tries to pick candidates that leave more room for future choices.
    """
    used, rows = set(), []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Look ahead a window of candidates
            candidates = []
            # We need to find valid x > x_last
            # Let's scan from x+1 up to x+look_ahead*2 to be safe
            # But we must ensure we don't miss a valid one if it's further out?
            # Greedy usually works with small windows if we restart.
            # Let's just find the first valid one in a window, or extend window.
            
            # To make it randomized, we collect valid candidates in a window
            # and pick one.
            start_x = row[-1] + 1
            end_x = start_x + look_ahead
            
            found_any = False
            for cand in range(start_x, end_x + 1):
                new_diffs = {cand - a for a in row}
                if not new_diffs & used:
                    candidates.append(cand)
                    found_any = True
            
            if not found_any:
                # Extend search if no candidate in window
                # This should be rare for small k
                cand = start_x
                while True:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        candidates.append(cand)
                        found_any = True
                        break
                    cand += 1
                    if cand > 200: # Safety break
                        break
                
            if not found_any:
                # Fallback to standard greedy step
                cand = row[-1] + 1
                while True:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        candidates.append(cand)
                        break
                    cand += 1
            
            # Pick a candidate
            # Bias towards smaller values to keep scope low, but add randomness
            candidates.sort()
            # 80% chance pick the smallest, 20% pick randomly from top 3
            if rng.random() < 0.8 or len(candidates) == 1:
                chosen = candidates[0]
            else:
                chosen = candidates[rng.randrange(min(3, len(candidates)))]
            
            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Local search to reduce scope.
    Strategy: Iteratively try to decrease each entry to the minimum valid value.
    We do this in multiple passes.
    """
    best = [row[:] for row in rows]
    best_scope = _scope(best)
    
    # Pre-compute diffs for other rows to speed up checking
    # Actually, for n=7, k=5, total diffs is small. 
    # Let's just recompute the global set excluding the current row being modified.
    
    def get_other_diffs(exclude_idx):
        other_diffs = set()
        for i, row in enumerate(best):
            if i == exclude_idx:
                continue
            n = len(row)
            for a in range(n):
                for b in range(a + 1, n):
                    other_diffs.add(row[b] - row[a])
        return other_diffs

    improved = True
    last_improve_time = time.time()
    
    while time.time() < deadline:
        improved = False
        # Randomize row order to break symmetry
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for r_idx in row_order:
            row = best[r_idx]
            other_diffs = get_other_diffs(r_idx)
            
            # Try to minimize each position from right to left
            # This is crucial because decreasing a later element allows earlier ones to potentially decrease too?
            # Actually, decreasing an element at pos p affects diffs with all other elements.
            # If we decrease row[p], it might allow row[p+1] to decrease if it was constrained by row[p].
            # So right-to-left is good.
            
            for p in range(len(row) - 1, 0, -1):
                # Current value
                current_val = row[p]
                # Lower bound: must be > row[p-1]
                lower_bound = row[p-1] + 1
                # Upper bound: current value (we only try to decrease)
                
                # Try values from lower_bound to current_val - 1
                # We want the smallest valid value.
                # Let's iterate from lower_bound upwards and pick the first valid one.
                # If the first valid one is smaller than current_val, we update.
                
                best_candidate = current_val
                found_better = False
                
                # Optimization: If current_val is already lower_bound, we can't go lower.
                if current_val == lower_bound:
                    continue
                    
                for cand in range(lower_bound, current_val):
                    # Check if cand is valid
                    # New diffs involving cand:
                    # With elements before p: cand - row[i] for i < p
                    # With elements after p: row[j] - cand for j > p
                    
                    valid = True
                    new_diffs = set()
                    
                    for i in range(p):
                        d = cand - row[i]
                        if d in other_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                        
                    for j in range(p + 1, len(row)):
                        d = row[j] - cand
                        if d in other_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Check internal consistency of the row with the new cand
                        # We already checked against other_diffs.
                        # We need to ensure new_diffs don't collide with existing diffs in this row 
                        # that are NOT involving cand.
                        # Existing diffs in row not involving cand:
                        # row[i] - row[j] where i,j != p.
                        # Since we are only changing one element, the set of diffs 
                        # is (old_diffs - diffs_involving_old_cand) U new_diffs.
                        # We checked new_diffs against other_diffs.
                        # We need to check new_diffs against (old_diffs - diffs_involving_old_cand).
                        
                        # Let's compute the "static" diffs of the row (those not involving p)
                        static_diffs = set()
                        for i in range(len(row)):
                            if i == p: continue
                            for j in range(i + 1, len(row)):
                                if j == p: continue
                                static_diffs.add(row[j] - row[i])
                                
                        if new_diffs & static_diffs:
                            valid = False
                            continue
                            
                        if valid:
                            # This candidate is valid
                            # Since we are iterating from lower_bound upwards, 
                            # the first valid one is the smallest possible for this position 
                            # given the other elements are fixed.
                            if cand < current_val:
                                row[p] = cand
                                best_candidate = cand
                                found_better = True
                            break # We found the smallest valid one, no need to check higher ones in this range
                
                if found_better:
                    improved = True
                    last_improve_time = time.time()
                    # Update other_diffs cache? 
                    # get_other_diffs is called per row in the outer loop, 
                    # so it will be recalculated for the next row.
                    # For the same row, if we change row[p], it affects 
                    # the "other_diffs" for subsequent positions in the same row?
                    # No, get_other_diffs excludes the current row entirely.
                    # So the check for row[p+1] uses the updated row[p] in 'row' 
                    # but 'other_diffs' remains the same (diffs from other rows).
                    # This is correct.
        
        if not improved:
            # No improvement in a full pass.
            # Try a random perturbation to escape local minimum?
            # Or just break if we've been stuck for a while?
            # Given the small problem size, we can try a few random restarts of local search.
            # Let's just break to save time for other initial solutions.
            break
    
    return best, _scope(best)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    # 1. Generate initial solutions
    # We will generate a few diverse initial solutions
    initial_solutions = []
    
    # Standard greedy
    initial_solutions.append(_greedy_fallback(n_rows, k))
    
    # Randomized greedy with different look_aheads and seeds
    rng = random.Random(42)
    for i in range(10):
        la = 3 + (i % 5) # Vary look ahead
        sol = _randomized_greedy(n_rows, k, rng, la)
        initial_solutions.append(sol)
        
        if time.time() > deadline - 20:
            break
            
    # 2. Local search on each
    best_rows = None
    best_scope = float('inf')
    
    # Sort initial solutions by scope to prioritize good ones? 
    # Or just iterate through them.
    # Let's assign time budget to each.
    num_solutions = len(initial_solutions)
    time_per_solution = (deadline - time.time() - 5) / num_solutions if num_solutions > 0 else 0
    
    for sol in initial_solutions:
        if time.time() > deadline - 5:
            break
            
        sol_scope = _scope(sol)
        # If this initial solution is already worse than our best, and we have little time, skip?
        # No, local search might improve it significantly.
        
        # Set a sub-deadline for this local search
        sub_deadline = time.time() + time_per_solution
        
        improved_rows, improved_scope = _local_search(sol, n_rows, k, sub_deadline, rng)
        
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
            
        # If we found a solution very close to the theoretical lower bound, we can stop early
        # Lower bound: n_rows * k*(k+1)/2
        # For n=7, k=5: 7 * 15 = 105.
        # If best_scope == 105, we are optimal.
        if best_scope == 105:
            break

    # If no solution found (shouldn't happen), return greedy
    if best_rows is None:
        return _greedy_fallback(n_rows, k)
        
    return best_rows
