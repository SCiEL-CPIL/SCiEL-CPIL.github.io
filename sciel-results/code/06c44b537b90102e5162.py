"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of length k+1,
    starting at 0, strictly increasing, such that all positive differences
    within each row are distinct across all rows.
    
    Strategy: Iterated Local Search (ILS) with Row Re-optimization.
    1. Initialize with a greedy construction.
    2. Repeatedly pick a random row, remove it, and re-generate it using a 
       greedy algorithm that minimizes the scope given the constraints of the 
       other rows.
    3. Use randomization in the greedy step to allow exploration (ILS).
    """
    
    def get_all_diffs(rows, exclude_idx=-1):
        """Return the set of all positive differences across all rows except exclude_idx."""
        diffs = set()
        for i, row in enumerate(rows):
            if i == exclude_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    diffs.add(row[b] - row[a])
        return diffs

    def greedy_row(used_diffs, k, max_scope_hint, randomize_factor=0):
        """
        Construct a row of length k+1 starting at 0, strictly increasing,
        minimizing the last element, such that no difference conflicts with used_diffs.
        
        randomize_factor: If > 0, at each step, pick from the top `randomize_factor` 
        smallest available candidates randomly. If 0, always pick the smallest.
        """
        row = [0]
        current_max = 0
        # We need to find k more elements.
        # To be efficient, we can bound the search.
        # The scope is the last element.
        
        # We will build the row element by element.
        # For each new element x, it must be > row[-1] and (x - a) not in used_diffs for all a in row.
        
        # Optimization: Pre-compute forbidden values? No, used_diffs changes as we add elements?
        # No, used_diffs is fixed from OTHER rows. The differences WITHIN the new row must also be distinct and not in used_diffs.
        
        # So, we need to select x_1, ..., x_k such that:
        # 1. 0 < x_1 < ... < x_k
        # 2. For all i < j, x_j - x_i not in used_diffs
        # 3. For all i, x_i - x_j (diffs) not in used_diffs (covered by 2)
        # 4. Minimize x_k.
        
        # Greedy approach:
        # Start with row = [0].
        # Next element x must satisfy: x > 0 and x not in used_diffs (since x-0=x).
        # Then next element y must satisfy: y > x, (y-x) not in used_diffs, (y-0) not in used_diffs.
        
        # This is a constraint satisfaction. Greedy: pick smallest valid x.
        
        row = [0]
        for _ in range(k):
            last = row[-1]
            found = False
            # We need to search for the next value.
            # Upper bound: We can use max_scope_hint if provided, otherwise search upwards.
            # To keep it fast, we don't want to search infinitely.
            # If we can't find a value, we return None.
            
            # Let's try values starting from last + 1.
            # We can limit the search to a reasonable range to avoid infinite loops in hard cases,
            # but for DTS, solutions usually exist within a reasonable bound.
            
            candidates = []
            # We will look for the smallest few candidates to allow randomization.
            # If randomize_factor is 0, we stop at the first valid one.
            
            # To speed up, we can check validity quickly.
            # A value x is valid if:
            # 1. x > last
            # 2. For all a in row: (x - a) not in used_diffs
            
            # We iterate x upwards.
            x = last + 1
            # Heuristic upper bound: if we don't find it in 500 steps, maybe try a different strategy?
            # For now, let's just search.
            
            while True:
                # Check validity
                valid = True
                for a in row:
                    if (x - a) in used_diffs:
                        valid = False
                        break
                
                if valid:
                    candidates.append(x)
                    if len(candidates) > randomize_factor:
                        break
                    # If randomize_factor is 0, we break immediately after finding 1
                    if randomize_factor == 0:
                        break
                x += 1
                
                # Safety break: if x gets too large, we might be stuck.
                # But for small k and n_rows, this should be fine.
                if x > 10000: # Arbitrary large limit to prevent hangs
                    return None

            if not candidates:
                return None
            
            if randomize_factor > 0:
                choice = random.choice(candidates[:randomize_factor])
            else:
                choice = candidates[0]
                
            row.append(choice)
            
        return row

    def get_scope(rows):
        return max(max(row) for row in rows)

    # 1. Initialization
    # Use a greedy construction for the whole set.
    # We build rows one by one.
    used_diffs = set()
    rows = []
    for i in range(n_rows):
        # Generate a row given the currently used diffs
        # We want to minimize scope, so greedy is good.
        # We pass used_diffs (from previous rows).
        r = greedy_row(used_diffs, k, 0, randomize_factor=0)
        if r is None:
            # Fallback: try with some randomization or just fail?
            # For robustness, if greedy fails, we might need a smarter init.
            # But greedy usually works for small k.
            # Let's try to construct a simple row if greedy fails? 
            # Actually, if greedy fails, it means no row exists with these constraints? 
            # Unlikely for small k.
            pass
        rows.append(r)
        # Update used_diffs
        for a in range(len(r)):
            for b in range(a + 1, len(r)):
                used_diffs.add(r[b] - r[a])
                
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)
    
    # 2. ILS Loop
    # Time limit: 100 seconds to be safe.
    start_time = time.time()
    time_limit = 100.0
    
    # Parameters
    # randomize_factor controls exploration. 
    # We can increase it as we go, or keep it constant.
    # Let's try a mix: mostly greedy (factor 0 or 1), occasionally more random.
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Pick a random row to replace
        row_idx = random.randint(0, n_rows - 1)
        
        # Compute diffs from OTHER rows
        other_diffs = get_all_diffs(rows, exclude_idx=row_idx)
        
        # Generate a new row
        # Strategy: 
        # 50% chance: Pure Greedy (factor 0) -> Local improvement
        # 50% chance: Randomized Greedy (factor 1-5) -> Escape local minima
        if random.random() < 0.5:
            rand_factor = 0
        else:
            rand_factor = random.randint(1, 5)
            
        new_row = greedy_row(other_diffs, k, 0, randomize_factor=rand_factor)
        
        if new_row is None:
            continue
            
        # Calculate new scope
        # We only need to check if the new row's max is smaller than the current global max,
        # or if it doesn't increase the global max.
        new_max = max(new_row)
        current_scope = get_scope(rows)
        
        # If the new row's max is larger than the current scope, the scope increases.
        # We generally want to minimize scope.
        # So we accept if new_max < current_scope (improvement) 
        # OR if new_max == current_scope (neutral, but might help future moves)
        # OR if new_max > current_scope but we are in "exploration" mode?
        
        # Simple Hill Climbing: Accept if new_max <= current_scope.
        # To allow ILS, we can accept worse solutions with a probability.
        # But since we are re-generating the *entire* row, a "worse" row is quite bad.
        # Let's stick to accepting only if it doesn't worsen the scope, 
        # but we rely on the randomization in generation to find better configurations.
        
        # Actually, if we only accept non-worsening moves, we might get stuck.
        # Let's try: Accept if new_max < current_scope.
        # If new_max == current_scope, accept with 50% probability to shuffle.
        # If new_max > current_scope, reject.
        
        if new_max < best_scope:
            # This is a new global best
            rows[row_idx] = new_row
            best_rows = [r[:] for r in rows]
            best_scope = new_max
        elif new_max == best_scope:
            # Accept to explore
            if random.random() < 0.5:
                rows[row_idx] = new_row
        # else: reject
        
    return best_rows
# EVOLVE-BLOCK-END
