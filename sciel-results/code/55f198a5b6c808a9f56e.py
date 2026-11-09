import time
import random
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set.
    Uses a bitmask-accelerated randomized local search (simulated annealing) 
    starting from a greedy baseline.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # 1. Greedy Baseline
    # We generate a standard greedy solution to get an initial valid state.
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used_diffs:
                row.append(x)
                used_diffs.update(diffs)
        rows.append(row)
    
    best_scope = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    
    # If k is small, greedy is often optimal or near-optimal. 
    # For this task (4,7), greedy gives 146, best known 134.
    # We need to search for 134 or lower.
    
    # 2. Local Search with Bitmask
    # State: list of rows.
    # We use a global bitmask for all differences to quickly check conflicts.
    # However, since we are moving elements, we need to update the mask efficiently.
    
    # Precompute necessary data structures for speed
    # Max possible difference is bounded by best_scope. 
    # Actually, during search, we might exceed best_scope temporarily? 
    # No, we want to minimize max. We can constrain moves to not exceed current best.
    # Or we can allow exceeding but penalize heavily.
    # Let's stick to a "repair" style or "min-max" style.
    # Strategy: 
    # 1. Start with greedy.
    # 2. Try to reduce the maximum element.
    # 3. Use random swaps/moves to escape local minima.
    
    # To make it faster, we track:
    # - used_diffs_mask: an integer or array of booleans for differences 1..max_limit
    # - rows: the current configuration
    
    # Let's use a list of booleans for differences up to a reasonable limit (e.g., 200)
    MAX_DIFF_LIMIT = 200 
    used_diffs = [False] * (MAX_DIFF_LIMIT + 1)
    
    # Initialize mask from best_rows
    def update_mask(row, add):
        """Add or remove differences of a row from the global mask."""
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d <= MAX_DIFF_LIMIT:
                    used_diffs[d] = used_diffs[d] ^ add # XOR for toggle? No, we need count or careful add/remove.
                    # Actually, since diffs are unique in a valid state, and we start valid,
                    # we can just use a set or a boolean array if we are careful.
                    # If we move an element, we remove old diffs and add new ones.
                    # If a diff is added that already exists, it's a conflict.
                    # For speed, let's use a set of used diffs. It's O(1) average.
                    pass
    
    # Let's revert to using a set for simplicity and robustness, but optimize the inner loop.
    # The bottleneck in the previous approach was the row-by-row construction.
    # Here we start with a valid set and mutate it.
    
    current_rows = [r[:] for r in best_rows]
    used_diffs_set = set()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used_diffs_set.add(r[j] - r[i])
                
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        # Check if all diffs are distinct
        s = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in s:
                        return False
                    s.add(d)
        return True
        
    # Optimized validity check for a single row change
    # When we change one element in one row, only diffs involving that element change.
    
    def try_move(row_idx, pos_idx, new_val):
        """
        Attempt to change current_rows[row_idx][pos_idx] to new_val.
        Returns (success, new_scope, new_used_diffs_set) or (False, None, None).
        We don't want to copy the whole set every time.
        """
        row = current_rows[row_idx]
        old_val = row[pos_idx]
        if old_val == new_val:
            return True, get_scope(current_rows), used_diffs_set
        
        # Remove old diffs involving pos_idx
        removed_diffs = set()
        for i in range(len(row)):
            if i == pos_idx: continue
            d = abs(row[i] - old_val)
            removed_diffs.add(d)
            
        # Check if new_val creates conflicts with existing elements in the row
        # And if the new diffs conflict with global used_diffs_set (minus removed)
        
        # New diffs involving pos_idx
        new_diffs = set()
        for i in range(len(row)):
            if i == pos_idx: continue
            d = abs(row[i] - new_val)
            new_diffs.add(d)
            
        # Check if new_diffs are in (used_diffs_set - removed_diffs)
        # Note: new_diffs might have duplicates within themselves? No, row is strictly increasing.
        # But if new_val breaks strict increase, it's invalid.
        
        # Check strict increase
        if pos_idx > 0 and new_val <= row[pos_idx - 1]:
            return False, None, None
        if pos_idx < len(row) - 1 and new_val >= row[pos_idx + 1]:
            return False, None, None
            
        # Check global conflicts
        # The set of available diffs is used_diffs_set - removed_diffs
        # We need new_diffs to be disjoint from this set.
        # Also new_diffs must not contain any diff that is in removed_diffs? 
        # No, removed_diffs are freed up. So new_diffs can reuse removed_diffs.
        # But new_diffs cannot reuse any other diff in used_diffs_set.
        
        # Efficient check:
        # conflict = False
        # for d in new_diffs:
        #     if d in used_diffs_set and d not in removed_diffs:
        #         conflict = True
        #         break
        
        # To speed this up, we can precompute the "available" set? 
        # No, used_diffs_set changes.
        # Let's just do the check. k is small (8 elements), so at most 7 diffs per element.
        
        for d in new_diffs:
            if d in used_diffs_set and d not in removed_diffs:
                return False, None, None
                
        # Also check if new_diffs have internal conflicts? 
        # Since row is strictly increasing, diffs from a fixed point to others are distinct.
        # So no internal conflict.
        
        # If valid, apply the move
        row[pos_idx] = new_val
        used_diffs_set -= removed_diffs
        used_diffs_set |= new_diffs
        
        new_scope = get_scope(current_rows)
        return True, new_scope, used_diffs_set
        
    # We need to be able to UNDO moves easily for backtracking or simulated annealing.
    # Let's implement a simpler "Random Restart Local Search" with hill climbing.
    
    # Strategy:
    # 1. Start with greedy.
    # 2. Try to reduce the maximum element.
    # 3. If stuck, randomize a part of the solution and try again.
    
    # Actually, a better approach for small k:
    # Iterate over the maximum element of the last row (or any row).
    # Try to find a configuration with max = M.
    # We can use a SAT-like solver or backtracking with bitmasks.
    # Given the time limit, let's use a randomized backtracking that is much faster
    # than the row-by-row greedy because it can look ahead and backtrack.
    
    # Let's try a different structural approach:
    # **Bitmask Backtracking with Randomization**
    
    # We will attempt to construct the set row by row, but with backtracking.
    # To make it fast, we use bitmasks for the used differences.
    
    def backtrack_construct(n_rows, k, limit, seed, max_steps=100000):
        random.seed(seed)
        
        # Bitmask for used differences. 
        # We need to represent differences up to 'limit'.
        # Python integers can be arbitrary precision, so we can use a big int as a bitmask.
        # Bit i is set if difference i is used.
        
        used_mask = 0
        rows = []
        
        def check_and_add_diffs(row, start_idx, candidate_val):
            """
            Check if adding candidate_val to row (at end, since we build forward)
            creates conflicts.
            Returns (valid, new_mask).
            """
            # Diffs are candidate_val - row[i] for all i < len(row)
            # We need to check if any of these bits are already set in used_mask.
            
            # Since we are building row by row, and row is increasing,
            # the diffs are distinct within the row.
            # We just need to check against used_mask.
            
            new_mask = used_mask
            # Check conflicts
            # We can do this by checking each diff.
            # To optimize, we can precompute the mask for the row?
            # No, row changes.
            
            for i in range(len(row)):
                d = candidate_val - row[i]
                bit = 1 << d
                if (used_mask & bit) != 0:
                    return False, 0
                # We don't update used_mask yet, we just check.
                # But we also need to ensure no two diffs in the current row conflict?
                # They are distinct because row is strictly increasing.
                # So we just need to check against global used_mask.
                
            # If we get here, no conflicts with existing global mask.
            # We also need to ensure that the new diffs don't conflict with each other?
            # No, they are distinct.
            
            # So we can safely add them.
            # But wait, we need to return the NEW mask if we were to add it.
            # For backtracking, we want to check validity first.
            
            # Let's compute the new mask to return if valid.
            for i in range(len(row)):
                d = candidate_val - row[i]
                new_mask |= (1 << d)
                
            return True, new_mask

        def solve_row(row_idx, depth):
            """
            Recursively build row row_idx.
            depth is the index in the row we are filling (0 is 0, so we start at 1).
            """
            if depth == k + 1:
                # Row complete
                return True
                
            row = rows[row_idx]
            current_max = row[-1] if row else 0
            
            # Determine range for next element
            # Start from current_max + 1
            # End at limit
            
            # To add randomness, we can shuffle the candidates or pick randomly.
            # But we must ensure we don't pick a value that makes it impossible to finish.
            
            # Heuristic: Pick candidates in a random order?
            # Or pick from a window?
            
            # Let's try a random subset of candidates.
            # Collect valid candidates.
            candidates = []
            for x in range(current_max + 1, limit + 1):
                valid, _ = check_and_add_diffs(row, depth, x)
                if valid:
                    candidates.append(x)
                    
            if not candidates:
                return False
                
            # Randomize order to escape local minima
            random.shuffle(candidates)
            
            # Limit the number of candidates we try to avoid deep backtracking explosion
            # If there are too many, pick a random subset?
            # Or just try the first few?
            # Let's try a random permutation and stop after a certain number of failures?
            
            # For efficiency, let's just try them in the shuffled order.
            # But if the tree is deep, this might be slow.
            # We have max_steps guard.
            
            for x in candidates:
                # Apply move
                old_mask = used_mask
                row.append(x)
                # Update mask
                for i in range(len(row) - 1):
                    d = x - row[i]
                    used_mask |= (1 << d)
                    
                if solve_row(row_idx, depth + 1):
                    return True
                    
                # Backtrack
                row.pop()
                used_mask = old_mask
                
            return False

        # Main loop
        for _ in range(n_rows):
            rows.append([0])
            if not solve_row(len(rows) - 1, 1):
                return None
                
        return rows

    # This backtracking might be slow if limit is tight and randomization is poor.
    # Let's combine it with the previous local search idea?
    # No, let's stick to one clear strategy.
    # The bitmask backtracking is a different design family from row-by-row greedy.
    
    # Let's optimize the bitmask check.
    # Instead of iterating through row elements to check bits, we can precompute?
    # No, row is dynamic.
    
    # Let's try to improve the backtracking by using a "fail-first" heuristic.
    # Try values that leave more room? Or values that use "scarce" differences?
    
    # Actually, the previous program's score was 1.49.
    # The best known is 134.
    # Greedy gives 146.
    # We need to find 134.
    
    # Let's use the bitmask backtracking but with a twist:
    # We don't just try to build row by row. We try to build the whole set.
    # But row by row is fine if we can backtrack.
    
    # To make it faster, we can use a heuristic:
    # When choosing the next element, prefer those that don't use small differences?
    # Or prefer those that use differences that are "harder" to place?
    
    # Let's implement the bitmask backtracking with random restarts.
    
    best_scope_found = best_scope
    best_rows_found = best_rows
    
    # Try to find a solution with scope < best_scope
    # We know greedy gives 146. We want 134.
    # Let's search for 134, 133, ...
    
    target_limit = best_scope - 1
    max_trials = 5000 # Number of restarts
    
    # We will use the backtrack_construct function above, but we need to make it faster.
    # The inner loop `for x in range(...)` can be expensive if limit is large.
    # But limit is ~130.
    
    # One optimization: Precompute the bitmask for a row?
    # No.
    
    # Let's just run the backtracking with random seeds.
    
    seed = 0
    while time.time() < deadline and target_limit >= 112: # 112 is counting bound
        found = False
        for trial in range(100): # 100 trials per limit
            if time.time() > deadline - 2:
                break
                
            rows = backtrack_construct(n_rows, k, target_limit, seed)
            if rows is not None:
                # Verify
                all_diffs = set()
                valid = True
                for r in rows:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                            
                if valid:
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope_found:
                        best_scope_found = new_scope
                        best_rows_found = rows
                    # If we found one, we can try to lower the limit
                    target_limit = new_scope - 1
                    found = True
                    break
                    
            seed += 1
            
        if not found:
            target_limit -= 1

    return best_rows_found
