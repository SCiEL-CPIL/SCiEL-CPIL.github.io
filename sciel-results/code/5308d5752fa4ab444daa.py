import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_rows = None
    best_max = float('inf')
    
    # Initial greedy solution to set a good upper bound
    used_diffs = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        greedy_rows.append(row)
    best_max = max(max(r) for r in greedy_rows)
    best_rows = greedy_rows

    # Structural change: Bitset-based CSP with forward checking and randomized restarts
    # We model the problem as selecting rows sequentially.
    # State: used_diffs (bitset or set), current row being built.
    # Optimization: Precompute diff masks for all possible row prefixes? 
    # Since max_val is ~134, diffs are 1..133.
    # We can use a set of used diffs.
    
    # Let's implement a smarter search:
    # 1. Try to construct rows with a target max M.
    # 2. Use randomized backtracking with forward checking.
    # 3. Start with M = best_max - 1 and decrease.
    
    def try_construct(target_max, n_rows, k, random_state):
        used_diffs = set()
        rows = []
        
        def backtrack_row(idx, row, used_diffs):
            if idx == k:
                return row[:]
            
            last = row[-1]
            rem = k - idx
            # Pruning: We need 'rem' more elements.
            # The minimum possible last element is at least last + rem (since diffs >= 1).
            # Also, we must not exceed target_max.
            # A tighter bound: The differences must be distinct.
            # The smallest possible differences for the remaining elements are 1, 2, ..., rem?
            # No, the differences are from the new element to ALL previous elements.
            # So if we add x, diffs are x - a_i for all a_i in row.
            # These must be distinct and not in used_diffs.
            
            # Let's iterate candidates x from last + 1 to target_max.
            # To speed up, we can shuffle the range? No, we want small x for minimization.
            # But randomization helps avoid local minima.
            # Let's pick a random valid candidate from a window of valid candidates.
            
            valid_candidates = []
            # Optimization: If we have many candidates, we don't need to check all.
            # But checking is fast (O(k)).
            
            # Heuristic: Start from last+1, but maybe skip some to diversify?
            # Let's just collect all valid candidates up to a limit, or all.
            # Since target_max is small (~134), collecting all is fine.
            
            for x in range(last + 1, target_max + 1):
                # Check if x - a_i are all distinct and not in used_diffs
                # Also check if x - a_i are distinct among themselves?
                # Since a_i are distinct and increasing, x - a_i are distinct and decreasing.
                # So we only need to check against used_diffs.
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    valid_candidates.append(x)
                    
            if not valid_candidates:
                return None
            
            # Randomize: Pick from the first few valid candidates to keep x small
            # but allow some diversity.
            # Or shuffle all and pick the first? That might pick a large x.
            # Let's pick from a window of the smallest valid candidates.
            W = 5
            if len(valid_candidates) > W:
                # Pick randomly from the first W
                x = random_state.choice(valid_candidates[:W])
            else:
                x = random_state.choice(valid_candidates)
            
            new_row = row + [x]
            new_diffs = {x - a for a in row}
            
            res = backtrack_row(idx + 1, new_row, used_diffs | new_diffs)
            if res is not None:
                return res
            
            # Backtrack: try next candidate?
            # The above code only tries ONE candidate per call.
            # For proper backtracking, we need to iterate.
            # But iterating all candidates might be slow if we don't prune well.
            # Let's modify to iterate, but with a limit on depth or time?
            # For now, let's try a "randomized greedy with limited backtracking".
            # If we fail with one choice, we could try another.
            # Let's implement full backtracking but with a node limit or time check?
            # Simpler: Just return None if the chosen x fails.
            # This is a "randomized greedy" not full backtracking.
            # To improve, we can try a few candidates if the first fails.
            
            # Let's try up to 3 candidates randomly from the valid list (sorted by x)
            # Actually, the current code picks one and returns None if it fails later.
            # This is weak. Let's improve it to try multiple candidates.
            
            # Revised backtrack_row:
            # 1. Get valid candidates.
            # 2. Shuffle them? No, sort by x to prefer small.
            # 3. Try them in random order from a subset?
            
            # Let's change the strategy:
            # Instead of picking one and committing, let's pick a random subset of candidates
            # and try them in random order.
            
            # But for efficiency, let's stick to the "pick from window" but if it fails,
            # we need to backtrack.
            # The current structure returns None immediately.
            # Let's rewrite backtrack_row to iterate.
            
            return None # Placeholder, will be replaced by full loop below

        # Let's rewrite the inner function properly with iteration.
        def backtrack_row_full(idx, row, used_diffs):
            if idx == k:
                return row[:]
            
            last = row[-1]
            valid_candidates = []
            for x in range(last + 1, target_max + 1):
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    valid_candidates.append(x)
                    
            if not valid_candidates:
                return None
            
            # Shuffle to explore different paths
            # But we want to prefer small x.
            # Let's sort by x, then take a random subset or shuffle the whole list?
            # Shuffling the whole list might pick a large x first, which is bad for minimization.
            # Let's pick from the first W candidates randomly.
            W = 5
            candidates_to_try = valid_candidates[:W] if len(valid_candidates) > W else valid_candidates
            random_state.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                new_row = row + [x]
                new_diffs = {x - a for a in row}
                res = backtrack_row_full(idx + 1, new_row, used_diffs | new_diffs)
                if res is not None:
                    return res
                    
            return None

        # Main loop for rows
        for i in range(n_rows):
            row = [0]
            res = backtrack_row_full(0, row, used_diffs)
            if res is None:
                return None
            used_diffs |= {res[j] - res[i] for i in range(len(res)) for j in range(i+1, len(res))}
            rows.append(res)
            
        return rows

    # Main search loop
    while time.time() < deadline:
        # Try to find a solution with max < best_max
        target_max = best_max - 1
        
        # Multiple restarts
        for restart in range(100):
            if time.time() > deadline:
                break
            random_state = random.Random()
            res = try_construct(target_max, n_rows, k, random_state)
            if res is not None:
                current_max = max(max(r) for r in res)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = res
                break # Found a solution for this target, try to improve further
        
        # If we can't find a solution for best_max - 1, we might be stuck.
        # We can try to construct with best_max to see if we find a better arrangement?
        # No, that won't improve the max.
        # We can try to increase target_max? No, we want to minimize.
        # If we fail, we just keep the best found.
        
        # Optional: Try to optimize within the current best_max?
        # E.g., if we have a solution with max 134, can we find one with max 133?
        # The loop above tries best_max - 1.
        # If it fails, we might want to try a different strategy for best_max - 1.
        # The random restarts should help.

    return best_rows
