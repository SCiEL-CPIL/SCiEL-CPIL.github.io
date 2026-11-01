import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Total differences required per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def greedy_init(window_size=20, rand_prob=0.1):
        """
        Randomized greedy construction.
        Builds rows one by one. For each new element, looks for valid candidates
        in a window and picks the smallest with high probability, or a random one
        with low probability to allow diversity.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                start_val = row[-1] + 1
                valid_candidates = []
                
                # Scan for valid candidates
                # We don't know the exact upper bound, but greedy usually stays tight.
                # If we fail to find any in the window, we extend the search.
                search_limit = start_val + window_size * 10
                found = False
                for x in range(start_val, search_limit):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        valid_candidates.append(x)
                        if len(valid_candidates) >= window_size:
                            found = True
                            break
                
                if not found:
                    # Fallback: scan until found (should be rare if window is large enough)
                    x = start_val
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            valid_candidates.append(x)
                            break
                        x += 1

                if valid_candidates:
                    if random.random() < rand_prob and len(valid_candidates) > 1:
                        x = random.choice(valid_candidates)
                    else:
                        x = valid_candidates[0]
                    
                    row.append(x)
                    # Update used_diffs with the diffs created by adding x
                    # These are x - a for all a in row[:-1]
                    for a in row[:-1]:
                        used_diffs.add(x - a)
            rows.append(row)
        
        # Rebuild used_diffs from scratch to be safe and consistent with local search state
        used_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
                    
        return rows, used_diffs

    def get_scope(rows):
        return max(max(r) for r in rows)

    # Local Search / Optimization
    def local_search(rows, used_diffs, deadline):
        """
        Optimizes a valid configuration by trying to reduce the scope (largest entry).
        It does this by attempting to move elements to smaller values if possible.
        """
        current_scope = get_scope(rows)
        
        # Pre-compute diffs for each row? 
        # For k=7, row size 8, diffs=28. n_rows=4. Total 112 diffs.
        # It is fast enough to recompute or update incrementally.
        # We will use a diff_counts dictionary for O(1) conflict checks.
        
        diff_counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        # conflict_count should be 0 for a valid start
        conflict_count = sum(1 for v in diff_counts.values() if v > 1)
        
        while time.time() < deadline:
            # Pick a random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick a random position to try to reduce
            # Bias towards the end of the row (higher indices) as they contribute more to scope
            # Or just pick random. Let's pick random but weighted towards higher indices.
            # Actually, any position can be moved to reduce the max if it's the max, 
            # or to create space for others.
            # Let's just pick a random position 1..k
            pos = random.randint(1, k)
            
            current_val = row[pos]
            
            # Determine valid range for new value
            lower_bound = row[pos - 1] + 1
            upper_bound = current_val - 1 # We only want to reduce scope, so try smaller values
            
            if pos < k:
                upper_bound = min(upper_bound, row[pos + 1] - 1)
                
            if lower_bound > upper_bound:
                continue
            
            # Calculate diffs to be removed
            old_diffs = []
            for i in range(pos):
                old_diffs.append(current_val - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs.append(row[j] - current_val)
            
            # We want to find a new_val in [lower_bound, upper_bound] such that
            # the new diffs don't conflict with (used_diffs - old_diffs)
            
            # To be efficient, we can iterate through candidate values.
            # Since we want to minimize scope, we should try smaller values first?
            # Or random? Random might find a valid spot faster if the space is sparse.
            # But trying smaller values first is a good heuristic for "reduction".
            
            # Let's try a random subset of the range to avoid scanning huge ranges if stuck
            # But for reduction, we are bounded by current_val - 1, which is at most scope.
            # Scope is ~100-200. So range is small.
            
            # Strategy: Try values in random order, or sorted order?
            # Sorted order (ascending) finds the smallest valid value, which is best for local min.
            # But it might be slow if no small value works.
            # Let's try a mix: a few random, then if time allows, scan from bottom.
            
            found = False
            # Try a few random candidates
            candidates = list(range(lower_bound, upper_bound + 1))
            if len(candidates) > 10:
                random.shuffle(candidates)
                candidates = candidates[:10] + [lower_bound, upper_bound]
                # Ensure unique
                candidates = list(set(candidates))
            else:
                random.shuffle(candidates)
            
            for new_val in candidates:
                # Calculate new diffs
                new_diffs = []
                for i in range(pos):
                    new_diffs.append(new_val - row[i])
                for j in range(pos + 1, len(row)):
                    new_diffs.append(row[j] - new_val)
                
                # Check conflicts
                # A diff d is valid if:
                # 1. It is in old_diffs (so it's "freed up") OR
                # 2. It is not in diff_counts (or count is 0)
                
                # Simpler: 
                # Temporarily remove old_diffs from a copy of diff_counts? No, too slow.
                # Check each new_diff:
                # If new_diff is in old_diffs, it's safe (it was used by this element before).
                # Otherwise, it must not be in diff_counts.
                
                valid = True
                for d in new_diffs:
                    if d in old_diffs:
                        continue
                    if d in diff_counts:
                        valid = False
                        break
                
                if valid:
                    # Apply move
                    row[pos] = new_val
                    
                    # Update diff_counts
                    # Remove old_diffs
                    for d in old_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    # Add new_diffs
                    for d in new_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    
                    found = True
                    break
            
            if found:
                # Update scope
                new_scope = get_scope(rows)
                if new_scope < current_scope:
                    current_scope = new_scope
            
        return rows, current_scope

    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multi-start Greedy
    # We will run several greedy constructions.
    # Early ones use smaller windows/probs, later ones more randomized.
    
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Increase randomization over time
        window = 10 + i * 2
        prob = 0.05 + 0.01 * i
        
        rows, used = greedy_init(window, prob)
        scope = get_scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            best_used = set(used)
            
        # If we have a very good scope, we can break early?
        # The theoretical min is 112. Best known 134.
        # If we hit < 130, that's a great result.
        if best_scope < 120:
            break

    # Phase 2: Local Search on the best found so far
    if best_rows is not None:
        # We need to pass a deadline for the local search
        # Leave 5 seconds for final processing
        local_deadline = deadline - 5
        optimized_rows, final_scope = local_search(best_rows, best_used, local_deadline)
        
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = [r[:] for r in optimized_rows]
            
    # Phase 3: If we still have time and haven't found a great solution,
    # try running local search on other good candidates from Phase 1?
    # We didn't store them. Let's just do a few more greedy+local cycles if time permits.
    
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            window = random.randint(20, 50)
            prob = random.uniform(0.1, 0.3)
            rows, used = greedy_init(window, prob)
            scope = get_scope(rows)
            
            if scope < best_scope:
                # Optimize this new best
                local_deadline = deadline - 5
                optimized_rows, final_scope = local_search(rows, used, local_deadline)
                if final_scope < best_scope:
                    best_scope = final_scope
                    best_rows = [r[:] for r in optimized_rows]
            elif scope < best_scope + 10:
                # Even if not better, optimize it, it might become better
                local_deadline = deadline - 5
                optimized_rows, final_scope = local_search(rows, used, local_deadline)
                if final_scope < best_scope:
                    best_scope = final_scope
                    best_rows = [r[:] for r in optimized_rows]

    if best_rows is None:
        # Fallback: pure greedy
        best_rows, _ = greedy_init(10, 0.0)

    return best_rows
