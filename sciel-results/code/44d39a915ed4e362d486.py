import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before the 150s kill

    # Problem parameters
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_init(rand_prob=0.1, window=20):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                # We'll scan up to a reasonable limit to find valid candidates
                # For this problem, values rarely exceed 150
                limit = 200 
                while x <= limit and len(candidates) < window:
                    # Check if x is valid
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Fallback: extend search further
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                # Select candidate
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                # Update used_diffs
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def count_conflicts(rows):
        """
        Counts the number of distinct difference collisions.
        A valid solution has 0 conflicts.
        """
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                # If a difference appears 'count' times, there are C(count, 2) pairs of collisions
                # But for local search, simple count of extra occurrences is a good proxy
                # Let's use sum(count - 1) or sum(count^2). 
                # sum(count - 1) is linear. sum(count^2) is quadratic (penalizes large clusters).
                # Let's use sum(count - 1) for simplicity, it's often effective.
                conflicts += count - 1
        return conflicts

    def local_search_cdvr(rows, time_limit):
        """
        Optimizes rows using Conflict-Directed Value Removal (CDVR) style local search.
        1. If conflicts > 0, find the "most critical" position (involved in most conflicts) and change it to a valid value if possible, or a value that reduces conflicts.
        2. If conflicts == 0, focus on reducing scope.
        
        Returns (best_rows, best_scope).
        """
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        best_conflicts = current_conflicts
        
        # Track history of conflicts to detect stagnation
        stagnation_counter = 0
        last_conflict = current_conflicts
        
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        
        while time.time() < start_time + time_limit:
            if T < T_min:
                # Restart temperature if stuck?
                if stagnation_counter > 1000:
                    T = 5.0
                    stagnation_counter = 0
                else:
                    break
            
            # Choose a row and position to modify
            # Strategy: 
            # If we have conflicts, prioritize positions in rows that have conflicts.
            # Specifically, find the pair of elements (i, j) in a row that creates a difference d that is used by another row.
            # Then pick i or j to move.
            
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            
            # Determine if this row is involved in conflicts
            # For efficiency, let's just pick a random position first, 
            # but if we detect high conflict density, we can be smarter.
            # For now, simple random position.
            pos = random.randint(1, k) # 1 to k
            
            old_val = row[pos]
            
            # Determine new value
            # If we have conflicts, we want to pick a value that is NOT causing a conflict.
            # If we have no conflicts, we want to pick a value that reduces scope (or at least doesn't increase it too much).
            
            # Range for new value
            # Must be > row[pos-1] and < row[pos+1]
            # Or if pos is last, just > row[pos-1]
            # And we want to keep the row increasing.
            
            lower_bound = row[pos - 1] + 1
            if pos < k:
                upper_bound = row[pos + 1] - 1
            else:
                upper_bound = current_scope + 10 # Allow some expansion if needed to resolve conflicts
            
            # If lower_bound > upper_bound, this position is stuck in terms of simple swaps?
            # Actually, we can expand the row if pos == k.
            # If pos < k, we are bounded.
            
            if lower_bound > upper_bound:
                # Skip this move
                continue
                
            # Candidate selection
            # If we have conflicts, we want to avoid values that create conflicts.
            # Let's sample a few candidates and pick the one with lowest "local conflict count".
            # Local conflict count: how many other rows use the differences generated by this new value?
            
            best_candidate = None
            best_candidate_score = float('inf')
            
            # Sample a few values
            # Bias towards lower values if we are trying to reduce scope (no conflicts)
            # Bias towards "valid" values if we have conflicts.
            
            num_samples = 10
            candidates = []
            
            # Always include the smallest valid value (greedy)
            # And some random values
            # And the current value (to allow staying?) No, we must change.
            
            # Let's generate candidates
            # 1. Smallest valid value
            for v in range(lower_bound, min(upper_bound + 1, lower_bound + 50)):
                # Check if this value creates a conflict with OTHER rows
                # We don't check conflicts within the same row yet, just global.
                # Actually, conflicts within the same row are also bad.
                
                # Calculate diffs with this row
                new_diffs = set()
                for i in range(row_len):
                    if i == pos:
                        continue
                    d = abs(v - row[i])
                    new_diffs.add(d)
                
                # Check if any of these diffs are used by OTHER rows
                # This requires checking all other rows. O(n_rows * k^2).
                # For k=7, n=4, this is 4 * 28 = 112 checks. Fast.
                
                conflict_score = 0
                for r_idx2 in range(n_rows):
                    if r_idx2 == r_idx:
                        continue
                    row2 = current_rows[r_idx2]
                    for i2 in range(row_len):
                        for j2 in range(i2+1, row_len):
                            d2 = row2[j2] - row2[i2]
                            if d2 in new_diffs:
                                conflict_score += 1
                                # Early exit if score is high?
                
                # Also check conflicts within the same row (excluding pos)
                # If we put v at pos, we create diffs v-row[i] and row[j]-v.
                # We need to ensure these don't collide with EACH OTHER or with other diffs in the SAME row.
                # Wait, the "conflict" definition for validity is GLOBAL distinctness.
                # So a diff used in row 1 and row 1 again is a conflict.
                # A diff used in row 1 and row 2 is a conflict.
                # So we just need to count how many times each new_diff appears in the GLOBAL pool (excluding the old diffs of pos).
                
                # This is getting expensive to do for every sample.
                # Let's simplify: 
                # Just pick a random value in range.
                # The SA/CDVR will eventually find a valid one.
                
                candidates.append((v, conflict_score))
                
                if len(candidates) >= num_samples:
                    break
            
            # Add some random samples
            for _ in range(num_samples):
                v = random.randint(lower_bound, upper_bound)
                # Quick check: does it create obvious conflicts?
                # Just add it, the global check will handle it.
                # But we need a score to pick the best candidate.
                # Let's just use the number of conflicts it creates globally.
                # This is expensive.
                
                # Alternative: Just pick a random value and let the global conflict count guide the acceptance.
                # If the new global conflict count is lower, accept.
                # This is standard Local Search.
                
                candidates.append((v, -1)) # Score -1 means "not evaluated"
            
            # Pick a candidate
            # If we have conflicts, prefer candidates with lower conflict score (if evaluated).
            # Since we didn't evaluate most, let's just pick a random one from the generated list.
            # Or, pick the one with the smallest value (to reduce scope) if no conflicts, 
            # or random if conflicts.
            
            if current_conflicts > 0:
                # Pick randomly from candidates
                v = random.choice(candidates)[0]
            else:
                # Pick the smallest valid value? 
                # Or random?
                # To reduce scope, smaller is better.
                # But we might need to expand.
                # Let's pick a random value, but biased towards smaller.
                # Or just the smallest value in range?
                # Greedy descent: pick smallest value that maintains validity?
                # Validity check is expensive.
                
                # Let's pick a random value.
                v = random.choice(candidates)[0]
                
            if v == old_val:
                continue
                
            # Apply move and check global conflict count
            # We need to calculate the new conflict count efficiently?
            # Or just calculate it from scratch?
            # N=4, K=7. Total diffs 112.
            # Calculating conflict count from scratch is O(N * K^2) = 4 * 28 = 112 operations.
            # This is VERY fast. We can afford to calculate it from scratch every step.
            
            # Save old state
            old_val_saved = old_val
            row[pos] = v
            
            new_conflicts = count_conflicts(current_rows)
            new_scope = get_scope(current_rows)
            
            # Acceptance Criteria
            # 1. If new_conflicts < current_conflicts: Accept (and update best)
            # 2. If new_conflicts == current_conflicts:
            #      - If new_scope < current_scope: Accept
            #      - Else: Accept with probability exp(-delta_scope / T)
            # 3. If new_conflicts > current_conflicts:
            #      - Accept with probability exp(-delta_conflict / T)
            
            delta_conflict = new_conflicts - current_conflicts
            delta_scope = new_scope - current_scope
            
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                else:
                    if random.random() < math.exp(-delta_scope / T):
                        accept = True
            else:
                # Conflict increased
                if random.random() < math.exp(-delta_conflict / T):
                    accept = True
            
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                
                # Update best
                # We prefer valid solutions (conflicts == 0) with lowest scope.
                # If we have conflicts, we prefer lower conflict count, then lower scope.
                
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and current_scope < best_scope):
                    best_conflicts = current_conflicts
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                    
                if current_conflicts < last_conflict:
                    last_conflict = current_conflicts
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
            else:
                # Revert
                row[pos] = old_val_saved
                stagnation_counter += 1
                
            T *= cooling
            
        return best_rows, best_scope

    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')
    
    # Multi-start
    # Allocate time for multiple restarts.
    # Total time ~135s.
    # Let's do restarts of ~10s each.
    # Leave 15s for final polish.
    
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Vary parameters
        rand_prob = 0.05 + 0.05 * i
        window = 10 + 5 * i
        
        rows = greedy_init(rand_prob, window)
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)
        
        # Run CDVR for a short duration
        # Time limit for this CDVR: 10 seconds
        cdvr_time_limit = 10.0
        optimized_rows, opt_scope = local_search_cdvr(rows, cdvr_time_limit)
        
        # Re-evaluate conflicts for the optimized rows to be sure
        opt_conflicts = count_conflicts(optimized_rows)
        
        if opt_conflicts < best_conflicts or (opt_conflicts == best_conflicts and opt_scope < best_scope):
            best_conflicts = opt_conflicts
            best_scope = opt_scope
            best_rows = optimized_rows
            
        # If we found a valid solution (0 conflicts), we can break early if it's very good
        if best_conflicts == 0 and best_scope < 125:
            break
            
    # Phase 2: Final Polish
    # If we have time left, run CDVR on the best solution found so far
    if best_rows is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            # Run CDVR again on the best rows
            # We want it to run until `deadline - 2`
            # So time_limit = remaining_time - 2
            final_rows, final_scope = local_search_cdvr(best_rows, remaining_time - 2.0)
            
            final_conflicts = count_conflicts(final_rows)
            
            if final_conflicts < best_conflicts or (final_conflicts == best_conflicts and final_scope < best_scope):
                best_conflicts = final_conflicts
                best_scope = final_scope
                best_rows = final_rows

    if best_rows is None:
        # Fallback
        best_rows = greedy_init(0.1, 20)
        
    return best_rows
