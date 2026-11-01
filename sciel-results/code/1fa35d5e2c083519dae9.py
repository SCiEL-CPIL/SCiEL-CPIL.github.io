import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # The evaluator kills at 150s, but task says 120s.
    # Use 140s as a hard deadline to be safe, but aim to finish earlier.
    deadline = start_time + 140.0
    
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
                # We want to find the next number x > last
                # such that x - a for all a in row are not in used_diffs
                # and x - a are distinct among themselves (always true if a are distinct)
                
                # Search range: usually we can look a bit ahead
                # In greedy, we pick the smallest valid x
                # To allow diversity, we might pick a random valid x in a window
                
                candidates = []
                # Estimate upper bound for search. 
                # The density of used diffs is low initially, high later.
                # For k=7, n=4, max diff ~130. Total diffs 112.
                # Range 1..130 has 130 slots. 112 are filled. 
                # It's dense. Greedy might fail if it gets too tight.
                # But greedy for difference bases usually works well.
                
                x = last + 1
                found = False
                # Scan until we find enough candidates or a hard limit
                # Hard limit: if we go too far, we might exceed the best known, but we just need a valid start.
                # We will scan up to last + 100 to find candidates.
                limit = last + 100
                while x <= limit:
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
                        if len(candidates) >= window:
                            break
                    x += 1
                
                if not candidates:
                    # Fallback: extend search
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

    def local_search_sa(rows, time_limit):
        """
        Optimizes rows using Simulated Annealing.
        Returns the best rows found.
        """
        # Prepare state
        current_rows = [r[:] for r in rows]
        
        # Build diff counts
        diff_counts = {}
        for r in current_rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        T = 50.0
        T_min = 0.01
        cooling = 0.995
        
        # We will perform moves until time runs out or T is too low
        # To avoid infinite loops on bad states, we add a max steps counter per "phase"
        
        while time.time() < deadline - time_limit:
            if T < T_min:
                break
                
            # Pick a random row and index
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            pos = random.randint(1, k) # 1 to k (0 is fixed at 0)
            
            current_val = row[pos]
            
            # Determine valid range for new value
            lower_bound = row[pos - 1] + 1
            upper_bound = row[pos + 1] - 1 if pos < k else current_val + 20 # Allow slight expansion
            
            # We primarily want to reduce scope, so we bias towards lower values.
            # But SA allows increases.
            # Let's pick a random value in [lower_bound, upper_bound]
            # To be efficient, we don't scan the whole range if it's large.
            # Range is small (max ~150).
            
            # Generate candidate
            # 50% chance to try to minimize (pick small valid), 50% random
            if random.random() < 0.5:
                # Try to find the smallest valid value >= lower_bound
                new_val = lower_bound
                # We will check validity in the main loop logic, 
                # but for "minimize" mode, we want to find the *first* valid one.
                # However, checking validity for every x in range is O(k * range).
                # Range is ~100. k=7. 700 ops. Fast.
                
                # Let's just pick a random candidate in the range for simplicity of SA logic,
                # but weight the probability or use a specific move type.
                # Actually, a "greedy step" is a specific move.
                
                # Let's stick to a single move type: Random value in range.
                # This explores the space.
                new_val = random.randint(lower_bound, upper_bound)
            else:
                # "Greedy" move: Find smallest valid value
                # This is expensive if we do it every step.
                # Let's just use random for now, it's robust.
                new_val = random.randint(lower_bound, upper_bound)
                
            if new_val == current_val:
                continue
            
            # Calculate old diffs
            old_diffs = []
            for i in range(pos):
                old_diffs.append(current_val - row[i])
            for j in range(pos + 1, row_len):
                old_diffs.append(row[j] - current_val)
            
            # Calculate new diffs
            new_diffs = []
            for i in range(pos):
                new_diffs.append(new_val - row[i])
            for j in range(pos + 1, row_len):
                new_diffs.append(row[j] - new_val)
            
            # Check validity
            # A diff d in new_diffs is valid if:
            # 1. It is in old_diffs (so it's being reused/moved)
            # 2. OR it is not in diff_counts (or count is 0)
            
            valid = True
            # Optimization: Check new_diffs against diff_counts
            # We must be careful: if a diff is in old_diffs, it's "reserved" for this move.
            # So we check:
            # For d in new_diffs:
            #   if d not in old_diffs:
            #       if d in diff_counts and diff_counts[d] > 0:
            #           valid = False
            
            # Note: old_diffs might have duplicates? No, row elements are distinct, so diffs are distinct.
            # Wait, if row is [0, 2, 4], diffs for 4 are 4, 2. Distinct.
            # Yes, within a row, differences to a fixed element are distinct.
            
            # However, multiple new_diffs might be the same? No, row elements distinct.
            
            # One subtle issue: if new_diffs contains a value that is in old_diffs, it's fine.
            # If new_diffs contains a value that is in diff_counts, it's a conflict UNLESS that count comes from old_diffs.
            # Since we are removing old_diffs, we can just check:
            # Count of d in (new_diffs) == Count of d in (old_diffs) + Count of d in (rest of system)
            # This is hard to do directly.
            
            # Easier:
            # Temporarily "remove" old_diffs from a copy? No.
            # Just check:
            # For each d in new_diffs:
            #   if d in old_diffs: continue
            #   if d in diff_counts: valid = False
            
            # This works because diff_counts contains the counts *before* the move.
            # If d is in new_diffs and NOT in old_diffs, it must not be used by anyone else.
            # If d is in old_diffs, it *is* used by this element currently, so we are "moving" it.
            # Is it possible that d is in old_diffs AND used by someone else?
            # No, the current state is valid. So if d is in old_diffs, its count in diff_counts is 1 (from this element).
            # So if we move it, we remove that 1.
            # If we put it back (d in new_diffs and d in old_diffs), we add it back. Net change 0.
            # If we put a different value, we remove old, add new.
            
            for d in new_diffs:
                if d in old_diffs:
                    continue
                if d in diff_counts:
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
            
            # Calculate cost change
            # Cost is max scope.
            # If new_val > current_scope, scope might increase.
            # If new_val < current_val and current_val was the max, scope might decrease.
            
            new_scope = current_scope
            if new_val > current_scope:
                new_scope = new_val
            elif new_val == current_val:
                # If we moved the max element, we need to recalculate max of this row
                # And check if it's still the global max
                pass # Optimization: just calculate get_scope if it might change?
                # For simplicity and correctness, calculate new scope if it's a candidate for change.
                # Actually, just calculate get_scope(current_rows) after applying?
                # No, we can estimate.
                # If new_val < current_scope, scope doesn't increase.
                # It might decrease if current_val was the unique global max.
                
            # Let's just apply the move and check scope.
            # Apply move
            row[pos] = new_val
            
            # Update diff_counts
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_scope = get_scope(current_rows)
            
            # Acceptance
            if new_scope <= current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Metropolis criterion
                delta = new_scope - current_scope
                if random.random() < math.exp(-delta / T):
                    current_scope = new_scope
                else:
                    # Reject: undo move
                    row[pos] = current_val
                    
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    for d in new_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
            
            T *= cooling
            
        return best_rows, best_scope

    best_rows = None
    best_scope = float('inf')
    
    # Multi-start
    # We allocate time for multiple restarts.
    # Each restart: Greedy + SA
    
    num_restarts = 10
    # Dynamic time allocation: if we find a good solution, we can spend more time polishing it?
    # Or just fixed time per restart.
    # Total time ~140s.
    # Let's do 10 restarts of ~10s each?
    # Or adaptive: start with short, if no improvement, longer?
    
    # Strategy:
    # 1. Run 5 fast greedy+SA cycles (5s each) to get a baseline.
    # 2. Run local search on the best found with remaining time.
    
    # Phase 1: Multi-start
    for i in range(num_restarts):
        if time.time() > deadline - 20: # Leave 20s for final polish
            break
            
        # Vary parameters
        rand_prob = 0.1 + 0.05 * i
        window = 10 + 5 * i
        
        rows = greedy_init(rand_prob, window)
        scope = get_scope(rows)
        
        # Run SA for a short duration
        # Time limit for this SA: 5 seconds
        sa_time_limit = 5.0
        optimized_rows, opt_scope = local_search_sa(rows, sa_time_limit)
        
        if opt_scope < best_scope:
            best_scope = opt_scope
            best_rows = optimized_rows
            
        # If we found a very good solution (close to bound), we can break
        if best_scope < 120:
            break
            
    # Phase 2: Final Polish
    # If we have time left, run SA on the best solution found so far
    if best_rows is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            # Run SA again on the best rows, but with a longer time limit
            # We need to pass the time limit correctly.
            # local_search_sa uses `deadline - time_limit`.
            # So we want it to run until `deadline - 2` (2s safety).
            # So time_limit should be 2.
            final_rows, final_scope = local_search_sa(best_rows, 2.0)
            if final_scope < best_scope:
                best_scope = final_scope
                best_rows = final_rows

    if best_rows is None:
        # Fallback
        best_rows = greedy_init(0.1, 20)
        
    return best_rows
