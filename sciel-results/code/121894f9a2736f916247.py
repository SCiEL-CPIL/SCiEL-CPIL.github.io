import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = k + 1
    total_diffs_per_row = n * (n - 1) // 2
    
    # Helper to check if a row is strictly increasing and starts at 0
    def is_valid_row(row):
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        return True

    # Helper to get all differences in a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Check if a set of new diffs conflicts with used diffs
    def has_conflict(new_diffs, used_diffs):
        return bool(new_diffs & used_diffs)

    # Greedy construction with randomization to get a valid start
    def greedy_construct_random():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Randomize the order of trying next numbers? 
            # Actually, simple greedy picks the smallest possible next number.
            # To diversify, we can occasionally pick a larger number.
            x = 0
            while len(row) < n:
                # Try a range of candidates
                # Standard greedy: x+1
                # Randomized: x + random.randint(1, 5)
                # But we must ensure we can complete the row.
                # Let's stick to a slightly randomized greedy:
                # Pick the smallest valid x, but with some probability pick a larger one.
                candidates = []
                # Look ahead a bit
                max_search = 50
                for dx in range(1, max_search):
                    cand = x + dx
                    new_diffs = {cand - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(cand)
                        if len(candidates) > 10:
                            break
                if not candidates:
                    # If we can't find one in the window, expand
                    for dx in range(max_search, max_search + 100):
                        cand = x + dx
                        new_diffs = {cand - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(cand)
                            break
                if not candidates:
                    # Should not happen if we have enough space, but fallback
                    return None
                
                # Pick a candidate. 
                # Bias towards smaller to keep max low, but allow larger for diversity
                if random.random() < 0.8:
                    x = candidates[0]
                else:
                    x = random.choice(candidates)
                
                row.append(x)
                used |= get_row_diffs(row) # Update used with new diffs
                # Optimization: we can update used incrementally
                # But get_row_diffs is O(n^2) which is small.
            rows.append(row)
        return rows

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    
    # Run a few greedy constructions to find a good starting point
    for _ in range(5):
        rows = greedy_construct_random()
        if rows:
            cur_max = max(max(r) for r in rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [row[:] for row in rows]
    
    if best_rows is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_max = max(max(r) for r in rows)

    current_rows = [row[:] for row in best_rows]
    current_max = best_max
    
    # Precompute used diffs for current state
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            used |= get_row_diffs(row)
        return used

    current_used = compute_used_diffs(current_rows)

    # Local Search: Try to reduce max entry
    def local_search(rows, used_diffs, max_val, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to reduce the max entry by shifting suffixes of rows
            # We want to find a row that has the current max_val and try to reduce it.
            # Or just try all rows.
            
            # Identify rows that contain the current max
            # Actually, we can just try to improve any row.
            
            # To be efficient, focus on rows with large entries.
            row_indices = list(range(n_rows))
            random.shuffle(row_indices)
            
            for r_idx in row_indices:
                row = rows[r_idx]
                if max(row) < max_val:
                    continue # This row doesn't contribute to the max, but reducing it might allow others? 
                             # No, reducing a row that doesn't have the max doesn't reduce the global max.
                             # However, it might free up differences, allowing OTHER rows to be reduced.
                             # So we should still try.
                
                # Try shifting suffixes
                # For each split point s, try shifting row[s:] by delta < 0
                # We want the smallest delta that keeps the row valid and conflict-free.
                
                # We can try to find the minimal valid shift for each suffix.
                # Let's try a few random shifts first, then deterministic minimal.
                
                # Strategy: For each s, try to shift left as much as possible.
                # The limit is row[s] + delta > row[s-1] (if s>0)
                # And no conflicts.
                
                for s in range(1, n):
                    # Try to shift left
                    # Max possible shift left is row[s-1] - row[s] + 1
                    # But we need to check conflicts.
                    
                    # Let's try a binary search or linear scan for the best delta?
                    # Linear scan from smallest magnitude (most reduction) to 0.
                    # Actually, we want the most negative delta that is valid.
                    # Let's try a few random deltas first to escape local minima?
                    # No, local search should be deterministic greedy: take the best improvement.
                    
                    best_delta_for_s = 0
                    best_new_max = max(row)
                    
                    # Try deltas from -1 down to (row[s-1] - row[s] + 1)
                    min_delta = row[s-1] - row[s] + 1
                    # We want to try large negative deltas first to see if we can reduce max significantly.
                    # But checking conflicts is O(n^2).
                    # Let's just try a few promising deltas.
                    
                    # Actually, let's just try to shift by -1, -2, ... until invalid?
                    # That's too slow.
                    
                    # Better: Try to shift by -1. If valid, do it. 
                    # This is a "steepest descent" step.
                    # To avoid getting stuck, we can try random deltas in a range.
                    
                    # Let's try a random delta in [-5, -1]
                    for _ in range(10):
                        delta = random.randint(-5, -1)
                        if delta < min_delta:
                            continue
                        # Check validity
                        new_row = row[:]
                        for j in range(s, n):
                            new_row[j] += delta
                        
                        if not is_valid_row(new_row):
                            continue
                        
                        # Check conflicts
                        # Remove old diffs for this row from used_diffs
                        old_diffs = get_row_diffs(row)
                        new_diffs = get_row_diffs(new_row)
                        
                        # Check if new_diffs intersect with (used_diffs - old_diffs)
                        other_diffs = used_diffs - old_diffs
                        if new_diffs & other_diffs:
                            continue
                        
                        # Valid move!
                        # Calculate new global max
                        new_global_max = max(max_val, max(new_row)) # Wait, if we reduce max_row, global max might drop
                        # Actually, max_val is the global max. If this row had the global max, it might drop.
                        # Let's just compute max of all rows.
                        # For speed, assume max_val is dominated by this row or another.
                        # Let's just check if max(new_row) < max_val or if this row was the only one with max_val.
                        
                        # Simple check:
                        # Apply move
                        rows[r_idx] = new_row
                        used_diffs = (used_diffs - old_diffs) | new_diffs
                        
                        # Update global max
                        new_global_max = 0
                        for r in rows:
                            if r[-1] > new_global_max:
                                new_global_max = r[-1]
                        
                        if new_global_max < max_val:
                            return rows, used_diffs, new_global_max, True
                        else:
                            # Revert
                            rows[r_idx] = row
                            used_diffs = (used_diffs - new_diffs) | old_diffs
                            break # Try next s? Or next row?
                    
                    # If no random move worked, try the minimal valid shift (most negative)
                    # This is expensive, so only do it if we have time
                    if time.time() < time_limit - 0.5:
                        for delta in range(min_delta, 0):
                            if delta == 0: break
                            # Check validity
                            new_row = row[:]
                            for j in range(s, n):
                                new_row[j] += delta
                            
                            if not is_valid_row(new_row):
                                continue
                            
                            old_diffs = get_row_diffs(row)
                            new_diffs = get_row_diffs(new_row)
                            other_diffs = used_diffs - old_diffs
                            if new_diffs & other_diffs:
                                continue
                            
                            # Valid
                            rows[r_idx] = new_row
                            used_diffs = (used_diffs - old_diffs) | new_diffs
                            
                            new_global_max = 0
                            for r in rows:
                                if r[-1] > new_global_max:
                                    new_global_max = r[-1]
                            
                            if new_global_max < max_val:
                                return rows, used_diffs, new_global_max, True
                            else:
                                # Revert
                                rows[r_idx] = row
                                used_diffs = (used_diffs - new_diffs) | old_diffs
                                break # Found one valid move, but it didn't reduce global max?
                                     # If it didn't reduce global max, it's not an improvement for our objective.
                                     # We should only accept moves that reduce the global max.
        
        return rows, used_diffs, max_val, False

    # Iterated Local Search
    while time.time() < deadline:
        # Local search phase
        current_rows, current_used, current_max, improved = local_search(
            current_rows, current_used, current_max, deadline - 0.1
        )
        
        if current_max < best_max:
            best_max = current_max
            best_rows = [row[:] for row in current_rows]
        
        # If no improvement, kick
        if not improved:
            # Apply a kick: random valid perturbation
            # Pick a random row and a random split, shift by a random small amount
            # Check validity, accept if valid (even if max increases)
            r_idx = random.randint(0, n_rows - 1)
            s_idx = random.randint(1, n - 1)
            delta = random.randint(-3, 3)
            
            row = current_rows[r_idx]
            new_row = row[:]
            for j in range(s_idx, n):
                new_row[j] += delta
            
            if is_valid_row(new_row):
                old_diffs = get_row_diffs(row)
                new_diffs = get_row_diffs(new_row)
                other_diffs = current_used - old_diffs
                if not (new_diffs & other_diffs):
                    # Accept kick
                    current_rows[r_idx] = new_row
                    current_used = (current_used - old_diffs) | new_diffs
                    current_max = max(max(r) for r in current_rows)
                    
                    # If this kick improved the max, update best
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [row[:] for row in current_rows]

    return best_rows
