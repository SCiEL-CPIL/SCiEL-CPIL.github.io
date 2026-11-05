import time
import random
import copy
import math

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s, task says 120s. Use 110s to be safe.
    deadline = start_time + 110
    
    # 1. Greedy Initialization
    def greedy_init(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                diffs = {x - a for a in row}
                if not (diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(diffs)
            rows.append(row)
        return rows

    # 2. State Helpers
    def get_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_penalty(counts):
        pen = 0
        for c in counts.values():
            if c > 1:
                pen += c - 1
        return pen

    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        counts = get_diff_counts(rows)
        return all(c == 1 for c in counts.values())

    def get_M(rows):
        return max(max(r) for r in rows)

    def compute_energy(M, penalty, T):
        if penalty == 0:
            return float(M)
        # Strong penalty for conflicts to drive towards validity
        # M is the primary objective
        lambda_pen = 50.0 + 50.0 * T 
        return M + lambda_pen * penalty

    def polish_solution(rows, deadline):
        """
        Final polish pass: try to reduce the max entry by attempting
        to move individual elements to smaller values.
        """
        if rows is None:
            return rows
            
        best = [list(r) for r in rows]
        best_M = max(max(r) for r in best)
        best_penalty = get_penalty(get_diff_counts(best))
        
        if best_penalty > 0:
            return best
            
        # Try to reduce the maximum entry
        # Find which row has the max entry
        max_row_idx = 0
        for i, r in enumerate(best):
            if r[-1] == best_M:
                max_row_idx = i
                break
                
        # Try to reduce elements in the row with the max entry
        # Work from the end backwards
        for attempt in range(200):
            if time.time() > deadline - 2:
                break
                
            improved = False
            # Try each position from the end
            for j in range(len(best[max_row_idx]) - 1, 0, -1):
                row = best[max_row_idx]
                old_val = row[j]
                
                # Try smaller values
                for new_val in range(row[j-1] + 1, old_val):
                    # Check if this creates conflicts
                    counts = get_diff_counts(best)
                    
                    # Remove old diffs involving position j
                    for i in range(j):
                        d = old_val - row[i]
                        counts[d] -= 1
                        if counts[d] == 0:
                            del counts[d]
                    for h in range(j+1, len(row)):
                        d = row[h] - old_val
                        counts[d] -= 1
                        if counts[d] == 0:
                            del counts[d]
                    
                    # Add new diffs
                    for i in range(j):
                        d = new_val - row[i]
                        counts[d] = counts.get(d, 0) + 1
                    for h in range(j+1, len(row)):
                        d = row[h] - new_val
                        counts[d] = counts.get(d, 0) + 1
                    
                    new_penalty = get_penalty(counts)
                    if new_penalty == 0:
                        # Accept the move
                        row[j] = new_val
                        improved = True
                        break
                    else:
                        # Revert
                        for i in range(j):
                            d = new_val - row[i]
                            counts[d] -= 1
                            if counts[d] == 0:
                                del counts[d]
                        for h in range(j+1, len(row)):
                            d = row[h] - new_val
                            counts[d] -= 1
                            if counts[d] == 0:
                                del counts[d]
                        for i in range(j):
                            d = old_val - row[i]
                            counts[d] = counts.get(d, 0) + 1
                        for h in range(j+1, len(row)):
                            d = row[h] - old_val
                            counts[d] = counts.get(d, 0) + 1
                        
                if improved:
                    break
                    
            if not improved:
                break
                
            # Update max row index
            best_M = max(max(r) for r in best)
            max_row_idx = 0
            for i, r in enumerate(best):
                if r[-1] == best_M:
                    max_row_idx = i
                    break
                    
        return best

    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # 3. Row Rebuild Strategy (Inspiration)
    # Attempts to rebuild a specific row to fit within a target max M
    def rebuild_row(row_idx, rows, used_diffs, target_M, rng, max_iters=5000):
        """
        Rebuilds rows[row_idx] such that all diffs are valid against other rows 
        and the row itself is valid. 
        Returns the new row if successful, None otherwise.
        """
        # Get forbidden diffs from other rows
        forbidden = set()
        for i in range(n_rows):
            if i == row_idx:
                continue
            forbidden.update(get_row_diffs(rows[i]))
        
        # We also need to ensure the new row doesn't conflict with itself
        # We build it incrementally
        
        # Strategy: Backtracking search for the row elements
        # To speed up, we don't do full backtracking, but a greedy-with-randomness
        # that restarts if stuck.
        
        best_row = None
        best_max = target_M + 100 # Slightly above target to allow some slack if needed
        
        for attempt in range(max_iters // 10):
            if time.time() > deadline:
                break
                
            row = [0]
            local_used = set()
            success = True
            
            for step in range(k):
                last = row[-1]
                # We want to find x > last such that:
                # 1. x - a is not in local_used for any a in row
                # 2. x - a is not in forbidden for any a in row
                # 3. Ideally x <= target_M
                
                found = False
                # Search range: last+1 to target_M (or slightly higher if we allow overflow)
                # To be efficient, we don't scan every integer if target_M is large.
                # However, k is small (8), so the row length is 9.
                # The gaps are usually small.
                
                # Randomized Greedy Selection
                # Collect valid candidates in a small window
                candidates = []
                search_limit = min(target_M + 20, last + 50) # Look ahead 50 or to target+20
                
                for x in range(last + 1, search_limit + 1):
                    new_diffs = {x - a for a in row}
                    if (new_diffs & local_used) or (new_diffs & forbidden):
                        continue
                    # Score: prefer smaller x, but also prefer x that doesn't create "tight" constraints?
                    # Simple score: x
                    candidates.append(x)
                    
                    # Don't collect too many, pick early to save time
                    if len(candidates) > 20:
                        break
                
                if not candidates:
                    success = False
                    break
                
                # Pick randomly from top candidates, weighted towards smaller values
                # Weight: 1 / (1 + x)
                weights = [1.0 / (1.0 + c) for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0.0
                chosen = candidates[0]
                for c, w in zip(candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = c
                        break
                
                row.append(chosen)
                local_used.update({chosen - a for a in row[:-1]}) # Update local used
                
                if chosen > target_M:
                    # We exceeded the target, but might still be valid.
                    # If we exceed too much, abort this attempt.
                    if chosen > target_M + 10:
                        success = False
                        break

            if success:
                # Verify the row is strictly increasing and valid
                if row[-1] <= target_M:
                    if best_row is None or row[-1] < best_row[-1]:
                        best_row = row
                        best_max = row[-1]
                elif best_row is None or row[-1] < best_max:
                    # Accept overflow if it's the best we have
                    best_row = row
                    best_max = row[-1]
                    
                # If we found a row within target, we can stop early
                if best_row and best_row[-1] <= target_M:
                    break

        return best_row

    # Multi-start SA
    best_rows = None
    best_M = float('inf')
    
    # Number of restarts - use more restarts with shorter time each
    max_restarts = 5
    time_per_restart = (deadline - start_time - 20) / max_restarts  # Reserve 20s for polish
    
    for restart in range(max_restarts):
        if time.time() > deadline - 25:
            break
            
        # Initialize
        if restart == 0:
            curr_rows = greedy_init()
        else:
            # Random perturbation of the best solution found so far, or fresh greedy
            if random.random() < 0.4:
                curr_rows = greedy_init()
            else:
                curr_rows = copy.deepcopy(best_rows)
                # Perturb some elements
                for _ in range(3):
                    r_idx = random.randint(0, n_rows - 1)
                    j_idx = random.randint(1, k)
                    row = curr_rows[r_idx]
                    low = row[j_idx - 1] + 1
                    high = row[j_idx + 1] - 1 if j_idx < k else max(max(r) for r in curr_rows) + 10
                    if high >= low:
                        row[j_idx] = random.randint(low, high)
                        # Fix ordering if violated
                        for i in range(1, k + 1):
                            if row[i] <= row[i-1]:
                                row[i] = row[i-1] + 1
        
        curr_counts = get_diff_counts(curr_rows)
        curr_penalty = get_penalty(curr_counts)
        curr_M = get_M(curr_rows)
        
        # SA Parameters
        T = 50.0
        cooling = 0.999
        curr_energy = compute_energy(curr_M, curr_penalty, T)
        
        # Local restart deadline
        local_deadline = start_time + (restart + 1) * time_per_restart
        
        # Stagnation counter
        no_improve = 0
        
        # SA Loop
        while time.time() < local_deadline:
            # Check for polish mode
            polish_mode = (local_deadline - time.time()) < 10.0
            
            # Decide move type: Single Element vs Row Rebuild
            # Row rebuild is expensive but powerful. Use it less often, or more when stuck.
            if polish_mode:
                use_rebuild = False
            else:
                if no_improve > 30:
                    use_rebuild = random.random() < 0.25
                else:
                    use_rebuild = random.random() < 0.03
            
            if use_rebuild:
                # Pick a row to rebuild - prefer the row with the max entry
                r_idx = random.randint(0, n_rows - 1)
                # 30% chance to pick the row with the max entry
                if random.random() < 0.3:
                    max_val = -1
                    for i, r in enumerate(curr_rows):
                        if r[-1] > max_val:
                            max_val = r[-1]
                            r_idx = i
                old_row = curr_rows[r_idx]
                
                # Remove old diffs from counts
                old_diffs = get_row_diffs(old_row)
                for d in old_diffs:
                    if d in curr_counts:
                        curr_counts[d] -= 1
                        if curr_counts[d] <= 0:
                            del curr_counts[d]
                        
                # Recalculate penalty
                curr_penalty = get_penalty(curr_counts)
                
                # Determine target M for rebuild
                # Try to fit within current M, or slightly smaller if we are polishing
                target_M = curr_M
                if polish_mode:
                    target_M = curr_M - 1 # Try to improve
                
                # Rebuild
                new_row = rebuild_row(r_idx, curr_rows, None, target_M, random, max_iters=1500)
                
                if new_row is None:
                    # Restore old row
                    for d in old_diffs:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    # Recalculate penalty and M
                    curr_penalty = get_penalty(curr_counts)
                    curr_M = get_M(curr_rows)
                    curr_energy = compute_energy(curr_M, curr_penalty, T)
                    no_improve += 1
                    continue
                
                # Check if new row is valid against others
                new_diffs = get_row_diffs(new_row)
                # Since rebuild_row ensures no conflict with 'forbidden' (other rows),
                # we just need to add them back.
                
                # Update counts
                for d in new_diffs:
                    curr_counts[d] = curr_counts.get(d, 0) + 1
                
                curr_rows[r_idx] = new_row
                curr_penalty = get_penalty(curr_counts)
                curr_M = get_M(curr_rows)
                new_energy = compute_energy(curr_M, curr_penalty, T)
                
                delta_E = new_energy - curr_energy
                
                # Acceptance
                if delta_E < 0 or (not polish_mode and random.random() < math.exp(-delta_E / T)):
                    curr_energy = new_energy
                    if curr_penalty == 0 and curr_M < best_M:
                        best_M = curr_M
                        best_rows = copy.deepcopy(curr_rows)
                        no_improve = 0
                    elif curr_penalty == 0:
                        no_improve = 0 # Valid, but not better M
                    else:
                        no_improve += 1
                else:
                    # Revert
                    curr_rows[r_idx] = old_row
                    # Remove new diffs
                    for d in new_diffs:
                        curr_counts[d] -= 1
                        if curr_counts[d] <= 0:
                            del curr_counts[d]
                    # Add old diffs
                    for d in old_diffs:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    curr_penalty = get_penalty(curr_counts)
                    curr_M = get_M(curr_rows)
                    curr_energy = compute_energy(curr_M, curr_penalty, T)
                    no_improve += 1

            else:
                # Standard Single Element Move
                r_idx = random.randint(0, n_rows - 1)
                row = curr_rows[r_idx]
                j_idx = random.randint(1, k)
                
                old_val = row[j_idx]
                
                low = row[j_idx - 1] + 1
                high = row[j_idx + 1] - 1 if j_idx < k else curr_M + 5
                
                if high < low:
                    continue
                    
                new_val = random.randint(low, high)
                
                if new_val == old_val:
                    continue
                
                # Remove old diffs and update penalty
                for i in range(j_idx):
                    d = old_val - row[i]
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1  # Removing one from a pair reduces penalty by 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                        
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - old_val
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                
                # Tentatively set new value
                row[j_idx] = new_val
                
                # Add new diffs and update penalty
                for i in range(j_idx):
                    d = new_val - row[i]
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1  # Adding one to an existing diff increases penalty by 1
                    curr_counts[d] = c + 1
                    
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
                    
                # Calculate new M
                other_max = 0
                for r in curr_rows:
                    if r is not row:
                        if max(r) > other_max:
                            other_max = max(r)
                new_M = max(max(row), other_max)
                
                new_energy = compute_energy(new_M, curr_penalty, T)
                delta_E = new_energy - curr_energy
                
                # Acceptance
                if delta_E < 0 or (not polish_mode and random.random() < math.exp(-delta_E / T)):
                    curr_energy = new_energy
                    curr_M = new_M
                    if curr_penalty == 0:
                        if curr_M < best_M:
                            best_M = curr_M
                            best_rows = copy.deepcopy(curr_rows)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    row[j_idx] = old_val
                    
                    # Remove new diffs and update penalty
                    for i in range(j_idx):
                        d = new_val - row[i]
                        c = curr_counts[d]
                        if c > 1:
                            curr_penalty -= 1
                        curr_counts[d] -= 1
                        if curr_counts[d] == 0:
                            del curr_counts[d]
                            
                    for h in range(j_idx + 1, k + 1):
                        d = row[h] - new_val
                        c = curr_counts[d]
                        if c > 1:
                            curr_penalty -= 1
                        curr_counts[d] -= 1
                        if curr_counts[d] == 0:
                            del curr_counts[d]
                            
                    # Add old diffs and update penalty
                    for i in range(j_idx):
                        d = old_val - row[i]
                        c = curr_counts.get(d, 0)
                        if c >= 1:
                            curr_penalty += 1
                        curr_counts[d] = c + 1
                            
                    for h in range(j_idx + 1, k + 1):
                        d = row[h] - old_val
                        c = curr_counts.get(d, 0)
                        if c >= 1:
                            curr_penalty += 1
                        curr_counts[d] = c + 1
                            
                    curr_M = max(max(r) for r in curr_rows)
                    curr_energy = compute_energy(curr_M, curr_penalty, T)
                    no_improve += 1

            # Cool down
            T *= cooling
            if T < 1.0:
                T = 1.0

    # Final polish pass
    if best_rows is not None and time.time() < deadline - 5:
        best_rows = polish_solution(best_rows, deadline)
    
    if best_rows is None:
        best_rows = greedy_init()
        
    return best_rows
