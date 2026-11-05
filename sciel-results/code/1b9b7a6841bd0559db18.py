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

    def get_M(rows):
        return max(max(r) for r in rows)

    def compute_energy(M, penalty, T):
        if penalty == 0:
            return float(M)
        # Strong penalty for conflicts to drive towards validity
        # M is the primary objective
        lambda_pen = 50.0 + 50.0 * T 
        return M + lambda_pen * penalty

    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # 3. Improved Row Rebuild Strategy
    # Uses a scoring function that balances row length and difference spread
    def rebuild_row(row_idx, rows, forbidden, target_M, rng, max_iters=5000):
        """
        Rebuilds rows[row_idx] such that all diffs are valid against other rows 
        and the row itself is valid. 
        Returns the new row if successful, None otherwise.
        """
        # forbidden is the set of diffs used by other rows
        
        best_row = None
        best_max = target_M + 100

        for attempt in range(max_iters // 10):
            if time.time() > deadline:
                break
                
            row = [0]
            local_used = set()
            success = True
            
            for step in range(k):
                last = row[-1]
                found = False
                
                # Collect valid candidates in a window
                candidates = []
                search_limit = min(target_M + 20, last + 50)
                
                for x in range(last + 1, search_limit + 1):
                    new_diffs = {x - a for a in row}
                    if (new_diffs & local_used) or (new_diffs & forbidden):
                        continue
                    
                    # Score: prefer smaller x, but also penalize large max diff to keep row tight
                    max_new_diff = max(new_diffs)
                    score = 0.8 * x + 0.2 * max_new_diff
                    candidates.append((x, new_diffs, score))
                    
                    if len(candidates) > 20:
                        break
                
                if not candidates:
                    success = False
                    break
                
                # Pick randomly from top candidates, weighted towards lower scores
                # Sort by score first
                candidates.sort(key=lambda t: t[2])
                n_top = min(5, len(candidates))
                top_candidates = candidates[:n_top]
                
                weights = [1.0 / (1.0 + c[2] * 0.1) for c in top_candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0.0
                chosen = top_candidates[0]
                for c, w in zip(top_candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = c
                        break
                
                x, new_diffs, _ = chosen
                row.append(x)
                local_used.update(new_diffs)
                
                if chosen > target_M:
                    if chosen > target_M + 10:
                        success = False
                        break

            if success:
                if row[-1] <= target_M:
                    if best_row is None or row[-1] < best_row[-1]:
                        best_row = row
                        best_max = row[-1]
                elif best_row is None or row[-1] < best_max:
                    best_row = row
                    best_max = row[-1]
                    
                if best_row and best_row[-1] <= target_M:
                    break

        return best_row

    # Multi-row rebuild: try to rebuild 2 rows simultaneously
    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit, row_i, row_j):
        """
        Try to rebuild rows i and j simultaneously.
        Returns (new_row_i, new_row_j) or (None, None) if failed.
        """
        # First, build row i with forbidden_diffs
        # We need a simplified rebuild that returns just the row, not using row_idx
        # We can reuse the logic but it's cleaner to write a helper or adapt
        # Let's adapt rebuild_row to take forbidden set directly
        
        # For now, we'll just call a simplified version
        # Actually, let's just implement a direct rebuild for this
        def _build_row(forbidden, limit):
            row = [0]
            local_used = set()
            for step in range(k):
                last = row[-1]
                candidates = []
                search_limit = min(limit + 20, last + 50)
                for x in range(last + 1, search_limit + 1):
                    new_diffs = {x - a for a in row}
                    if (new_diffs & local_used) or (new_diffs & forbidden):
                        continue
                    max_new_diff = max(new_diffs)
                    score = 0.8 * x + 0.2 * max_new_diff
                    candidates.append((x, new_diffs, score))
                    if len(candidates) > 20:
                        break
                if not candidates:
                    return None
                candidates.sort(key=lambda t: t[2])
                n_top = min(5, len(candidates))
                top_candidates = candidates[:n_top]
                weights = [1.0 / (1.0 + c[2] * 0.1) for c in top_candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0.0
                chosen = top_candidates[0]
                for c, w in zip(top_candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = c
                        break
                x, new_diffs, _ = chosen
                row.append(x)
                local_used.update(new_diffs)
                if x > limit:
                    if x > limit + 10:
                        return None
            return row

        row_i_new = _build_row(forbidden_diffs, max_val_limit)
        if row_i_new is None:
            return None, None
        
        diffs_i = get_row_diffs(row_i_new)
        
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = _build_row(forbidden_j, max_val_limit)
        if row_j_new is None:
            return None, None
        
        return row_i_new, row_j_new

    # Multi-start SA
    best_rows = None
    best_M = float('inf')
    
    # Number of restarts
    max_restarts = 3
    time_per_restart = (deadline - start_time) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if restart == 0:
            curr_rows = greedy_init()
        else:
            # Random perturbation of the best solution found so far, or fresh greedy
            if random.random() < 0.5:
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
            polish_mode = (local_deadline - time.time()) < 15.0
            
            # Decide move type: Single Element vs Row Rebuild vs Two Row Rebuild
            # Row rebuild is expensive but powerful. Use it less often, or more when stuck.
            if polish_mode:
                use_rebuild = False
                use_two_rebuild = False
            else:
                if no_improve > 50:
                    use_rebuild = random.random() < 0.3
                    use_two_rebuild = random.random() < 0.1 if n_rows >= 2 else False
                else:
                    use_rebuild = random.random() < 0.05
                    use_two_rebuild = random.random() < 0.02 if n_rows >= 2 else False
            
            if use_two_rebuild:
                # Pick two distinct rows to rebuild
                r_idx1 = random.randint(0, n_rows - 1)
                r_idx2 = random.randint(0, n_rows - 1)
                while r_idx2 == r_idx1:
                    r_idx2 = random.randint(0, n_rows - 1)
                
                old_row1 = curr_rows[r_idx1]
                old_row2 = curr_rows[r_idx2]
                
                old_diffs1 = get_row_diffs(old_row1)
                old_diffs2 = get_row_diffs(old_row2)
                
                # Remove old diffs from counts
                for d in old_diffs1:
                    curr_counts[d] -= 1
                    if curr_counts[d] <= 0:
                        del curr_counts[d]
                for d in old_diffs2:
                    curr_counts[d] -= 1
                    if curr_counts[d] <= 0:
                        del curr_counts[d]
                
                # Recalculate penalty
                curr_penalty = get_penalty(curr_counts)
                
                target_M = curr_M
                if polish_mode:
                    target_M = curr_M - 1
                
                new_row1, new_row2 = rebuild_two_rows(k, set(curr_counts.keys()), random, target_M, r_idx1, r_idx2)
                
                if new_row1 is None or new_row2 is None:
                    # Restore old rows
                    for d in old_diffs1:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    for d in old_diffs2:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    curr_rows[r_idx1] = old_row1
                    curr_rows[r_idx2] = old_row2
                    curr_penalty = get_penalty(curr_counts)
                    curr_M = get_M(curr_rows)
                    curr_energy = compute_energy(curr_M, curr_penalty, T)
                    no_improve += 1
                    continue
                
                new_diffs1 = get_row_diffs(new_row1)
                new_diffs2 = get_row_diffs(new_row2)
                
                # Check for conflicts between the two new rows
                if new_diffs1 & new_diffs2:
                    # Conflict, revert
                    curr_rows[r_idx1] = old_row1
                    curr_rows[r_idx2] = old_row2
                    for d in old_diffs1:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    for d in old_diffs2:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    curr_penalty = get_penalty(curr_counts)
                    curr_M = get_M(curr_rows)
                    curr_energy = compute_energy(curr_M, curr_penalty, T)
                    no_improve += 1
                    continue
                
                # Add new diffs
                for d in new_diffs1:
                    curr_counts[d] = curr_counts.get(d, 0) + 1
                for d in new_diffs2:
                    curr_counts[d] = curr_counts.get(d, 0) + 1
                
                curr_rows[r_idx1] = new_row1
                curr_rows[r_idx2] = new_row2
                
                curr_penalty = get_penalty(curr_counts)
                curr_M = get_M(curr_rows)
                new_energy = compute_energy(curr_M, curr_penalty, T)
                
                delta_E = new_energy - curr_energy
                
                if delta_E < 0 or (not polish_mode and random.random() < math.exp(-delta_E / T)):
                    curr_energy = new_energy
                    if curr_penalty == 0 and curr_M < best_M:
                        best_M = curr_M
                        best_rows = copy.deepcopy(curr_rows)
                        no_improve = 0
                    elif curr_penalty == 0:
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    curr_rows[r_idx1] = old_row1
                    curr_rows[r_idx2] = old_row2
                    for d in new_diffs1:
                        curr_counts[d] -= 1
                        if curr_counts[d] <= 0:
                            del curr_counts[d]
                    for d in new_diffs2:
                        curr_counts[d] -= 1
                        if curr_counts[d] <= 0:
                            del curr_counts[d]
                    for d in old_diffs1:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    for d in old_diffs2:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    curr_penalty = get_penalty(curr_counts)
                    curr_M = get_M(curr_rows)
                    curr_energy = compute_energy(curr_M, curr_penalty, T)
                    no_improve += 1

            elif use_rebuild:
                # Pick a row to rebuild
                r_idx = random.randint(0, n_rows - 1)
                old_row = curr_rows[r_idx]
                
                # Remove old diffs from counts
                old_diffs = get_row_diffs(old_row)
                for d in old_diffs:
                    if d in curr_counts:
                        curr_counts[d] -= 1
                        if curr_counts[d] <= 0:
                            del curr_counts[d]
                        
                curr_penalty = get_penalty(curr_counts)
                
                target_M = curr_M
                if polish_mode:
                    target_M = curr_M - 1
                
                new_row = rebuild_row(r_idx, curr_rows, set(curr_counts.keys()), target_M, random, max_iters=2000)
                
                if new_row is None:
                    for d in old_diffs:
                        curr_counts[d] = curr_counts.get(d, 0) + 1
                    curr_penalty = get_penalty(curr_counts)
                    curr_M = get_M(curr_rows)
                    curr_energy = compute_energy(curr_M, curr_penalty, T)
                    no_improve += 1
                    continue
                
                new_diffs = get_row_diffs(new_row)
                
                # Check if new row conflicts with others (should be handled by rebuild, but verify)
                # rebuild_row uses forbidden set, so it should be fine.
                
                for d in new_diffs:
                    curr_counts[d] = curr_counts.get(d, 0) + 1
                
                curr_rows[r_idx] = new_row
                curr_penalty = get_penalty(curr_counts)
                curr_M = get_M(curr_rows)
                new_energy = compute_energy(curr_M, curr_penalty, T)
                
                delta_E = new_energy - curr_energy
                
                if delta_E < 0 or (not polish_mode and random.random() < math.exp(-delta_E / T)):
                    curr_energy = new_energy
                    if curr_penalty == 0 and curr_M < best_M:
                        best_M = curr_M
                        best_rows = copy.deepcopy(curr_rows)
                        no_improve = 0
                    elif curr_penalty == 0:
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    curr_rows[r_idx] = old_row
                    for d in new_diffs:
                        curr_counts[d] -= 1
                        if curr_counts[d] <= 0:
                            del curr_counts[d]
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
                high = row[j_idx + 1] - 1 if j_idx < k else curr_M + 10
                
                if high < low:
                    continue
                    
                new_val = random.randint(low, high)
                
                if new_val == old_val:
                    continue
                
                # Remove old diffs
                for i in range(j_idx):
                    d = old_val - row[i]
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
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
                        
                row[j_idx] = new_val
                
                # Add new diffs
                for i in range(j_idx):
                    d = new_val - row[i]
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
                    
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
                    
                other_max = 0
                for r in curr_rows:
                    if r is not row:
                        if max(r) > other_max:
                            other_max = max(r)
                new_M = max(max(row), other_max)
                
                new_energy = compute_energy(new_M, curr_penalty, T)
                delta_E = new_energy - curr_energy
                
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
                    row[j_idx] = old_val
                    
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

    if best_rows is None:
        best_rows = greedy_init()
        
    return best_rows
