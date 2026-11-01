import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    
    # Safety margin: The evaluator kills at 150s. We aim to finish by 145s to be safe.
    deadline = _START + 145.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check if rows are valid (strictly increasing, start at 0, unique diffs)."""
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.1, window=20, seed=None):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
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
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def rebuild_row(used_other, jitter=0.0, max_val=None, rng_seed=None):
        """Rebuild a single row given differences already used by other rows."""
        if rng_seed is not None:
            local_rng = random.Random(rng_seed)
        else:
            local_rng = random.Random()

        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            search_end = max_val if max_val else (1000)
            # Ensure search_end is at least x+1
            if search_end <= x:
                search_end = x + 100
                
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter and trial_x < 300:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def targeted_descent(rows, time_budget):
        """
        Aggressively try to reduce the scope by targeting the largest elements.
        Returns (new_rows, new_scope) or (None, None) if no improvement.
        """
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False
        
        # Cache all diffs for quick fixed_diffs calculation?
        # Actually, recalculating fixed_diffs for each move is O(N^2) which might be slow if N is large.
        # But N is small (4 rows, 8 elements). Total diffs ~ 112.
        # Let's optimize the check.
        
        def get_fixed_diffs(r_idx, c_idx, current_val):
            """Get set of diffs that do NOT involve cur[r_idx][c_idx]."""
            fixed = set()
            # Add all diffs from other rows
            for i in range(n_rows):
                if i == r_idx:
                    continue
                row = cur[i]
                for a in range(len(row)):
                    for b in range(a+1, len(row)):
                        fixed.add(row[b] - row[a])
            
            # Add diffs from the same row that don't involve c_idx
            row = cur[r_idx]
            for a in range(len(row)):
                if a == c_idx:
                    continue
                for b in range(a+1, len(row)):
                    if b == c_idx:
                        continue
                    fixed.add(row[b] - row[a])
                    
            return fixed

        # Main loop: try to reduce max element
        while time.time() < start + time_budget:
            # Find global max
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(cur):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break # Cannot reduce 0
            
            current_val = max_val
            prev_val = cur[max_r_idx][max_c_idx - 1]
            
            fixed_diffs = get_fixed_diffs(max_r_idx, max_c_idx, current_val)
            
            found_improvement = False
            # Try values from current_val - 1 down to prev_val + 1
            # We only care about reducing the scope, so we stop at the first valid reduction
            # that lowers the global max. However, sometimes reducing a non-max element
            # allows a larger reduction later. But for now, let's just try to lower the max.
            
            # Optimization: If current_val is not the unique global max, reducing it might not lower the scope.
            # But it frees up differences.
            
            row = cur[max_r_idx]
            
            # We want to try new_val < current_val.
            # The new diffs are {new_val - row[i] for i < c_idx} and {row[j] - new_val for j > c_idx}.
            
            # To speed up, we can iterate new_val downwards.
            # If we find a valid new_val, we accept it.
            
            # Limit search range to avoid O(N^2) per step if range is huge.
            # But range is bounded by prev_val.
            
            # Let's try a few random candidates first to jump gaps?
            # No, strict descent is safer for validity.
            
            # Try decreasing values
            # If the gap is large, this might be slow.
            # But typically gaps are small in these problems.
            
            # Check if we can just decrement by 1?
            # Let's try a window of values.
            
            # Actually, the "fixed_diffs" check is O(1) per diff check if we use a set.
            # Number of diffs to check per new_val is O(k).
            # Range of new_vals is O(scope).
            # So O(scope * k) per move.
            
            # Let's try to find ANY valid new_val < current_val.
            # We prefer the largest valid new_val to keep things tight?
            # Or the smallest?
            # Usually, keeping elements as small as possible is good for future flexibility,
            # but we want to reduce the MAX. So if we are reducing the max, any valid reduction helps.
            # If we are reducing a non-max element, we want to make it small to free up space?
            
            # For now, let's just find the largest valid new_val < current_val.
            # This maintains the "tightness" of the row.
            
            valid_new_val = None
            # Start from current_val - 1 and go down
            # But if current_val is huge and prev_val is small, this is slow.
            # Let's try a few random steps first?
            # Or just step down by 1.
            
            # Heuristic: Try current_val - 1, current_val - 2, ... up to 20 steps.
            # If not found, try random samples in [prev_val+1, current_val-1].
            
            checked = set()
            for step in range(20):
                nv = current_val - step
                if nv <= prev_val:
                    break
                if nv in checked:
                    continue
                checked.add(nv)
                
                # Check validity
                valid = True
                new_diffs = set()
                
                # Left diffs
                for i in range(max_c_idx):
                    d = nv - row[i]
                    if d <= 0:
                        valid = False
                        break
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Right diffs
                for j in range(max_c_idx + 1, len(row)):
                    d = row[j] - nv
                    if d <= 0:
                        valid = False
                        break
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    valid_new_val = nv
                    break
            
            if valid_new_val is None:
                # Try random samples if the gap is large
                if current_val - prev_val > 50:
                    for _ in range(10):
                        nv = random.randint(prev_val + 1, current_val - 1)
                        if nv in checked:
                            continue
                        checked.add(nv)
                        
                        valid = True
                        new_diffs = set()
                        
                        for i in range(max_c_idx):
                            d = nv - row[i]
                            if d <= 0:
                                valid = False
                                break
                            if d in fixed_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if not valid:
                            continue
                        
                        for j in range(max_c_idx + 1, len(row)):
                            d = row[j] - nv
                            if d <= 0:
                                valid = False
                                break
                            if d in fixed_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if valid:
                            valid_new_val = nv
                            break
            
            if valid_new_val is not None:
                row[max_c_idx] = valid_new_val
                new_scope = get_scope(cur)
                if new_scope < cur_scope:
                    cur_scope = new_scope
                    improved = True
                # Even if scope didn't drop (because another row has the same max),
                # we accept the move to free up differences.
                continue
            
            # If we couldn't reduce the global max, try reducing other large elements
            # to potentially allow the max to be reduced later, or just to tighten up.
            
            # Pick a random non-max element that is relatively large
            candidates = []
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if r_idx == max_r_idx and c_idx == max_c_idx:
                        continue
                    # Only consider elements that are part of the "top" of the distribution
                    # Or just any element > 10
                    if cur[r_idx][c_idx] > 10:
                        candidates.append((r_idx, c_idx))
            
            if not candidates:
                break
            
            r_idx, c_idx = random.choice(candidates)
            current_val = cur[r_idx][c_idx]
            prev_val = cur[r_idx][c_idx - 1]
            
            fixed_diffs = get_fixed_diffs(r_idx, c_idx, current_val)
            row = cur[r_idx]
            
            valid_new_val = None
            checked = set()
            
            # Try to reduce this element
            for step in range(10):
                nv = current_val - step
                if nv <= prev_val:
                    break
                if nv in checked:
                    continue
                checked.add(nv)
                
                valid = True
                new_diffs = set()
                
                for i in range(c_idx):
                    d = nv - row[i]
                    if d <= 0:
                        valid = False
                        break
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                for j in range(c_idx + 1, len(row)):
                    d = row[j] - nv
                    if d <= 0:
                        valid = False
                        break
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    valid_new_val = nv
                    break
            
            if valid_new_val is not None:
                row[c_idx] = valid_new_val
                # Don't update cur_scope here unless it actually improves the max?
                # Actually, if we reduce an element, the max might stay the same, but we made progress.
                # We should accept this move to keep the search moving.
                # However, if we accept moves that don't reduce scope, we might wander.
                # Let's only accept if it reduces scope OR if we are stuck.
                # For now, let's just accept it to change the state.
                new_scope = get_scope(cur)
                if new_scope < cur_scope:
                    cur_scope = new_scope
                    improved = True
                else:
                    # If it didn't improve scope, we might want to revert?
                    # No, let's keep it to escape local minima.
                    # But we need to be careful not to increase scope.
                    # Since we only decrease values, scope cannot increase.
                    pass
            
            if not improved and time.time() > start + time_budget * 0.8:
                # Stagnation?
                break

        return cur, cur_scope

    def kick(rows, num_kicks=1):
        """Perturb by rebuilding random rows with high jitter."""
        rows = [list(r) for r in rows]
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    row = rows[j]
                    for i in range(len(row)):
                        for l in range(i+1, len(row)):
                            used_other.add(row[l] - row[i])
            
            success = False
            for attempt in range(10):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, jitter=jitter, max_val=get_scope(rows) + 20)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                # Fallback: greedy rebuild
                new_row = rebuild_row(used_other, jitter=0.0, max_val=get_scope(rows) + 30)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    # Phase 1: Initialization
    best_rows = greedy_init(rand_prob=0.1, seed=42)
    best_scope = get_scope(best_rows)
    
    # Phase 2: ILS with Targeted Descent
    max_iters = 1000
    for it in range(max_iters):
        if time.time() > deadline - 20:
            break

        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            # Kick every few iterations to escape local minima
            if it % 5 == 0:
                cur = kick(best_rows, num_kicks=random.choice([1, 1, 2]))
            else:
                cur = [r[:] for r in best_rows]
                
        # Ensure validity after kick (rebuild_row should guarantee it, but check)
        if not validate(cur):
            continue

        cur_scope = get_scope(cur)
        
        # Apply Targeted Descent
        # Give it a budget of 5 seconds per iteration
        desc_rows, desc_scope = targeted_descent(cur, time_budget=5.0)
        
        if desc_rows is not None:
            if desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
            # Even if not better, we might want to keep the improved state for the next kick?
            # Usually ILS keeps the best.
            
    # Phase 3: Final Polish
    # Try to squeeze out the last bit by systematically reducing the last element of each row
    if time.time() < deadline - 5:
        cur = [r[:] for r in best_rows]
        cur_scope = get_scope(cur)
        
        # Systematic reduction of last elements
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for r_idx in range(n_rows):
                row = cur[r_idx]
                c_idx = len(row) - 1
                current_val = row[c_idx]
                prev_val = row[c_idx - 1]
                
                # Try to reduce
                for nv in range(current_val - 1, prev_val, -1):
                    # Quick check
                    fixed_diffs = set()
                    for i in range(n_rows):
                        if i == r_idx:
                            continue
                        r = cur[i]
                        for a in range(len(r)):
                            for b in range(a+1, len(r)):
                                fixed_diffs.add(r[b] - r[a])
                    
                    # Same row, excluding c_idx
                    for a in range(len(row)):
                        if a == c_idx:
                            continue
                        for b in range(a+1, len(row)):
                            if b == c_idx:
                                continue
                            fixed_diffs.add(row[b] - row[a])
                    
                    valid = True
                    new_diffs = set()
                    for i in range(c_idx):
                        d = nv - row[i]
                        if d <= 0 or d in fixed_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    for j in range(c_idx + 1, len(row)):
                        # No j > c_idx for last element
                        pass
                    
                    if valid:
                        row[c_idx] = nv
                        new_scope = get_scope(cur)
                        if new_scope < cur_scope:
                            cur_scope = new_scope
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [r[:] for r in cur]
                        improved = True
                        break # Restart loop with new state? Or just continue?
                        # Breaking inner loop, but 'improved' is True so outer while continues.
                        # But we modified row, so we should probably restart the scan of rows
                        # or at least this row.
                        # Let's just break and let the while loop restart.
                        break
        
        # One last targeted descent on the best
        if time.time() < deadline - 1:
            desc_rows, desc_scope = targeted_descent(best_rows, time_budget=1.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    return best_rows
