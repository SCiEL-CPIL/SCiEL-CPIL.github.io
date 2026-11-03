import time
import random
import math

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, rng, bias=1.0):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            candidates = []
            x = last + 1
            max_x = last + 60
            while x <= max_x and len(candidates) < 10:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
                x += 1
            if not candidates:
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        candidates.append((x, new_diffs))
                        break
                    x += 1
            weights = [1.0 / (c[0] - last + 1) ** bias for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    counts = {}
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                counts[d] = counts.get(d, 0) + 1
    return counts

def _local_search_optimize(rows, k, deadline, rng):
    n_rows = len(rows)
    rows = [list(r) for r in rows]
    diff_counts = _build_diff_counts(rows)
    
    # Calculate initial conflicts: sum of max(0, count - 1) for all diffs
    # Or simply sum of counts - (number of unique diffs)
    # A valid state has all counts <= 1.
    # Conflict score = sum(count[d] for d in diff_counts if count[d] > 1) - (number of such d) ?
    # Simpler: Total "excess" occurrences. If d appears 3 times, it contributes 2 to conflict score.
    # We want to minimize this score to 0.
    
    def get_conflict_score():
        score = 0
        for c in diff_counts.values():
            if c > 1:
                score += c - 1
        return score

    best_rows = [list(r) for r in rows]
    best_score = get_conflict_score()
    best_scope = _scope(rows) if best_score == 0 else float('inf')
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        current_score = get_conflict_score()
        
        # If valid, try to improve scope
        if current_score == 0:
            # Find the row with the largest max element
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            r = rows[max_row_idx]
            # Try to reduce the last element or any element
            # Pick a random index j > 0
            j = rng.randint(1, k)
            old_val = r[j]
            
            # Try to decrease old_val
            # New value must be > r[j-1] and < r[j+1] (if exists)
            lo = r[j-1] + 1
            hi = r[j+1] - 1 if j < k else old_val - 1
            
            if lo < hi:
                # Try a few random values in [lo, hi]
                for _ in range(5):
                    new_val = rng.randint(lo, hi)
                    if new_val == old_val:
                        continue
                    
                    # Check if this move creates conflicts
                    # Remove old diffs involving j
                    for t in range(k + 1):
                        if t != j:
                            d = abs(r[t] - old_val)
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                    
                    # Add new diffs involving j
                    new_conflicts = 0
                    valid_move = True
                    for t in range(k + 1):
                        if t != j:
                            d = abs(r[t] - new_val)
                            if d in diff_counts:
                                # If d already exists, this move increases its count
                                # It's a conflict if count becomes > 1? 
                                # Actually, if count[d] is 1, adding another makes it 2 (conflict).
                                # If count[d] is 0, adding makes it 1 (ok).
                                pass 
                            # We need to simulate the count change to see if score increases
                            # Let's just check if any new diff is already in diff_counts
                            if d in diff_counts:
                                # This specific diff already exists. 
                                # If we add it, count becomes count[d]+1.
                                # If count[d] was 1, it becomes 2 (conflict).
                                # If count[d] was >1, it becomes >2 (worse).
                                # So if d is in diff_counts, it's a potential conflict source.
                                # But wait, if we are removing old diffs, some might be removed.
                                # We already removed old diffs.
                                # So if d is in diff_counts now, it means it's used by OTHER rows/pairs.
                                # So adding it here creates a duplicate.
                                new_conflicts += 1
                            
                            # Tentatively add to check? No, just count conflicts.
                            # Actually, if d is in diff_counts, it means it's used elsewhere.
                            # So the new state will have a conflict for d.
                            
                            # We can just reject if new_conflicts > 0?
                            # Or accept with probability?
                            # Since we are in a valid state (score 0), any conflict makes score > 0.
                            # We want to stay valid if possible.
                            # So if new_conflicts > 0, this move invalidates the state.
                            # We should only do this if we accept a worse state (simulated annealing).
                            # But for now, let's just try to find a move that keeps it valid.
                            pass
                            
                    # If no new conflicts, the move is valid and reduces scope (if new_val < old_val)
                    if new_conflicts == 0:
                        # Apply move
                        r[j] = new_val
                        for t in range(k + 1):
                            if t != j:
                                d = abs(r[t] - new_val)
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                        # Update best
                        new_scope = _scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(x) for x in rows]
                        break
                    else:
                        # Revert tentative removal
                        for t in range(k + 1):
                            if t != j:
                                d = abs(r[t] - old_val)
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                        # Try another value
                        pass
            else:
                # Cannot reduce
                pass
                
        else:
            # Invalid state. Try to reduce conflicts.
            # Find a difference d with count > 1
            conflicting_diffs = [d for d, c in diff_counts.items() if c > 1]
            if not conflicting_diffs:
                continue
            
            d_conf = rng.choice(conflicting_diffs)
            
            # Find rows that contain this difference
            # We need to find pairs (row_idx, i, j) such that rows[row_idx][j] - rows[row_idx][i] == d_conf
            # This is expensive to scan all rows.
            # Optimization: Keep a list of (row_idx, i, j) for each diff?
            # That's too much memory/state.
            # Alternative: Just pick a random row, and check if it contains d_conf.
            # If not, pick another.
            
            # Better: Iterate through rows, find one that has d_conf.
            target_row_idx = None
            target_i = None
            target_j = None
            
            for ri in range(n_rows):
                r = rows[ri]
                # Check if d_conf is in this row
                # Since row is sorted, we can do a quick check
                found = False
                for i in range(k + 1):
                    # Check if r[i] + d_conf is in r[i+1:]
                    # Simple linear scan is O(k^2) per row. k is small (6).
                    for j in range(i + 1, k + 1):
                        if r[j] - r[i] == d_conf:
                            target_row_idx = ri
                            target_i = i
                            target_j = j
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if target_row_idx is None:
                # Should not happen if d_conf is in diff_counts
                continue
                
            r = rows[target_row_idx]
            
            # Try to move r[target_j] or r[target_i]
            # Moving r[target_j] (the larger one) is usually easier to reduce scope or change diffs
            idx_to_move = target_j
            
            old_val = r[idx_to_move]
            
            # Try to change old_val to a new_val that eliminates d_conf
            # The new diffs involving idx_to_move should NOT include d_conf
            # Also, we hope to not create other conflicts.
            
            # Range for new_val
            lo = r[idx_to_move - 1] + 1
            hi = r[idx_to_move + 1] - 1 if idx_to_move < k else old_val + 100
            
            # We want to try values that don't create d_conf
            # d_conf = old_val - r[target_i]
            # If we change old_val to new_val, the diff with r[target_i] becomes new_val - r[target_i].
            # We want new_val - r[target_i] != d_conf.
            
            # Try a few random values
            found_improvement = False
            for _ in range(20):
                if hi <= lo:
                    break
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue
                
                # Check if this move creates d_conf
                # Diff with r[target_i]
                d_new = new_val - r[target_i]
                if d_new == d_conf:
                    continue
                
                # Check if this move creates any OTHER conflicts
                # Remove old diffs
                for t in range(k + 1):
                    if t != idx_to_move:
                        d = abs(r[t] - old_val)
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                
                # Check new diffs
                valid = True
                for t in range(k + 1):
                    if t != idx_to_move:
                        d = abs(r[t] - new_val)
                        if d in diff_counts:
                            # This diff already exists elsewhere.
                            # Adding it will increase its count.
                            # If count was 1, it becomes 2 (conflict).
                            # If count was >1, it becomes >2 (worse).
                            # So this creates a conflict.
                            # We can still accept if it reduces total conflict score?
                            # Let's calculate delta score.
                            pass
                        # Tentatively add
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                        
                new_score = get_conflict_score()
                old_score = current_score
                
                # Accept if new_score < old_score
                # Or with probability if new_score == old_score (neutral)
                # Or with probability if new_score > old_score (worse, but might escape local min)
                
                if new_score < old_score:
                    found_improvement = True
                    # Apply
                    r[idx_to_move] = new_val
                    # Update best if valid
                    if new_score == 0:
                        new_scope = _scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(x) for x in rows]
                    break
                elif new_score == old_score:
                    if rng.random() < 0.5:
                        found_improvement = True
                        r[idx_to_move] = new_val
                        break
                else:
                    # Worse. Accept with probability exp(-delta/T)
                    delta = new_score - old_score
                    if rng.random() < math.exp(-delta / T):
                        found_improvement = True
                        r[idx_to_move] = new_val
                        break
                    else:
                        # Revert tentative addition
                        for t in range(k + 1):
                            if t != idx_to_move:
                                d = abs(r[t] - new_val)
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                        # Re-add old diffs
                        for t in range(k + 1):
                            if t != idx_to_move:
                                d = abs(r[t] - old_val)
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                        continue
            
            if not found_improvement:
                # Revert all changes if we didn't break out with a move
                # Wait, the logic above reverts on rejection.
                # But if we break out with acceptance, we don't revert.
                # If we loop through 20 times and reject all, we are left with the state after the last rejection's revert?
                # No, the revert happens inside the loop on rejection.
                # So if we finish the loop without breaking, the state is unchanged.
                pass

        T = max(T * cooling, min_T)

    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy Initialization
    init_deadline = start_time + 5.0
    candidates = []
    
    while time.time() < init_deadline:
        bias = rng.uniform(0.5, 2.0)
        rows = _greedy_construct(n_rows, k, rng, bias=bias)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
        
    if not candidates:
        # Fallback
        rows = _greedy_construct(n_rows, k, rng, bias=1.0)
        if _valid(rows, k):
            best_rows = rows
            best_scope = _scope(rows)
        else:
            return [[0, 1, 2, 3, 4, 5]] * n_rows
    
    # Phase 2: Local Search Optimization on best candidates
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:5]
    
    remaining_time = deadline - time.time()
    time_per_ls = remaining_time / len(top_candidates)
    
    for i, (s, rows) in enumerate(top_candidates):
        if time.time() >= deadline - 5:
            break
        
        ls_deadline = time.time() + time_per_ls * 0.9
        ls_rows = _local_search_optimize(rows, k, ls_deadline, rng)
        
        if _valid(ls_rows, k):
            actual_scope = _scope(ls_rows)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = ls_rows
    
    return best_rows
