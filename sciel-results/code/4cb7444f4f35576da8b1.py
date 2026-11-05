import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    
    # Phase 1: Greedy construction with packing heuristic
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple seeds to find a good initial packing
    max_seeds = 100
    for seed in range(max_seeds):
        if time.time() > deadline - 30:
            break
            
        rows = _greedy_construct_heuristic(n_rows, k, seed)
        if rows is None:
            continue
            
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
        # Early termination if we hit a theoretical lower bound or good value
        # Lower bound for scope M is roughly sqrt(2 * n_rows * num_diffs_per_row)
        # But we are minimizing the max entry.
        
    if best_rows is None:
        # Fallback to simple greedy if heuristic failed
        best_rows = _greedy_construct_simple(n_rows, k, 0)
        best_scope = _scope(best_rows)

    # Phase 2: Local Search via Row Replacement
    if time.time() < deadline - 5:
        improved = _local_search_row_replace(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final Polish - try to shrink the max element of each row
    if time.time() < deadline - 1:
        polished = _polish_max_element(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    """Return frozenset of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _greedy_construct_simple(n_rows, k, seed):
    """Simple greedy fallback."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            x = last_x + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
    return rows

def _greedy_construct_heuristic(n_rows, k, seed):
    """Greedy construction using a packing score to minimize fragmentation."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []  # Keep sorted for binary search
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            best_x = None
            best_score = float('inf')
            candidates = []
            
            # Search for next x
            # Limit search to reasonable range, e.g., current max + 20
            limit = last_x + 50
            if not row:
                limit = 50
            
            x = last_x + 1
            while x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # Calculate score
                    score = _packing_score(new_diffs, used_sorted)
                    candidates.append((score, x))
                    if score < best_score:
                        best_score = score
                        best_x = x
                x += 1
            
            if not candidates:
                # If no candidates found in range, expand or fail
                # For robustness, just pick the first valid x beyond limit
                x = limit + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        best_x = x
                        break
                    x += 1
                candidates = [(0, best_x)]
            
            # Sort candidates by score
            candidates.sort(key=lambda t: t[0])
            
            # Pick from top candidates with some randomness
            # If scores are close, randomize to explore
            top_n = min(5, len(candidates))
            # Bias towards better scores
            # Weights: 1/score (or similar)
            # Simple: pick uniformly from top N if N>1, else pick best
            if top_n > 1 and rng.random() < 0.5:
                chosen_idx = rng.randint(0, top_n - 1)
            else:
                chosen_idx = 0
            
            chosen_x = candidates[chosen_idx][1]
            row.append(chosen_x)
            
            # Update used diffs
            new_diffs = {chosen_x - a for a in row[:-1]}
            for d in new_diffs:
                used.add(d)
                # Insert into sorted list
                pos = bisect.bisect_left(used_sorted, d)
                used_sorted.insert(pos, d)
        
        rows.append(row)
    
    return rows

def _packing_score(new_diffs, used_sorted):
    """
    Score a set of new differences based on how well they pack into the used space.
    Lower score is better.
    We penalize differences that are far from existing used differences,
    as this creates large gaps.
    """
    if not used_sorted:
        # No used diffs yet, prefer smaller x? 
        # Actually, if no used diffs, any is fine. 
        # But we want to start small.
        return sum(new_diffs) * 0.1
    
    total_dist = 0.0
    for d in new_diffs:
        # Find nearest used diff
        pos = bisect.bisect_left(used_sorted, d)
        
        min_dist = float('inf')
        if pos < len(used_sorted):
            min_dist = min(min_dist, abs(used_sorted[pos] - d))
        if pos > 0:
            min_dist = min(min_dist, abs(used_sorted[pos - 1] - d))
        
        if min_dist == float('inf'):
            min_dist = 100.0 # Large penalty
        
        total_dist += min_dist
    
    # Average distance
    avg_dist = total_dist / len(new_diffs)
    
    # Also penalize the max value of new diffs slightly to keep rows compact
    max_new = max(new_diffs)
    
    return avg_dist + 0.05 * max_new

def _local_search_row_replace(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search that tries to replace entire rows with better-fitting ones.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    max_iterations = 5000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline - 5:
            break
        iteration += 1
        
        # Pick a random row to replace
        idx = rng.randint(0, n_rows - 1)
        old_row = rows[idx]
        old_max = old_row[-1]
        
        # Collect diffs from all other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == idx:
                continue
            used_diffs.update(_row_diffs(rows[j]))
        
        # Try to find a new row with max < old_max
        # Use randomized backtracking with a limit
        limit = old_max - 1
        if limit < k:
            limit = k + 10 # Allow some expansion if needed, but prefer shrink
        
        new_row = _find_row_with_limit(used_diffs, k, limit, rng)
        
        if new_row is not None:
            # Check if it's actually better (smaller max)
            if new_row[-1] < old_max:
                rows[idx] = new_row
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            else:
                # Even if max is not smaller, replacing the row might free up 
                # "space" in the difference spectrum for other rows?
                # No, the constraint is global uniqueness of diffs.
                # Replacing a row with one of the same max doesn't help directly 
                # unless it allows *other* rows to shrink later.
                # So we accept it with some probability (Metropolis) if it's valid.
                # But we must ensure it doesn't increase the global scope.
                # Since we only replace if new_row[-1] <= old_max, global scope 
                # won't increase.
                # Let's accept it to explore the search space.
                if new_row[-1] <= old_max:
                    rows[idx] = new_row
        
        # Also try to shrink the current row's max element specifically
        # This is a more targeted move
        if time.time() > deadline - 5:
            break
        
        # Try to reduce the max element of rows[idx]
        current_max = rows[idx][-1]
        # Try values from current_max - 1 down to k
        # We need to reconstruct the row with a smaller last element
        # This is expensive, so we only do it occasionally
        if rng.random() < 0.1:
            for target_max in range(current_max - 1, k - 1, -1):
                if time.time() > deadline - 5:
                    break
                used_diffs_other = set()
                for j in range(n_rows):
                    if j == idx:
                        continue
                    used_diffs_other.update(_row_diffs(rows[j]))
                
                candidate_row = _find_row_with_limit(used_diffs_other, k, target_max, rng)
                if candidate_row is not None:
                    rows[idx] = candidate_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    break # Stop reducing this row for now
    
    return best_rows

def _find_row_with_limit(used_diffs, k, max_val, rng):
    """
    Find a row of k+1 elements starting at 0, max <= max_val,
    with all internal diffs avoiding used_diffs.
    Uses randomized backtracking.
    """
    row = [0]
    
    def backtrack(pos):
        if pos == k + 1:
            return True
        
        last_x = row[-1]
        # Determine range for next element
        # Must be > last_x and <= max_val
        # Also, the remaining elements must fit.
        # Minimum possible value for next element is last_x + 1
        # Maximum possible value is max_val
        
        # Heuristic: try values close to last_x first? Or random?
        # Randomized search to escape local minima
        candidates = []
        for x in range(last_x + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                candidates.append(x)
        
        if not candidates:
            return False
        
        # Shuffle candidates for randomization
        rng.shuffle(candidates)
        
        # Bias towards smaller values to keep row compact
        # We can sort by value, but then shuffle with bias
        # Simpler: just pick randomly from candidates
        # Or pick the smallest few and randomize among them
        top_n = min(10, len(candidates))
        # Sort candidates
        candidates.sort()
        chosen_subset = candidates[:top_n]
        rng.shuffle(chosen_subset)
        
        for x in chosen_subset:
            row.append(x)
            # Update used_diffs temporarily? 
            # No, we check against the *global* used_diffs from other rows.
            # We also need to ensure *internal* diffs are distinct.
            # The check `not (new_diffs & used_diffs)` ensures no conflict with other rows.
            # But we also need internal distinctness.
            # Wait, `new_diffs` are the diffs of the new element x with previous elements in the row.
            # We must ensure these are not already in the row's internal diffs.
            # Let's maintain a set of internal diffs for the current row.
            
            # This function signature is a bit tricky. 
            # Let's refactor to maintain internal state.
            
            pass # Placeholder for logic below
            
    # Refactored backtracking with internal state
    internal_diffs = set()
    
    def backtrack_v2(pos):
        if pos == k + 1:
            return True
        
        last_x = row[-1]
        
        # Generate candidates
        valid_xs = []
        for x in range(last_x + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            # Check conflict with global used_diffs
            if new_diffs & used_diffs:
                continue
            # Check conflict with internal_diffs
            if new_diffs & internal_diffs:
                continue
            valid_xs.append((x, new_diffs))
        
        if not valid_xs:
            return False
        
        # Randomize order
        rng.shuffle(valid_xs)
        
        # Bias towards smaller x
        valid_xs.sort(key=lambda t: t[0])
        top_n = min(5, len(valid_xs))
        candidates = valid_xs[:top_n]
        rng.shuffle(candidates)
        
        for x, new_diffs in candidates:
            row.append(x)
            internal_diffs.update(new_diffs)
            
            if backtrack_v2(pos + 1):
                return True
            
            # Backtrack
            row.pop()
            internal_diffs.difference_update(new_diffs)
            
        return False
    
    if backtrack_v2(1):
        return row
    return None

def _polish_max_element(initial_rows, n_rows, k, start_time, deadline):
    """
    Final pass to try to reduce the max element of each row.
    """
    rng = random.Random(777)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for idx in range(n_rows):
        if time.time() > deadline - 1:
            break
            
        old_row = rows[idx]
        old_max = old_row[-1]
        
        # Collect diffs from other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == idx:
                continue
            used_diffs.update(_row_diffs(rows[j]))
        
        # Try to find a row with max < old_max
        # We already did this in local search, but let's try harder here
        # Try limits from old_max - 1 down to k
        for target_max in range(old_max - 1, k - 1, -1):
            if time.time() > deadline - 1:
                break
            
            # Try a few random restarts for this target
            found = False
            for _ in range(10):
                new_row = _find_row_with_limit(used_diffs, k, target_max, rng)
                if new_row is not None:
                    rows[idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    found = True
                    break
            
            if found:
                break
    
    return best_rows
