import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Phase 1: Get a baseline solution using the existing greedy method
    # This ensures we have a valid solution to return if bounded search fails
    baseline_rows = _greedy_construct(n_rows, k, 0)
    if baseline_rows is None:
        # Fallback: simple random construction
        baseline_rows = _random_construct(n_rows, k)
        
    baseline_scope = _scope(baseline_rows)
    
    # Phase 2: Iterative Deepening on Scope
    # Try to find a solution with scope M, starting from lower_bound up to baseline_scope
    for M in range(lower_bound, baseline_scope + 1):
        if time.time() > deadline - 5:
            break
            
        # Allocate a small time slice for this M. 
        # We want to be able to check at least a few M values.
        # Let's say we spend max 5s per M to allow checking more values.
        time_slice = 5.0
        if time.time() + time_slice > deadline - 5:
            time_slice = deadline - 5 - time.time()
            if time_slice < 0.5:
                break
        
        solution = _try_solve_bounded(n_rows, k, M, time.time() + time_slice)
        if solution is not None:
            # Verify scope
            if _scope(solution) <= M:
                return solution
    
    return baseline_rows

def _random_construct(n_rows, k):
    """Fallback random construction."""
    rng = random.Random(1)
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            while True:
                x = rng.randint(row[-1] + 1, 200)
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
                    break
        rows.append(row)
    return rows

def _try_solve_bounded(n_rows, k, M, deadline):
    """
    Try to find a valid set of rows with max element <= M.
    Uses randomized restarts with backtracking.
    """
    rng = random.Random(int(time.time() * 1000) % 10000)
    
    while time.time() < deadline:
        rows = _backtrack_solve_single(n_rows, k, M, rng)
        if rows is not None:
            return rows
        
    return None

def _backtrack_solve_single(n_rows, k, M, rng):
    """
    Attempt to construct n_rows rows with max element <= M using backtracking.
    Returns list of rows or None if failed.
    """
    rows = []
    used_diffs = set()
    
    def solve_row(idx):
        if idx == n_rows:
            return True
        
        # Try to build row idx
        # We use a randomized greedy/backtrack for this single row
        row = [0]
        
        # We need to choose k elements from 1..M
        # Let's try a few random strategies for this row
        for attempt in range(100): # Limit attempts per row to avoid infinite loop
            candidate_row = _greedy_row_constrained(row, k, M, used_diffs, rng)
            if candidate_row is not None:
                rows.append(candidate_row)
                # Update used_diffs
                new_diffs = set()
                n = len(candidate_row)
                for i in range(n):
                    for j in range(i+1, n):
                        new_diffs.add(candidate_row[j] - candidate_row[i])
                used_diffs.update(new_diffs)
                
                if solve_row(idx + 1):
                    return True
                
                # Backtrack
                used_diffs.difference_update(new_diffs)
                rows.pop()
            else:
                # If greedy failed, try a more random approach?
                # For now, break to trigger restart
                break
        
        return False
    
    return rows if solve_row(0) else None

def _greedy_row_constrained(current_row, k, M, used_diffs, rng):
    """
    Try to extend current_row to have k more elements, all <= M,
    such that no new difference is in used_diffs.
    """
    row = list(current_row)
    local_used = set(used_diffs)
    
    for _ in range(k):
        valid_xs = []
        last_x = row[-1]
        
        # Collect valid candidates
        # To speed up, we don't check all 1..M, we check a random subset or a range
        # But M is small (<=120), so we can check all.
        for x in range(last_x + 1, M + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & local_used):
                valid_xs.append(x)
        
        if not valid_xs:
            return None
        
        # Pick a candidate
        # Heuristic: pick one that minimizes future conflicts?
        # For now, random pick from valid_xs to diversify
        chosen_x = rng.choice(valid_xs)
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        local_used.update(new_diffs)
        
    return row

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, seed):
    """Best-fit greedy: prefer values whose differences pack tightly into used space."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]
            limit = 500
            
            x = last_x + 1
            while len(valid_xs) < 40 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    # Score: prefer x whose differences are close to already-used values
                    # (minimizes fragmentation of the difference space)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 40:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = _packing_score(new_diffs, used_sorted, x)
                        scored_xs.append((score, x))
                    x += 1
            
            if not valid_xs:
                return None  # should not happen for reasonable limits
            
            # Best-fit: sort by packing score (lower = better fit), pick from top few
            scored_xs.sort(key=lambda t: t[0])
            # Pick from top 5 with some randomness to avoid getting stuck
            top_n = min(5, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            # Keep sorted list updated
            for d in new_diffs:
                bisect.insort(used_sorted, d)
        
        rows.append(row)
    
    return rows

def _packing_score(new_diffs, used, x):
    """
    Score a candidate x by how well its differences 'pack' into the used difference space.
    Lower score = better fit (less fragmentation).
    
    Strategy: for each new difference d, measure its distance to the nearest used difference.
    Sum these distances, weighted to prefer smaller overall x.
    Uses sorted list for efficient nearest-neighbor lookup.
    """
    if not used:
        return x * 1.0  # no used diffs, just prefer small x
    
    # Convert to sorted list once for efficient binary search
    if not isinstance(used, list):
        used_sorted = sorted(used)
    else:
        used_sorted = used
    
    total_dist = 0.0
    for d in new_diffs:
        # Binary search for nearest used difference
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist
    
    # Combine: prefer small x AND tight packing
    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """Simulated annealing with best-fit row reconstruction and targeted max reduction."""
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute difference sets for each row for faster validity checks
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    # Build global used set (maintained incrementally)
    global_used = set()
    for diffs in row_diffs:
        global_used.update(diffs)
    
    # Temperature schedule
    T_start = 30.0
    T_end = 0.5
    total_steps = 30000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    
    T = T_start
    steps_since_improve = 0
    max_no_improve = 5000
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Occasionally do a targeted move on the row with the maximum element
        if rng.random() < 0.3:
            # Find row with max last element
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        # Save old row and its diffs
        old_row = list(rows[row_idx])
        old_diffs = row_diffs[row_idx]
        
        # Generate a new candidate row using best-fit greedy
        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        
        if new_row is None:
            continue
        
        # Compute new diffs for the new row
        n = len(new_row)
        new_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                new_diffs.add(new_row[j] - new_row[i])
        
        # Check validity: new_diffs must not overlap with global_used minus old_diffs
        other_diffs = global_used - old_diffs
        if new_diffs & other_diffs:
            continue  # Invalid
        
        # Compute new scope
        new_scope = _scope(rows) if rows[row_idx][-1] >= new_row[-1] else max(new_row[-1], max(r[-1] for i, r in enumerate(rows) if i != row_idx))
        
        # Smooth epigraph objective: penalize large elements beyond the minimum needed
        # This encourages reducing the max element even when scope doesn't change
        old_max = rows[row_idx][-1]
        new_max = new_row[-1]
        # Soft penalty: if new row is smaller, bonus; if larger, penalty
        max_penalty = max(0, new_max - old_max) * 0.5
        
        # Accept or reject based on Metropolis criterion with smooth objective
        delta = new_scope - current_scope + max_penalty
        if delta <= 0:
            # Improvement, always accept
            # Update global_used: remove old_diffs, add new_diffs
            global_used.difference_update(old_diffs)
            global_used.update(new_diffs)
            rows[row_idx] = new_row
            row_diffs[row_idx] = new_diffs
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Worse, accept with probability exp(-delta/T)
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                global_used.difference_update(old_diffs)
                global_used.update(new_diffs)
                rows[row_idx] = new_row
                row_diffs[row_idx] = new_diffs
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        T *= cool_rate
        
        # If stuck for too long, do a small perturbation
        if steps_since_improve > max_no_improve:
            # Perturb a random row
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
            if pert_row is not None:
                # Check validity
                n = len(pert_row)
                pert_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        pert_diffs.add(pert_row[j] - pert_row[i])
                
                other_diffs = global_used - row_diffs[pert_idx]
                if not (pert_diffs & other_diffs):
                    global_used.difference_update(row_diffs[pert_idx])
                    global_used.update(pert_diffs)
                    rows[pert_idx] = pert_row
                    row_diffs[pert_idx] = pert_diffs
                    current_scope = _scope(rows)
            steps_since_improve = 0
    
    return best_rows

def _is_valid(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == n_rows * (k + 1) * k // 2

def _generate_candidate_row(rows, exclude_idx, k, rng):
    """Generate a new candidate row using best-fit packing heuristic."""
    # Collect all differences used by other rows
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    # Build new row using best-fit greedy
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 500
        
        x = last_x + 1
        while len(valid_xs) < 30 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            # Try harder
            x = last_x + 1
            while len(valid_xs) < 30:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
        
        if not valid_xs:
            return None
        
        # Best-fit: sort by packing score, pick from top few
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        # Update sorted list
        for d in new_diffs:
            bisect.insort(used_sorted, d)
    
    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: repeatedly try to reduce the maximum element in the row that has it.
    Uses targeted local search on the max element.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    max_iter = 500
    
    # Precompute row diffs for faster validity checks
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    # Build global used set for O(1) validity checks
    global_used = set()
    for diffs in row_diffs:
        global_used.update(diffs)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        # Find row with maximum last element
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Try to construct a new version of this row with a smaller maximum
        # by doing a local search that specifically targets reducing the last element
        improved_row = _reduce_max_row(rows, max_idx, k, rng, current_max)
        
        if improved_row is not None and improved_row[-1] < current_max:
            # Check validity using precomputed diffs
            n = len(improved_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(improved_row[j] - improved_row[i])
            
            other_diffs = global_used - row_diffs[max_idx]
            
            if not (new_diffs & other_diffs):
                new_rows = rows[:]
                new_rows[max_idx] = improved_row
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    # Update global_used
                    global_used.difference_update(row_diffs[max_idx])
                    global_used.update(new_diffs)
                    rows = new_rows
                    row_diffs[max_idx] = new_diffs
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _reduce_max_row(rows, row_idx, k, rng, current_max):
    """
    Try to find a replacement for rows[row_idx] with a smaller maximum element.
    Uses a constrained greedy that tries to keep all elements below current_max.
    """
    # Collect differences from other rows
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    # Try to build a row with max element < current_max
    target_max = current_max - 1
    for attempt in range(30):
        row = [0]
        success = True
        local_used = set(used_diffs)
        local_sorted = list(used_sorted)
        
        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]
            
            x = last_x + 1
            while x <= target_max:
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, local_sorted, x)
                    scored_xs.append((score, x))
                x += 1
            
            if not valid_xs:
                success = False
                break
            
            # Use packing score to choose, preferring tighter fits
            scored_xs.sort(key=lambda t: t[0])
            # Pick from top 3 with randomness
            top_n = min(3, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            local_used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(local_sorted, d)
        
        if success and len(row) == k + 1:
            return row
        
        # Relax target slightly for next attempt
        target_max = min(target_max + 1, current_max - 1)
    
    return None
