import time
import random
import math
import bisect
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with best-fit greedy
    seed = 0
    while time.time() < deadline - 60:  # Reserve 60s for SA + polish
        rows = _greedy_construct(n_rows, k, seed)
        if rows is None:
            seed += 1
            continue
        scope = _scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            if best_scope == lower_bound:
                break
        
        seed += 1
    
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated annealing to improve
    if time.time() < deadline - 20:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Final polish pass targeting the maximum element
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
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
                return None
            
            # Best-fit: sort by packing score (lower = better fit), pick from top few
            scored_xs.sort(key=lambda t: t[0])
            top_n = min(5, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(used_sorted, d)
        
        rows.append(row)
    
    return rows

def _packing_score(new_diffs, used_sorted, x):
    """
    Score a candidate x by how well its differences 'pack' into the used difference space.
    Lower score = better fit (less fragmentation).
    Uses binary search on sorted used differences for efficiency.
    """
    if not used_sorted:
        return x * 1.0
    
    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, used_sorted[idx] - d)
        if idx > 0:
            min_dist = min(min_dist, d - used_sorted[idx - 1])
        total_dist += min_dist
    
    return total_dist / max(len(new_diffs), 1) + 0.05 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """Simulated annealing with O(1) diff-count validity checking."""
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute row diff sets and global diff counter
    row_diffs = []
    diff_counter = Counter()
    for row in rows:
        n = len(row)
        dset = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                dset.add(d)
                diff_counter[d] += 1
        row_diffs.append(dset)
    
    # Temperature schedule
    T_start = 30.0
    T_end = 0.5
    total_steps = 50000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    
    T = T_start
    steps_since_improve = 0
    max_no_improve = 8000
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Occasionally do a targeted move on the row with the maximum element
        if rng.random() < 0.35:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        # Generate a new candidate row using best-fit greedy
        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        
        if new_row is None:
            continue
        
        # Compute new diff set for the new row
        n = len(new_row)
        new_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                new_diffs.add(new_row[j] - new_row[i])
        
        # O(1)-amortized validity check using diff_counter
        # A diff is "used by others" if diff_counter[d] > (1 if d in row_diffs[row_idx] else 0)
        valid = True
        for d in new_diffs:
            used_by_others = diff_counter[d] - (1 if d in row_diffs[row_idx] else 0)
            if used_by_others > 0:
                valid = False
                break
        
        if not valid:
            continue
        
        # Compute new scope
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        # Accept or reject based on Metropolis criterion
        delta = new_scope - current_scope
        if delta <= 0:
            # Update diff_counter: remove old row's diffs, add new row's diffs
            old_dset = row_diffs[row_idx]
            for d in old_dset:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in new_diffs:
                diff_counter[d] += 1
            
            rows = new_rows
            row_diffs[row_idx] = new_diffs
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                old_dset = row_diffs[row_idx]
                for d in old_dset:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for d in new_diffs:
                    diff_counter[d] += 1
                
                rows = new_rows
                row_diffs[row_idx] = new_diffs
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        T *= cool_rate
        
        # If stuck for too long, do a small perturbation
        if steps_since_improve > max_no_improve:
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
            if pert_row is not None:
                n = len(pert_row)
                pert_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        pert_diffs.add(pert_row[j] - pert_row[i])
                
                # Check validity with diff_counter
                valid = True
                for d in pert_diffs:
                    used_by_others = diff_counter[d] - (1 if d in row_diffs[pert_idx] else 0)
                    if used_by_others > 0:
                        valid = False
                        break
                
                if valid:
                    old_dset = row_diffs[pert_idx]
                    for d in old_dset:
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    for d in pert_diffs:
                        diff_counter[d] += 1
                    
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
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 400
        
        x = last_x + 1
        while len(valid_xs) < 25 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 25:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)
    
    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: repeatedly try to reduce the maximum element in the row that has it.
    Uses targeted local search with multiple strategies.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    max_iter = 300
    
    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Strategy 1: Greedy smallest-first reduction
        improved_row = _reduce_max_row(rows, max_idx, k, rng, current_max)
        
        if improved_row is not None and improved_row[-1] < current_max:
            new_rows = rows[:]
            new_rows[max_idx] = improved_row
            if _is_valid(new_rows, n_rows, k):
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    rows = new_rows
                continue
        
        # Strategy 2: Try reducing by a larger margin
        if time.time() < deadline - 4:
            improved_row2 = _reduce_max_row_aggressive(rows, max_idx, k, rng, current_max)
            if improved_row2 is not None and improved_row2[-1] < current_max:
                new_rows = rows[:]
                new_rows[max_idx] = improved_row2
                if _is_valid(new_rows, n_rows, k):
                    new_scope = _scope(new_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in new_rows]
                        rows = new_rows
                    continue
        
        # No improvement this iteration
        break
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _reduce_max_row(rows, row_idx, k, rng, current_max):
    """
    Try to find a replacement for rows[row_idx] with a smaller maximum element.
    Uses a constrained greedy that tries to keep all elements below current_max.
    """
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    target_max = current_max - 1
    for attempt in range(20):
        row = [0]
        success = True
        temp_used_diffs = set(used_diffs)
        temp_used_sorted = list(used_sorted)
        
        for _ in range(k):
            valid_xs = []
            last_x = row[-1]
            
            x = last_x + 1
            while x <= target_max:
                new_diffs = {x - a for a in row}
                if not (new_diffs & temp_used_diffs):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                success = False
                break
            
            # Pick smallest to minimize max
            chosen_x = valid_xs[0]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            temp_used_diffs.update(new_diffs)
            for d in new_diffs:
                bisect.insort(temp_used_sorted, d)
        
        if success and len(row) == k + 1:
            return row
        
        target_max = min(target_max + 1, current_max - 1)
    
    return None

def _reduce_max_row_aggressive(rows, row_idx, k, rng, current_max):
    """
    More aggressive reduction: try to find a row with max significantly below current_max.
    Uses randomized restarts with tighter bounds.
    """
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    # Try progressively tighter targets
    for reduction in range(1, 15):
        target_max = current_max - reduction
        if target_max < k:  # Can't fit k+1 elements
            break
        
        for attempt in range(5):
            row = [0]
            success = True
            temp_used_diffs = set(used_diffs)
            
            for _ in range(k):
                valid_xs = []
                last_x = row[-1]
                
                x = last_x + 1
                while x <= target_max:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & temp_used_diffs):
                        valid_xs.append(x)
                    x += 1
                
                if not valid_xs:
                    success = False
                    break
                
                # Randomized choice among top candidates for diversity
                if len(valid_xs) > 1 and rng.random() < 0.3:
                    chosen_x = valid_xs[rng.randint(0, min(3, len(valid_xs)-1))]
                else:
                    chosen_x = valid_xs[0]
                
                row.append(chosen_x)
                new_diffs = {chosen_x - a for a in row[:-1]}
                temp_used_diffs.update(new_diffs)
            
            if success and len(row) == k + 1:
                return row
    
    return None
