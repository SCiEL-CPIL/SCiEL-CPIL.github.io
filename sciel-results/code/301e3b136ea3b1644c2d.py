import time
import random
import math
import bisect
import heapq
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with interval-fragmentation greedy
    # Use multiple seeds to find diverse starting points
    seed = 0
    max_initial_seeds = 80
    while time.time() < deadline - 70 and seed < max_initial_seeds:
        rows = _greedy_construct(n_rows, k, seed)
        if rows is None:
            seed += 1
            continue
        scope = _scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
        seed += 1
    
    if best_rows is None:
        # Fallback if greedy failed completely
        best_rows = _greedy_construct(n_rows, k, 0)
        if best_rows is None:
            # Last resort: try with a very high limit
            best_rows = _greedy_construct(n_rows, k, 999, limit_override=5000)
            if best_rows is None:
                return [[0, 1, 3, 7, 15, 31]] * n_rows  # Will be invalid, but avoids crash
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

def _greedy_construct(n_rows, k, seed, limit_override=None):
    """Best-fit greedy using interval fragmentation minimization."""
    rng = random.Random(seed)
    used = set()
    rows = []
    base_limit = limit_override if limit_override else 500
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            limit = base_limit
            
            x = last_x + 1
            best_score = float('inf')
            best_x = None
            candidates = []
            
            while x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    score = _interval_score(new_diffs, used, x)
                    candidates.append((score, x))
                    if score < best_score:
                        best_score = score
                        best_x = x
                x += 1
            
            if best_x is None:
                # Expand search if needed
                x = limit + 1
                while x <= limit * 3:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        score = _interval_score(new_diffs, used, x)
                        candidates.append((score, x))
                        if score < best_score:
                            best_score = score
                            best_x = x
                    x += 1
            
            if best_x is None:
                return None
            
            # Pick from top candidates with some randomness for diversity
            candidates.sort(key=lambda t: t[0])
            top_n = min(5, len(candidates))
            chosen_x = candidates[rng.randint(0, top_n - 1)][1]
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _interval_score(new_diffs, used, x):
    """
    Score based on interval fragmentation: count how many new isolated gaps are created
    by adding these differences to the used set. Fewer new intervals = better packing.
    """
    if not used:
        return float(len(new_diffs))
    
    new_count = 0
    for d in new_diffs:
        has_neighbor = False
        for delta in [1, 2, 3]:
            if (d - delta) in used or (d + delta) in used:
                has_neighbor = True
                break
        
        if not has_neighbor:
            new_count += 1
    
    # Penalize large x to prefer compact solutions
    return float(new_count) + 0.01 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """Simulated annealing with best-fit row reconstruction."""
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    total_steps = 25000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    
    T = T_start
    steps_since_improve = 0
    max_no_improve = 3000
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Occasionally do a targeted move on the row with the maximum element
        if rng.random() < 0.4:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        # Generate a new candidate row using best-fit greedy
        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        
        if new_row is None:
            continue
        
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        # Check validity: all differences must be distinct
        all_diffs = _all_diffs(new_rows)
        expected_diffs = n_rows * (k + 1) * k // 2
        if len(all_diffs) != expected_diffs:
            continue
        
        # Accept or reject based on Metropolis criterion
        delta = new_scope - current_scope
        if delta <= 0:
            rows = new_rows
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Avoid overflow in exp
            if T > 1e-9:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                rows = new_rows
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        T *= cool_rate
        
        # If stuck for too long, do a small perturbation (restart-like)
        if steps_since_improve > max_no_improve:
            # Perturb a random row
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
            if pert_row is not None:
                test_rows = rows[:]
                test_rows[pert_idx] = pert_row
                if _is_valid(test_rows, n_rows, k):
                    rows = test_rows
                    current_scope = _scope(rows)
            steps_since_improve = 0
            # Increase temperature slightly to escape local minima
            T = max(T, 1.0)
    
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
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 800
        
        x = last_x + 1
        while len(valid_xs) < 40 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _interval_score(new_diffs, used_diffs, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 40:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _interval_score(new_diffs, used_diffs, x)
                    scored_xs.append((score, x))
                x += 1
                if x > limit * 10:
                    break
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(6, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
    
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
    
    for iteration in range(150):
        if time.time() > deadline - 3:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
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
    
    # Try to find a row with max < current_max
    target_max = current_max - 1
    for attempt in range(30):
        row = [0]
        success = True
        for _ in range(k):
            valid_xs = []
            last_x = row[-1]
            
            x = last_x + 1
            while x <= target_max:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                success = False
                break
            
            # Pick smallest to minimize max, with some randomness
            if attempt % 3 == 0:
                chosen_x = valid_xs[0]
            else:
                chosen_x = valid_xs[rng.randint(0, min(5, len(valid_xs)-1))]
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used_diffs.update(new_diffs)
        
        if success and len(row) == k + 1:
            return row
    
    return None
