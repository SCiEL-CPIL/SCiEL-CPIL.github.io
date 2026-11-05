import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    # Phase 1: Iterative deepening on maximum element M
    # Try to find a solution with max element <= M for increasing M
    best_scope = float('inf')
    best_rows = None

    # Start with a reasonable initial M estimate
    # Lower bound on M: each row needs differences 1..M covering num_diffs_per_row distinct values
    # But differences can go up to M, so we need M >= num_diffs_per_row per row
    # Actually, the max difference in a row is at most M (since elements are 0..M)
    # And we need num_diffs_per_row distinct differences per row
    # So M must be at least such that we can pick num_diffs_per_row distinct values from 1..M
    # That's trivially M >= num_diffs_per_row
    # But also the row structure constrains which differences are possible
    
    # Heuristic initial M: start from lower bound and increase
    m_initial = max(lower_bound, n_rows * num_diffs_per_row)
    
    # First, get a feasible solution with a larger M to have something to work with
    m_start = m_initial + 50
    rows = _construct_with_max_constraint(n_rows, k, m_start, start_time, deadline - 60)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)
    
    if best_rows is None:
        # Fallback: use greedy to get any valid solution
        best_rows = _greedy_construct(n_rows, k, 0)
        best_scope = _scope(best_rows)
    
    # Phase 2: Iterative deepening - try to reduce M
    # Binary search or linear search for minimum M
    m_low = max(lower_bound, best_scope - 20)
    m_high = best_scope
    
    # Try to find the minimum M by testing feasibility
    while time.time() < deadline - 40:
        if m_low >= m_high:
            break
        
        m_mid = (m_low + m_high) // 2
        
        # Try to construct with max element <= m_mid
        rows = _construct_with_max_constraint(n_rows, k, m_mid, start_time, deadline - 40)
        if rows is not None:
            scope = _scope(rows)
            if scope <= m_mid:
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                m_high = m_mid
            else:
                m_low = m_mid + 1
        else:
            m_low = m_mid + 1
    
    # Phase 3: Local search to further improve
    if time.time() < deadline - 10 and best_rows is not None:
        improved = _local_search_improve(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

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

def _construct_with_max_constraint(n_rows, k, max_elem, start_time, deadline):
    """
    Try to construct n_rows rows with all elements <= max_elem.
    Uses backtracking with constraint propagation.
    Returns list of rows if successful, None otherwise.
    """
    rng = random.Random(int((deadline - start_time) * 1000) % 100000)
    used_diffs = set()
    rows = []
    
    # Precompute: for a given max_elem, what differences are possible?
    # Differences range from 1 to max_elem
    # We need to assign n_rows * num_diffs_per_row distinct differences
    
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # Quick feasibility check: we need total_diffs_needed distinct differences in 1..max_elem
    if max_elem < total_diffs_needed:
        return None
    
    # Build rows one by one with backtracking
    for row_idx in range(n_rows):
        if time.time() > deadline - 5:
            return None
        
        # Try to build a row with elements <= max_elem
        row = _build_row_constrained(used_diffs, k, max_elem, rng, row_idx)
        if row is None:
            # Try a few more times with different random seeds
            success = False
            for attempt in range(3):
                rng2 = random.Random(rng.randint(0, 10**9))
                row = _build_row_constrained(used_diffs, k, max_elem, rng2, row_idx + attempt)
                if row is not None:
                    success = True
                    break
            if not success:
                return None
        
        rows.append(row)
        # Update used differences
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                used_diffs.add(row[j] - row[i])
    
    # Verify all rows are valid
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None  # Duplicate difference
                all_diffs.add(d)
    
    return rows

def _build_row_constrained(used_diffs, k, max_elem, rng, row_idx):
    """
    Build a single row of k+1 increasing integers starting at 0, all <= max_elem,
    such that all differences are not in used_diffs and are distinct within the row.
    """
    row = [0]
    
    # Available differences: 1 to max_elem, minus used_diffs
    # We'll use backtracking to pick elements
    
    def backtrack(current_row, depth):
        if depth == k:
            return list(current_row)
        
        last_val = current_row[-1]
        
        # Calculate differences already in current_row
        current_internal_diffs = set()
        n = len(current_row)
        for i in range(n):
            for j in range(i+1, n):
                current_internal_diffs.add(current_row[j] - current_row[i])
        
        # Try candidates from last_val+1 to max_elem
        # Use some randomization to escape local minima
        candidates = []
        for x in range(last_val + 1, max_elem + 1):
            new_diffs = {x - a for a in current_row}
            # Check against global used diffs
            if new_diffs & used_diffs:
                continue
            # Check against internal diffs
            if new_diffs & current_internal_diffs:
                continue
            candidates.append(x)
        
        if not candidates:
            return None
        
        # Randomize order: pick from candidates with some preference for smaller values
        # but with randomness to explore different solutions
        if len(candidates) > 1:
            # Bias toward smaller values but allow exploration
            weights = [1.0 / (1 + i) for i in range(len(candidates))]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = len(candidates) - 1
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            # Swap chosen to front
            candidates[chosen_idx], candidates[0] = candidates[0], candidates[chosen_idx]
        
        # Try candidates in order
        for x in candidates:
            current_row.append(x)
            result = backtrack(current_row, depth + 1)
            if result is not None:
                return result
            current_row.pop()
        
        return None
    
    return backtrack(row, 0)

def _backtrack_row(used_diffs, k, rng, time_limit=5.0):
    """
    Backtracking search for a single row with forward checking.
    Tries to find a row of k+1 increasing integers starting at 0
    whose differences don't conflict with used_diffs.
    """
    start_time = time.time()
    row = [0]
    # We need k more elements
    # Use iterative deepening or just DFS with pruning

    def get_internal_diffs(r):
        diffs = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.add(r[j] - r[i])
        return diffs

    def backtrack(current_row, depth):
        nonlocal start_time, used_diffs, k, rng

        if time.time() - start_time > time_limit:
            return None

        if depth == k:
            return list(current_row)

        last_val = current_row[-1]
        
        # Calculate differences already present in current_row
        # to ensure internal distinctness
        current_internal_diffs = get_internal_diffs(current_row)
        
        # Try candidates in ascending order, but with some randomization
        # to escape local minima
        candidates = []
        x = last_val + 1
        # Limit search space for efficiency
        max_x = last_val + 200
        while x <= max_x:
            new_diffs = {x - a for a in current_row}
            # Check against global used diffs
            if new_diffs & used_diffs:
                x += 1
                continue
            # Check against internal diffs of current_row
            if new_diffs & current_internal_diffs:
                x += 1
                continue
            candidates.append(x)
            x += 1

        if not candidates:
            return None

        # Randomize order slightly to get different solutions
        if rng.random() < 0.3:
            # Shuffle a subset
            if len(candidates) > 5:
                idx = rng.randint(0, len(candidates) - 1)
                candidates[idx], candidates[0] = candidates[0], candidates[idx]

        for x in candidates:
            current_row.append(x)
            # No need to re-check new_diffs against used_diffs here as done above,
            # but we must recurse. The check was done before appending.
            
            result = backtrack(current_row, depth + 1)
            if result is not None:
                return result

            current_row.pop()

        return None

    return backtrack(row, 0)

def _greedy_row(used_diffs, k, rng):
    """Best-fit greedy for a single row."""
    row = [0]
    used_sorted = sorted(used_diffs)

    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 500

        x = last_x + 1
        while len(valid_xs) < 40 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1

        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 40:
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
        top_n = min(5, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)

    return row

def _greedy_construct(n_rows, k, seed):
    """Pure best-fit greedy for all rows."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []

    for _ in range(n_rows):
        row = _greedy_row(used, k, rng)
        if row is None:
            return None
        rows.append(row)
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                used.add(d)
                bisect.insort(used_sorted, d)

    return rows

def _packing_score(new_diffs, used, x):
    """
    Score a candidate x by how well its differences 'pack' into the used difference space.
    Lower score = better fit (less fragmentation).
    """
    if not used:
        return x * 1.0

    if not isinstance(used, list):
        used_sorted = sorted(used)
    else:
        used_sorted = used

    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist

    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _local_search_improve(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search: repeatedly try to reduce the maximum element by reconstructing
    the row that contains it, using constraint propagation.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute row differences
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    max_iter = 500
    
    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        # Find the row with the maximum element
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Try to reconstruct this row with a smaller maximum
        target_max = current_max - 1
        improved = False
        
        for attempt in range(5):
            if target_max < 1:
                break
            
            # Build used_diffs excluding this row
            used_diffs = set()
            for i in range(n_rows):
                if i != max_idx:
                    used_diffs.update(row_diffs[i])
            
            # Try to build a new row with elements <= target_max
            new_row = _build_row_constrained(used_diffs, k, target_max, rng, max_idx + attempt)
            
            if new_row is not None:
                # Verify no conflicts
                n = len(new_row)
                new_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(new_row[j] - new_row[i])
                
                if not (new_diffs & used_diffs):
                    # Check if this reduces the scope
                    new_scope_candidate = max(
                        max(rows[i][-1] for i in range(n_rows) if i != max_idx),
                        new_row[-1]
                    )
                    
                    if new_scope_candidate < best_scope:
                        rows[max_idx] = new_row
                        row_diffs[max_idx] = new_diffs
                        best_scope = new_scope_candidate
                        best_rows = [list(r) for r in rows]
                        improved = True
                        break
        
        if not improved:
            # Try a different row
            # Pick a random row and try to improve it
            if rng.random() < 0.5:
                row_idx = rng.randint(0, n_rows - 1)
                current_row_max = rows[row_idx][-1]
                
                used_diffs = set()
                for i in range(n_rows):
                    if i != row_idx:
                        used_diffs.update(row_diffs[i])
                
                # Try to find a row with smaller max
                for target in range(current_row_max - 1, max(current_row_max - 5, 0), -1):
                    new_row = _build_row_constrained(used_diffs, k, target, rng, row_idx)
                    if new_row is not None:
                        n = len(new_row)
                        new_diffs = set()
                        for i in range(n):
                            for j in range(i + 1, n):
                                new_diffs.add(new_row[j] - new_row[i])
                        
                        if not (new_diffs & used_diffs):
                            new_scope_candidate = max(
                                max(rows[i][-1] for i in range(n_rows) if i != row_idx),
                                new_row[-1]
                            )
                            if new_scope_candidate < best_scope:
                                rows[row_idx] = new_row
                                row_diffs[row_idx] = new_diffs
                                best_scope = new_scope_candidate
                                best_rows = [list(r) for r in rows]
                            break
                break  # Only try one random row per iteration
    
    return best_rows

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

        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]

        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)

    return row

def _construct_with_max_constraint(n_rows, k, max_elem, start_time, deadline):
    """
    Try to construct n_rows rows with all elements <= max_elem.
    Uses backtracking with constraint propagation.
    Returns list of rows if successful, None otherwise.
    """
    rng = random.Random(int(time.time() * 1000) % 100000)
    used_diffs = set()
    rows = []
    
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    if max_elem < total_diffs_needed:
        return None
    
    for row_idx in range(n_rows):
        if time.time() > deadline - 5:
            return None
        
        row = _build_row_constrained(used_diffs, k, max_elem, rng, row_idx)
        if row is None:
            success = False
            for attempt in range(3):
                rng2 = random.Random(rng.randint(0, 10**9))
                row = _build_row_constrained(used_diffs, k, max_elem, rng2, row_idx + attempt)
                if row is not None:
                    success = True
                    break
            if not success:
                return None
        
        rows.append(row)
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                used_diffs.add(row[j] - row[i])
    
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)
    
    return rows

def _polish_max(rows, n_rows, k, start_time, deadline):
    """Final polish: repeatedly try to reduce the maximum element."""
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    max_iter = 200

    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)

    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break

        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]

        improved_row = _reduce_max_row(rows, max_idx, k, rng, current_max)

        if improved_row is not None and improved_row[-1] < current_max:
            n = len(improved_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(improved_row[j] - improved_row[i])

            other_diffs = set()
            for i in range(n_rows):
                if i != max_idx:
                    other_diffs.update(row_diffs[i])

            if not (new_diffs & other_diffs):
                new_rows = rows[:]
                new_rows[max_idx] = improved_row
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    rows = new_rows
                    row_diffs[max_idx] = new_diffs

    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _reduce_max_row(rows, row_idx, k, rng, current_max):
    """Try to find a replacement row with a smaller maximum element."""
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
    for attempt in range(15):
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

            scored_xs.sort(key=lambda t: t[0])
            top_n = min(3, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            local_used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(local_sorted, d)

        if success and len(row) == k + 1:
            return row

        target_max = min(target_max + 2, current_max - 1)

    return None

def _build_row_constrained(used_diffs, k, max_elem, rng, row_idx):
    """
    Build a single row of k+1 increasing integers starting at 0, all <= max_elem,
    such that all differences are not in used_diffs and are distinct within the row.
    """
    row = [0]
    
    def backtrack(current_row, depth):
        if depth == k:
            return list(current_row)
        
        last_val = current_row[-1]
        
        current_internal_diffs = set()
        n = len(current_row)
        for i in range(n):
            for j in range(i+1, n):
                current_internal_diffs.add(current_row[j] - current_row[i])
        
        candidates = []
        for x in range(last_val + 1, max_elem + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            if new_diffs & current_internal_diffs:
                continue
            candidates.append(x)
        
        if not candidates:
            return None
        
        if len(candidates) > 1:
            weights = [1.0 / (1 + i) for i in range(len(candidates))]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = len(candidates) - 1
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            candidates[chosen_idx], candidates[0] = candidates[0], candidates[chosen_idx]
        
        for x in candidates:
            current_row.append(x)
            result = backtrack(current_row, depth + 1)
            if result is not None:
                return result
            current_row.pop()
        
        return None
    
    return backtrack(row, 0)
