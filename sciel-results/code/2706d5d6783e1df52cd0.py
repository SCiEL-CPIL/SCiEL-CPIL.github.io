import time
import random
import math
from collections import defaultdict

def _scope(rows):
    if not rows:
        return 0
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

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s

def _build_diff_counter(rows):
    """Build a counter of all differences across all rows."""
    counter = defaultdict(int)
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counter[d] += 1
    return counter

def _collision_count_from_counter(counter):
    """Count number of collisions (differences appearing more than once)."""
    collisions = 0
    for d, cnt in counter.items():
        if cnt > 1:
            collisions += cnt - 1
    return collisions

def _greedy_init(n_rows, k, rng, scope_limit=None):
    """
    Greedy construction that tries to minimize scope.
    For each row, pick the smallest valid next element.
    """
    best_rows = None
    best_scope = float('inf')
    
    for restart in range(30):
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                start = row[-1] + 1
                if scope_limit is not None:
                    end = scope_limit + 1
                else:
                    end = start + 300
                for x in range(start, end):
                    if scope_limit is not None and x > scope_limit:
                        break
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        
        if success:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Last resort: no scope limit
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
    
    return best_rows, best_scope

def _csp_build_row(pos, current_row, used_diffs, max_val, deadline, k):
    """
    Build a row of length k+1 starting at 0, strictly increasing,
    with all diffs not in used_diffs and internally distinct.
    Uses MRV heuristic for candidate ordering.
    """
    if time.time() > deadline:
        return None
    n = k + 1
    if pos == n:
        return list(current_row)
    
    prev = current_row[-1]
    min_next = prev + 1
    max_next = min(max_val, prev + (n - pos - 1))
    
    if min_next > max_next:
        return None
    
    # Generate candidates and sort by MRV (fewest conflicts first)
    candidates = []
    for val in range(min_next, max_next + 1):
        # Count how many conflicts this value would create
        conflict_count = 0
        for idx in range(pos):
            d = val - current_row[idx]
            if d in used_diffs:
                conflict_count += 1
        candidates.append((conflict_count, val))
    
    # Sort by conflict count, then by value
    candidates.sort()
    
    # Take top candidates (limit to avoid excessive branching)
    top_candidates = [val for _, val in candidates[:20]]
    
    # Add some randomness to break ties
    if len(top_candidates) > 10:
        rng_local = random.Random(pos * 1000 + len(current_row) * 7)
        rng_local.shuffle(top_candidates[5:])
    
    for val in top_candidates:
        if time.time() > deadline:
            return None
        
        # Check conflicts with used_diffs
        conflict = False
        for idx in range(pos):
            d = val - current_row[idx]
            if d in used_diffs:
                conflict = True
                break
        if conflict:
            continue
        
        # Check internal consistency
        new_diffs = [val - current_row[idx] for idx in range(pos)]
        existing_diffs = set()
        for i in range(len(current_row)):
            for j in range(i + 1, len(current_row)):
                existing_diffs.add(current_row[j] - current_row[i])
        for d in new_diffs:
            if d in existing_diffs:
                conflict = True
                break
        if conflict:
            continue
        
        current_row.append(val)
        result = _csp_build_row(pos + 1, current_row, used_diffs, max_val, deadline, k)
        current_row.pop()
        
        if result is not None:
            return result
    
    return None

def _csp_construct(n_rows, k, max_val, deadline, rng):
    """
    Construct n_rows using CSP backtracking with MRV heuristic.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        best_row = None
        best_max = max_val + 1
        
        for attempt in range(40):
            if time.time() > deadline:
                break
            
            current_row = [0]
            result = _csp_build_row(1, current_row, used_diffs, max_val, deadline, k)
            
            if result is not None:
                row_max = result[-1]
                if row_max < best_max:
                    best_max = row_max
                    best_row = result
                if best_max <= max_val:
                    break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        for d in _diffs_for_row(best_row):
            used_diffs.add(d)
    
    return rows

def _min_conflicts_search(rows, k, deadline, rng):
    """
    Min-conflicts local search: start with a random assignment,
    then iteratively move one row at a time to reduce total conflicts.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    
    # Build initial diff counter
    counter = _build_diff_counter(current)
    current_collisions = _collision_count_from_counter(counter)
    
    best = [list(r) for r in current]
    best_collisions = current_collisions
    best_scope = _scope(current)
    
    T = 50.0
    cooling = 0.9999
    min_T = 0.01
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Pick a random row to move
        i = rng.randint(0, n_rows - 1)
        
        # Remove this row's diffs from counter
        old_diffs = _diffs_for_row(current[i])
        for d in old_diffs:
            counter[d] -= 1
            if counter[d] == 0:
                del counter[d]
        
        # Try to find a better row for position i
        best_new_row = None
        best_new_collisions = current_collisions
        
        # Generate candidate rows by modifying the current row
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Create a perturbed version of the current row
            candidate = list(current[i])
            
            # Randomly perturb some elements
            num_perturb = rng.randint(1, min(3, len(candidate) - 1))
            indices = rng.sample(range(1, len(candidate)), num_perturb)
            for idx in indices:
                delta = rng.randint(-10, 10)
                new_val = candidate[idx] + delta
                if new_val > candidate[idx - 1] and (idx == len(candidate) - 1 or new_val < candidate[idx + 1]):
                    candidate[idx] = new_val
            
            # Check if candidate is valid (strictly increasing, starts at 0)
            if candidate[0] != 0 or not all(candidate[j] < candidate[j+1] for j in range(len(candidate)-1)):
                continue
            
            # Compute new diffs
            new_diffs = _diffs_for_row(candidate)
            
            # Check internal consistency
            if len(set(new_diffs)) != len(new_diffs):
                continue
            
            # Compute collisions if we use this candidate
            temp_collisions = 0
            for d in new_diffs:
                temp_collisions += max(0, counter.get(d, 0))
            
            if temp_collisions < best_new_collisions:
                best_new_collisions = temp_collisions
                best_new_row = candidate
            
            # Accept with probability if not better
            if temp_collisions <= current_collisions:
                prob = 1.0
            else:
                delta = temp_collisions - current_collisions
                prob = math.exp(-delta / max(T, min_T))
            
            if rng.random() < prob and best_new_row is None:
                best_new_row = candidate
                best_new_collisions = temp_collisions
        
        if best_new_row is not None:
            # Update counter
            new_diffs = _diffs_for_row(best_new_row)
            for d in new_diffs:
                counter[d] = counter.get(d, 0) + 1
            
            current[i] = best_new_row
            current_collisions = best_new_collisions
            
            new_scope = _scope(current)
            if current_collisions < best_collisions or (current_collisions == best_collisions and new_scope < best_scope):
                best_collisions = current_collisions
                best_scope = new_scope
                best = [list(r) for r in current]
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        else:
            no_improve_count += 1
        
        T = max(T * cooling, min_T)
    
    # Final pass: try to reduce scope while keeping collisions at 0
    if best_collisions == 0:
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Find rows with max value
            max_val = _scope(best)
            max_rows = [i for i, r in enumerate(best) if r[-1] == max_val]
            if not max_rows:
                break
            
            progress = False
            for i in max_rows:
                # Try to rebuild this row with max_val - 1
                other_used = set()
                for j in range(n_rows):
                    if j != i:
                        other_used.update(_row_diffs_set(best[j]))
                
                target_max = max_val - 1
                if target_max < k:
                    continue
                
                new_row = None
                for attempt in range(20):
                    if time.time() > deadline:
                        break
                    current_row = [0]
                    new_row = _csp_build_row(1, current_row, other_used, target_max, deadline, k)
                    if new_row is not None:
                        break
                
                if new_row is not None:
                    new_diff_set = _row_diffs_set(new_row)
                    if not (new_diff_set & other_used):
                        best[i] = new_row
                        progress = True
                        break
            
            if not progress:
                break
    
    return best_scope, best

def _sa_optimize(rows, k, deadline, rng):
    """
    Simulated Annealing with row-rebuild moves.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    row_diff_sets = [_row_diffs_set(r) for r in current]
    
    best = [list(r) for r in current]
    best_scope = current_scope
    
    T = 40.0
    cooling = 0.99995
    min_T = 0.01
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        
        other_used = set()
        for j in range(n_rows):
            if j != i:
                other_used.update(row_diff_sets[j])
        
        target_max = min(current_scope, current[i][-1] + rng.randint(-8, 8))
        target_max = max(target_max, k)
        
        new_row = None
        for attempt in range(25):
            if time.time() > deadline:
                break
            current_row = [0]
            new_row = _csp_build_row(1, current_row, other_used, target_max, deadline, k)
            if new_row is not None:
                break
        
        if new_row is None:
            continue
        
        new_row_max = new_row[-1]
        new_diff_set = _row_diffs_set(new_row)
        
        if new_diff_set & other_used:
            continue
        
        new_scope = max(r[-1] for j, r in enumerate(current) if j != i)
        new_scope = max(new_scope, new_row_max)
        
        old_c = current_scope
        new_c = new_scope
        
        if new_c <= old_c:
            accept = True
        else:
            delta = new_c - old_c
            prob = math.exp(-delta / max(T, min_T))
            accept = rng.random() < prob
        
        if accept:
            current[i] = new_row
            row_diff_sets[i] = new_diff_set
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [list(r) for r in current]
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def _scope_squeeze(rows, k, deadline, rng):
    """
    Iteratively try to reduce scope by 1.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    
    for _ in range(300):
        if time.time() > deadline:
            break
        
        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break
        
        progress = False
        for i in max_rows:
            if time.time() > deadline:
                break
            
            other_used = set()
            for j in range(len(rows)):
                if j != i:
                    other_used.update(_row_diffs_set(rows[j]))
            
            target_max = current_scope - 1
            if target_max < k:
                continue
            
            new_row = None
            for attempt in range(25):
                if time.time() > deadline:
                    break
                current_row = [0]
                new_row = _csp_build_row(1, current_row, other_used, target_max, deadline, k)
                if new_row is not None:
                    break
            
            if new_row is not None:
                new_diff_set = _row_diffs_set(new_row)
                if not (new_diff_set & other_used):
                    rows[i] = new_row
                    current_scope = target_max
                    progress = True
                    break
        
        if not progress:
            break
    
    return rows

def construct(n_rows, k):
    deadline = time.time() + 140.0
    rng = random.Random(12345)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: CSP construction with binary search on scope
    if time.time() < deadline - 80:
        total_diffs = n_rows * k * (k + 1) // 2
        low = max(1, total_diffs)
        high = 250
        
        best_rows_csp = None
        best_M = high
        
        for M in range(high, low - 1, -7):
            if time.time() > deadline - 80:
                break
            rows = _csp_construct(n_rows, k, M, deadline - 80, rng)
            if rows is not None:
                s = _scope(rows)
                if s < best_score:
                    best_score = s
                    best_result = rows
                    best_M = M
                if M <= low:
                    break
        
        if best_result is not None and time.time() < deadline - 50:
            for M in range(max(low, best_M - 15), min(high, best_M + 1)):
                if time.time() > deadline - 50:
                    break
                rows = _csp_construct(n_rows, k, M, deadline - 50, rng)
                if rows is not None:
                    s = _scope(rows)
                    if s < best_score:
                        best_score = s
                        best_result = rows
    
    # Phase 2: Greedy fallback if CSP failed
    if best_result is None:
        rows, s = _greedy_init(n_rows, k, rng)
        best_result = rows
        best_score = s
    
    # Phase 3: Min-conflicts search
    if time.time() < deadline - 30:
        mc_deadline = time.time() + (deadline - time.time()) * 0.5
        s, rows = _min_conflicts_search(best_result, k, mc_deadline, rng)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 4: SA optimization
    if time.time() < deadline - 20:
        sa_deadline = time.time() + (deadline - time.time()) * 0.6
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 5: Compress
    if time.time() < deadline - 5:
        compress_deadline = time.time() + (deadline - time.time()) * 0.8
        squeezed = _scope_squeeze(best_result, k, compress_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed
    
    # Final validation
    if best_result is not None and not _valid(best_result, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows
    
    return best_result
