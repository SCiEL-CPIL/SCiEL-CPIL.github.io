import time
import random
import math
from collections import Counter, defaultdict

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
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _build_diff_counter(rows):
    c = Counter()
    for r in rows:
        for d in _diffs_for_row(r):
            c[d] += 1
    return c

def _collision_count_from_counter(counter):
    return sum(max(0, v - 1) for v in counter.values())

def _count_conflicts(diff_counter):
    """Count total conflicts: sum of (count-1) for each diff with count > 1"""
    return sum(max(0, v - 1) for v in diff_counter.values())

def _generate_valid_row(used_diffs, k, max_val, rng, attempts=50):
    """
    Generate a valid row of length k+1 starting at 0, strictly increasing,
    with all differences distinct within the row and not in used_diffs.
    Returns the row or None.
    """
    for _ in range(attempts):
        row = [0]
        current_diffs = set()
        valid = True
        for pos in range(1, k + 1):
            # Try to find a value that works
            prev = row[-1]
            min_next = prev + 1
            max_next = min(max_val, prev + (k - pos + 1))  # need at least 1 step each
            if min_next > max_next:
                valid = False
                break
            
            # Try candidates in random order with bias toward smaller
            candidates = list(range(min_next, max_next + 1))
            # Bias toward smaller values for compactness
            rng.shuffle(candidates)
            # Sort with some randomness: first half sorted, second half shuffled
            half = len(candidates) // 2
            first_half = sorted(candidates[:half])
            second_half = candidates[half:]
            rng.shuffle(second_half)
            candidates = first_half + second_half
            
            found = False
            for val in candidates:
                # Check if this value creates conflicts
                conflict = False
                new_diffs = []
                for idx in range(len(row)):
                    d = val - row[idx]
                    if d in current_diffs or d in used_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if not conflict:
                    row.append(val)
                    current_diffs.update(new_diffs)
                    found = True
                    break
            
            if not found:
                valid = False
                break
        
        if valid and len(row) == k + 1:
            return row
    
    return None

def _construct_initial(n_rows, k, target_scope, rng, time_budget=30.0):
    """
    Construct initial solution using randomized greedy with multiple restarts.
    Tries to fit within target_scope.
    """
    deadline = time.time() + time_budget
    best_rows = None
    best_scope = float('inf')
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        used_diffs = set()
        rows = []
        success = True
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                success = False
                break
            
            # Try to generate a row within target_scope
            row = _generate_valid_row(used_diffs, k, target_scope, rng, attempts=30)
            if row is None:
                success = False
                break
            
            rows.append(row)
            for d in _diffs_for_row(row):
                used_diffs.add(d)
        
        if success and len(rows) == n_rows:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
                if s <= target_scope:
                    break
    
    return best_rows

def _min_conflicts_search(rows, k, deadline, rng, max_iters=None):
    """
    Min-conflicts style local search to resolve collisions and minimize scope.
    Each move: pick a row with conflicts, try to reassign one of its elements
    to reduce conflicts.
    """
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 500000
    
    current = [list(r) for r in rows]
    
    # Build diff counter
    diff_counter = _build_diff_counter(current)
    
    best = [list(r) for r in current]
    best_scope = _scope(current)
    best_conflicts = _count_conflicts(diff_counter)
    
    T = 100.0
    cooling = 0.99995
    min_T = 0.1
    
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1
        
        # Find rows with conflicts
        # A row has a conflict if any of its diffs appear more than once globally
        conflicted_rows = []
        for i, r in enumerate(current):
            for d in _diffs_for_row(r):
                if diff_counter[d] > 1:
                    conflicted_rows.append(i)
                    break
        
        if not conflicted_rows:
            # No conflicts! Try to minimize scope
            # Pick a random row and try to reduce its max
            i = rng.randint(0, n_rows - 1)
            row = current[i]
            old_max = row[-1]
            if old_max <= 0:
                continue
            prev = row[k - 1]
            
            # Try to reduce the last element
            for new_val in range(old_max - 1, prev, -1):
                row[k] = new_val
                new_diffs = _diffs_for_row(row)
                
                # Check internal validity
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                
                # Check cross-row conflicts
                conflict = False
                new_diffs_set = set(new_diffs)
                for other_idx, other_row in enumerate(current):
                    if other_idx == i:
                        continue
                    for d in new_diffs_set:
                        if d in _row_diffs_set(other_row):
                            conflict = True
                            break
                    if conflict:
                        break
                
                if not conflict:
                    # Accept
                    current_scope = _scope(current)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best = [list(r) for r in current]
                    break
                else:
                    row[k] = old_max
                    break
            continue
        
        # Pick a random conflicted row
        i = rng.choice(conflicted_rows)
        row = current[i]
        
        # Pick a random position to change (not 0)
        j = rng.randint(1, k)
        old_val = row[j]
        prev_val = row[j - 1]
        next_val = row[j + 1] if j < k else old_val + 10000
        
        # Generate candidate values
        # Bias toward values that might reduce conflicts
        candidates = set()
        
        # Try values near the current value
        for delta in range(-15, 16):
            cv = old_val + delta
            if cv > prev_val and (j >= k or cv < next_val):
                candidates.add(cv)
        
        # Try smaller values to reduce scope
        if old_val > prev_val + 1:
            for cv in range(prev_val + 1, min(old_val, prev_val + 20)):
                if j >= k or cv < next_val:
                    candidates.add(cv)
        
        # Try random values
        for _ in range(5):
            cv = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30))
            if j >= k or cv < next_val:
                candidates.add(cv)
        
        if not candidates:
            continue
        
        best_candidate = None
        best_new_conflicts = _count_conflicts(diff_counter)
        best_new_scope = _scope(current)
        
        for new_val in list(candidates):
            if new_val == old_val:
                continue
            
            # Apply change
            row[j] = new_val
            new_diffs = _diffs_for_row(row)
            
            # Check internal validity
            if len(new_diffs) != len(set(new_diffs)):
                row[j] = old_val
                continue
            
            # Update diff counter temporarily
            old_diffs = _diffs_for_row(row)
            # Wait, we already changed row. Let me redo this.
            
            # Revert to compute old diffs properly
            row[j] = old_val
            old_diffs = _diffs_for_row(row)
            row[j] = new_val
            new_diffs = _diffs_for_row(row)
            
            # Check internal validity again
            if len(new_diffs) != len(set(new_diffs)):
                row[j] = old_val
                continue
            
            # Compute new conflict count
            temp_counter = dict(diff_counter)
            for d in old_diffs:
                temp_counter[d] = temp_counter.get(d, 0) - 1
                if temp_counter[d] == 0:
                    del temp_counter[d]
            for d in new_diffs:
                temp_counter[d] = temp_counter.get(d, 0) + 1
            
            new_conflicts = _count_conflicts(temp_counter)
            new_scope = _scope(current)
            
            # Score: prefer fewer conflicts, then smaller scope
            score = new_conflicts * 10000 + new_scope
            
            if best_candidate is None or score < best_new_conflicts * 10000 + best_new_scope:
                best_candidate = new_val
                best_new_conflicts = new_conflicts
                best_new_scope = new_scope
        
        if best_candidate is None:
            continue
        
        # Apply the best candidate
        row[j] = best_candidate
        new_diffs = _diffs_for_row(row)
        
        # Update diff counter
        for d in old_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        for d in new_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1
        
        current_scope = _scope(current)
        current_conflicts = _count_conflicts(diff_counter)
        
        # Update best if valid and better
        if current_conflicts == 0 and current_scope < best_scope:
            best_scope = current_scope
            best = [list(r) for r in current]
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def _sa_optimize(rows, k, deadline, rng, max_iters=None):
    """
    Simulated Annealing to minimize scope with collision penalty.
    Uses efficient incremental diff counting.
    """
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 2000000

    current = [list(r) for r in rows]
    current_scope = _scope(current)

    # Build diff counter
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count_from_counter(diff_counter)

    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions

    collision_weight = 100000

    def cost(scope, coll):
        return coll * collision_weight + scope

    T = 50.0
    cooling = 0.9999
    min_T = 0.01

    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1

        # Choose random row and position
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)

        old_val = current[i][j]
        prev_val = current[i][j - 1]
        next_val = current[i][j + 1] if j < k else old_val + 10000

        # Move type
        move_type = rng.random()

        if move_type < 0.5:
            # Try to reduce (especially if near max)
            upper = old_val - 1 if old_val > prev_val + 1 else prev_val
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        elif move_type < 0.8:
            # Small perturbation
            delta = rng.randint(-8, 8)
            new_val = old_val + delta
        else:
            # Larger move
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30))

        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        if new_val == old_val:
            continue

        # Compute old diffs for this row
        old_row = current[i]
        old_diffs = _diffs_for_row(old_row)

        # Apply change
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)

        # Check internal validity (all diffs positive and distinct within row)
        if len(new_diffs) != len(set(new_diffs)):
            current[i][j] = old_val
            continue

        # Update diff counter
        for d in old_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]

        for d in new_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1

        new_collisions = _collision_count_from_counter(diff_counter)

        # New scope
        new_scope = current_scope
        if new_val > current_scope:
            new_scope = new_val
        elif old_val == current_scope and new_val < old_val:
            # Need to recompute
            new_scope = _scope(current)

        old_c = cost(current_scope, collisions)
        new_c = cost(new_scope, new_collisions)

        if new_c <= old_c:
            accept = True
        else:
            delta_c = new_c - old_c
            prob = math.exp(-delta_c / max(T, min_T))
            accept = rng.random() < prob

        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1

        T = max(T * cooling, min_T)

    return best_scope, best

def _scope_squeeze(rows, k, deadline, rng):
    """
    Try to reduce the scope by 1 iteratively using local search.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)

    # Precompute other rows' diff sets for faster checking
    other_diff_sets = []
    for i, r in enumerate(rows):
        s = _row_diffs_set(r)
        other_diff_sets.append(s)

    for _ in range(200):
        if time.time() > deadline:
            break

        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break

        i = rng.choice(max_rows)
        row = rows[i]
        old_max = row[-1]
        prev = row[k - 1]

        found = False
        for new_val in range(old_max - 1, prev, -1):
            row[k] = new_val
            new_diffs = _diffs_for_row(row)

            if len(new_diffs) != len(set(new_diffs)):
                continue

            # Check cross-row conflicts
            conflict = False
            new_diffs_set = set(new_diffs)
            for other_idx, other_set in enumerate(other_diff_sets):
                if other_idx == i:
                    continue
                for d in new_diffs_set:
                    if d in other_set:
                        conflict = True
                        break
                if conflict:
                    break

            if not conflict:
                current_scope = new_val
                found = True
                # Update the diff set for this row
                other_diff_sets[i] = new_diffs_set
                break

        if not found:
            row[k] = old_max
            continue

    return rows

def _fast_greedy(n_rows, k, rng, scope_limit=None):
    """
    Fast greedy construction. For each row, greedily pick the smallest
    value that doesn't create a duplicate difference.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
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
    return rows

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin
    rng = random.Random(12345)

    best_result = None
    best_score = float('inf')

    # Phase 1: Try to construct with target scope using randomized greedy
    # Start with a reasonable target and work down
    total_diffs = n_rows * k * (k + 1) // 2
    # Lower bound: at least total_diffs, but also each row needs at least k
    low_bound = max(total_diffs, k)
    
    # Try targets from low_bound upward
    target = low_bound
    max_target = 200  # known best is 112 for (7,5)
    
    while target <= max_target and time.time() < deadline - 40:
        rows = _construct_initial(n_rows, k, target, rng, time_budget=min(10.0, deadline - 40 - time.time()))
        if rows is not None:
            s = _scope(rows)
            if s < best_score:
                best_score = s
                best_result = [list(r) for r in rows]
                # If we achieved the target, try to go lower
                if s <= target:
                    target = max(low_bound, target - 3)
                else:
                    target += 2
            else:
                target += 2
        else:
            target += 2
    
    # Phase 2: Fallback to simple greedy if construction failed
    if best_result is None:
        rows = _fast_greedy(n_rows, k, rng)
        if rows is not None:
            best_result = rows
            best_score = _scope(rows)
        else:
            # Last resort: simple sequential
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
            best_score = _scope(rows)

    # Phase 3: Min-conflicts search to resolve any remaining conflicts and minimize scope
    if time.time() < deadline - 20:
        remaining = deadline - time.time()
        mc_deadline = time.time() + remaining * 0.5
        max_iters = int(remaining * 30000)
        s, rows = _min_conflicts_search(best_result, k, mc_deadline, rng, max_iters=max_iters)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows

    # Phase 4: Simulated Annealing optimization
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * 0.7
        max_iters = int(remaining * 50000)
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng, max_iters=max_iters)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows

    # Phase 5: Scope squeeze
    if time.time() < deadline - 3:
        remaining = deadline - time.time()
        squeeze_deadline = time.time() + remaining * 0.8
        squeezed = _scope_squeeze(best_result, k, squeeze_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed

    # Final validation
    if best_result is not None and not _valid(best_result, k):
        # Shouldn't happen, but fallback
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
