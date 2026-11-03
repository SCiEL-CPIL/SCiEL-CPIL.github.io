<<<<<<< SEARCH
"""Difference triangle set (4,7) — simulated annealing with conflict-based repair."""
import time
import random
import sys

# EVOLVE-BLOCK-START


def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _max_entry(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _count_conflicts(rows):
    """Count how many duplicate differences exist across all rows."""
    from collections import Counter
    seen = Counter()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                seen[d] += 1
    conflicts = 0
    for d, cnt in seen.items():
        if cnt > 1:
            conflicts += cnt - 1
    return conflicts


def _greedy_build(n_rows, k, rng, bias=1.0):
    """Greedy build with optional randomness."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if bias >= 1.0 or rng.random() < bias:
                    row.append(x)
                    used |= new
                else:
                    if rng.random() > 0.3:
                        row.append(x)
                        used |= new
        rows.append(row)
    return rows


def _sa_optimize(rows, deadline, rng):
    """
    Simulated annealing on the full set of rows.
    Moves: pick a random row and a random position, then try to compact
    that element or the whole row by shifting elements down.
    Accepts invalid states with probability based on conflict count and temperature.
    """
    n_rows = len(rows)
    k = len(rows[0]) - 1
    
    # Current state
    current = [r[:] for r in rows]
    current_conflicts = _count_conflicts(current)
    current_scope = _max_entry(current)
    
    # Best state
    best = [r[:] for r in current]
    best_scope = current_scope
    best_conflicts = current_conflicts
    
    # Temperature schedule
    t_start = time.time()
    total_time = deadline - t_start
    t0 = 100.0  # initial temperature (in terms of conflict units)
    t_end = 0.01
    
    def temperature(elapsed_frac):
        # Exponential cooling
        return t0 * (t_end / t0) ** elapsed_frac
    
    steps = 0
    while time.time() < deadline:
        elapsed_frac = (time.time() - t_start) / max(total_time, 1e-9)
        if elapsed_frac >= 1.0:
            break
        T = temperature(elapsed_frac)
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Compact a single element: try to reduce row[r][p] to a smaller value
            r_idx = rng.randrange(n_rows)
            if len(current[r_idx]) <= 1:
                continue
            p = rng.randrange(1, len(current[r_idx]))
            old_val = current[r_idx][p]
            
            # Try values from old_val-1 down to current[r_idx][p-1]+1
            found = False
            for new_val in range(old_val - 1, current[r_idx][p - 1], -1):
                # Check if this creates a conflict within the row
                row = current[r_idx]
                row[p] = new_val
                # Check intra-row distinctness
                valid_intra = True
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if row[j] == row[i]:
                            valid_intra = False
                            break
                    if not valid_intra:
                        break
                if not valid_intra:
                    row[p] = old_val
                    continue
                # Check if this creates inter-row conflicts
                # Only check if new_val - row[i] or row[j] - new_val conflicts with other rows
                new_diffs = set()
                for i in range(len(row)):
                    if i != p:
                        d = abs(row[p] - row[i])
                        new_diffs.add(d)
                
                # Check against other rows
                conflict_count = 0
                for other_idx in range(n_rows):
                    if other_idx == r_idx:
                        continue
                    other_row = current[other_idx]
                    for i in range(len(other_row)):
                        for j in range(i + 1, len(other_row)):
                            d = other_row[j] - other_row[i]
                            if d in new_diffs:
                                conflict_count += 1
                
                if conflict_count == 0:
                    found = True
                    break
                else:
                    row[p] = old_val
                    # Accept with probability based on conflicts
                    if rng.random() < T / max(conflict_count, 1):
                        found = True
                        break
            
            if found:
                new_scope = _max_entry(current)
                new_conflicts = _count_conflicts(current)
                # Accept if better or with SA probability
                delta_scope = new_scope - current_scope
                delta_conf = new_conflicts - current_conflicts
                if delta_scope < 0 or delta_conf < 0:
                    current = [r[:] for r in current]
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                    if new_scope < best_scope and new_conflicts == 0:
                        best = [r[:] for r in current]
                        best_scope = new_scope
                        best_conflicts = 0
                elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                    current = [r[:] for r in current]
                    current_scope = new_scope
                    current_conflicts = new_conflicts
        
        elif move_type < 0.8:
            # Rebuild a single row greedily with randomness
            r_idx = rng.randrange(n_rows)
            old_row = current[r_idx][:]
            
            # Collect used diffs from other rows
            used_others = set()
            for i in range(n_rows):
                if i != r_idx:
                    used_others |= _all_diffs(current[i])
            
            # Greedy rebuild with randomness
            new_row = [0]
            x = 0
            max_x = old_row[-1] + 5  # Allow some expansion
            while len(new_row) < len(old_row) and x < max_x:
                x += 1
                new = {x - a for a in new_row}
                if not new & used_others:
                    if rng.random() < 0.8:
                        new_row.append(x)
                    else:
                        if rng.random() < 0.4:
                            new_row.append(x)
            
            if len(new_row) == len(old_row):
                # Check validity
                trial = current[:]
                trial[r_idx] = new_row
                new_conflicts = _count_conflicts(trial)
                new_scope = _max_entry(trial)
                
                delta_scope = new_scope - current_scope
                delta_conf = new_conflicts - current_conflicts
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best = trial
                    best_scope = new_scope
                    best_conflicts = 0
                    current = trial
                    current_scope = new_scope
                    current_conflicts = 0
                elif delta_scope < 0 or delta_conf < 0:
                    current = trial
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                    current = trial
                    current_scope = new_scope
                    current_conflicts = new_conflicts
        
        else:
            # Swap elements between two rows (if same length)
            r1 = rng.randrange(n_rows)
            r2 = rng.randrange(n_rows)
            if r1 == r2:
                continue
            if len(current[r1]) != len(current[r2]):
                continue
            # Try swapping the last element
            old1 = current[r1][-1]
            old2 = current[r2][-1]
            if old1 == old2:
                continue
            current[r1][-1] = old2
            current[r2][-1] = old1
            
            new_conflicts = _count_conflicts(current)
            new_scope = _max_entry(current)
            delta_scope = new_scope - current_scope
            delta_conf = new_conflicts - current_conflicts
            
            if new_conflicts == 0 and new_scope < best_scope:
                best = [r[:] for r in current]
                best_scope = new_scope
                best_conflicts = 0
                current_scope = new_scope
                current_conflicts = new_conflicts
            elif delta_scope < 0 or delta_conf < 0:
                current_scope = new_scope
                current_conflicts = new_conflicts
            elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                current_scope = new_scope
                current_conflicts = new_conflicts
            else:
                # Reject
                current[r1][-1] = old1
                current[r2][-1] = old2
        
        steps += 1
        # Periodically check best
        if steps % 1000 == 0 and current_conflicts == 0:
            if current_scope < best_scope:
                best = [r[:] for r in current]
                best_scope = current_scope
                best_conflicts = 0
    
    # Return best valid solution
    if best_conflicts == 0:
        return best
    # Fallback: return current if it's valid
    if current_conflicts == 0:
        return current
    # Otherwise return best even if invalid (shouldn't happen often)
    return best


def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    rng = random.Random(42)
    
    # Start with multiple greedy baselines
    best = None
    best_score = float('inf')
    
    for seed in range(30):
        if time.time() > deadline - 10.0:
            break
        trial = _greedy_build(n_rows, k, random.Random(seed), bias=rng.choice([1.0, 0.95, 0.9, 0.85]))
        if _is_valid(trial):
            sc = _max_entry(trial)
            if sc < best_score:
                best_score = sc
                best = trial
    
    if best is None:
        # Fallback: use first valid greedy
        best = _greedy_build(n_rows, k, random.Random(0), bias=1.0)
        best_score = _max_entry(best)
    
    # Run simulated annealing
    best = _sa_optimize(best, deadline, rng)
    
    return best

# EVOLVE-BLOCK-END
=======
"""Difference triangle set (4,7) — backtracking with bitset tracking + local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START


def _max_entry(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _backtrack_construct(n_rows, k, max_scope, used_diffs, row_idx, deadline):
    """
    Backtracking construction of one row at a time.
    used_diffs is a boolean array where used_diffs[d] = True means difference d is already used.
    Returns the completed rows list or None if no solution found within deadline.
    """
    if time.time() > deadline:
        return None
    
    if row_idx == n_rows:
        return []
    
    row = [0]
    n_entries = k + 1  # need k+1 entries total including 0
    
    def build_row(pos, current_row, current_used):
        """Recursively build the current row."""
        if time.time() > deadline:
            return None
        
        if pos == n_entries:
            return current_row[:]
        
        last = current_row[-1]
        # Try next values starting from last+1
        for x in range(last + 1, max_scope + 1):
            # Check if adding x creates any new difference conflicts
            new_diffs = []
            ok = True
            for a in current_row:
                d = x - a
                if d > max_scope:
                    break
                if used_diffs[d] or d in current_used:
                    ok = False
                    break
                new_diffs.append(d)
            
            if not ok:
                continue
            
            # Also check intra-row: all differences must be distinct
            # Since we're building in order, new diffs are x - a for a in current_row
            # These are all distinct because current_row is strictly increasing
            # But we need to check they don't conflict with each other (they can't since x-a is unique for each a)
            
            current_used.update(new_diffs)
            current_row.append(x)
            
            result = build_row(pos + 1, current_row, current_used)
            if result is not None:
                return result
            
            current_row.pop()
            for d in new_diffs:
                current_used.discard(d)
        
        return None
    
    result = build_row(1, [0], set())
    if result is None:
        return None
    
    # Mark all diffs from this row as used
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d = result[j] - result[i]
            used_diffs[d] = True
    
    remaining = _backtrack_construct(n_rows, k, max_scope, used_diffs, row_idx + 1, deadline)
    if remaining is None:
        # Undo
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d = result[j] - result[i]
                used_diffs[d] = False
        return None
    
    return [result] + remaining


def _local_search(rows, deadline, rng):
    """
    Local search: try to reduce the scope by moving individual elements to smaller values.
    Uses a boolean array for O(1) difference lookup.
    """
    n_rows = len(rows)
    k = len(rows[0]) - 1
    current = [r[:] for r in rows]
    current_scope = _max_entry(current)
    best = [r[:] for r in current]
    best_scope = current_scope
    
    # Build used_diffs array
    max_possible = current_scope + 50
    used = [False] * (max_possible + 1)
    for row in current:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= max_possible:
                    used[d] = True
    
    t_start = time.time()
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # Try to reduce each element in each row
        for r_idx in range(n_rows):
            row = current[r_idx]
            for p in range(1, len(row)):
                old_val = row[p]
                # Try smaller values
                for new_val in range(row[p-1] + 1, old_val):
                    # Check if new_val creates valid intra-row differences
                    # and doesn't conflict with used differences
                    new_diffs = []
                    ok = True
                    for i in range(len(row)):
                        if i == p:
                            continue
                        d = abs(new_val - row[i])
                        # Check intra-row: no duplicate differences within this row
                        if d in new_diffs:
                            ok = False
                            break
                        # Check against used differences (excluding those from this row that involve position p)
                        new_diffs.append(d)
                    
                    if not ok:
                        continue
                    
                    # Check against global used differences
                    # First, remove this row's differences involving position p from used
                    for i in range(len(row)):
                        if i != p:
                            d = abs(row[p] - row[i])
                            if d <= max_possible:
                                used[d] = False
                    
                    # Now check if any new_diff is in used
                    conflict = False
                    for d in new_diffs:
                        if d <= max_possible and used[d]:
                            conflict = True
                            break
                    
                    if conflict:
                        # Restore
                        for i in range(len(row)):
                            if i != p:
                                d = abs(row[p] - row[i])
                                if d <= max_possible:
                                    used[d] = True
                        continue
                    
                    # Check intra-row: all differences in the row must be distinct
                    row_diffs = set()
                    valid = True
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in row_diffs:
                                valid = False
                                break
                            row_diffs.add(d)
                        if not valid:
                            break
                    
                    if not valid:
                        # Restore
                        for i in range(len(row)):
                            if i != p:
                                d = abs(row[p] - row[i])
                                if d <= max_possible:
                                    used[d] = True
                        continue
                    
                    # Accept the move
                    row[p] = new_val
                    # Update used array
                    for i in range(len(row)):
                        if i != p:
                            d = abs(new_val - row[i])
                            if d <= max_possible:
                                used[d] = True
                    
                    new_scope = _max_entry(current)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]
                    improved = True
                    no_improve_count = 0
                    break  # Move to next position
                
                if time.time() > deadline:
                    return best
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
            # Random perturbation: swap two elements from different rows
            if n_rows >= 2:
                r1 = rng.randrange(n_rows)
                r2 = rng.randrange(n_rows)
                if r1 != r2:
                    # Try swapping last elements
                    v1 = current[r1][-1]
                    v2 = current[r2][-1]
                    if v1 != v2:
                        # Temporarily swap and check validity
                        current[r1][-1] = v2
                        current[r2][-1] = v1
                        if _is_valid(current):
                            new_scope = _max_entry(current)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best = [r[:] for r in current]
                            improved = True
                            no_improve_count = 0
                        else:
                            current[r1][-1] = v1
                            current[r2][-1] = v2
    
    return best


def _randomized_greedy(n_rows, k, rng, max_scope_hint=None):
    """
    Randomized greedy construction.
    Builds rows one at a time, choosing the smallest available value that doesn't create conflicts.
    With randomness: sometimes skip the smallest and pick a larger one to explore different regions.
    """
    if max_scope_hint is None:
        max_scope_hint = 200
    
    used = [False] * (max_scope_hint + 1)
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        n_entries = k + 1
        
        while len(row) < n_entries:
            last = row[-1]
            found = False
            
            # Try values in order, but with some randomization
            candidates = list(range(last + 1, max_scope_hint + 1))
            
            # With some probability, shuffle a portion to introduce randomness
            if rng.random() < 0.3:
                # Shuffle a window
                window_size = min(20, len(candidates))
                start = rng.randrange(max(0, len(candidates) - window_size))
                end = min(len(candidates), start + window_size)
                candidates[start:end] = rng.sample(candidates[start:end], end - start)
            
            for x in candidates:
                # Check if adding x creates any new difference conflicts
                new_diffs = [x - a for a in row]
                ok = True
                for d in new_diffs:
                    if d > max_scope_hint:
                        ok = False
                        break
                    if used[d]:
                        ok = False
                        break
                    # Check intra-row distinctness (new diffs must be distinct)
                    if new_diffs.count(d) > 1:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # Also check that all pairwise differences in the new row are distinct
                test_row = row + [x]
                row_diffs = set()
                valid = True
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        if d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Accept
                for d in new_diffs:
                    if d <= max_scope_hint:
                        used[d] = True
                row.append(x)
                found = True
                break
            
            if not found:
                # Couldn't complete this row; try with a larger scope or fail
                return None
        
        rows.append(row)
    
    return rows


def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best = None
    best_scope = float('inf')
    
    # Phase 1: Multiple randomized greedy starts
    max_greedy_attempts = 50
    for attempt in range(max_greedy_attempts):
        if time.time() > deadline - 30.0:
            break
        seed = rng.randrange(10000)
        trial = _randomized_greedy(n_rows, k, random.Random(seed), max_scope_hint=150)
        if trial is not None and _is_valid(trial):
            sc = _max_entry(trial)
            if sc < best_scope:
                best_scope = sc
                best = trial
    
    if best is None:
        # Fallback: pure greedy
        trial = _randomized_greedy(n_rows, k, random.Random(0), max_scope_hint=200)
        if trial is not None:
            best = trial
            best_scope = _max_entry(trial)
        else:
            # Last resort: construct a trivial valid solution
            best = [[0, 1, 3, 6, 10, 15, 21, 28, 36]] * n_rows
            # Make them distinct by offsetting
            for i in range(n_rows):
                best[i] = [x + i * 100 for x in [0, 1, 3, 6, 10, 15, 21, 28, 36]]
            best_scope = _max_entry(best)
    
    # Phase 2: Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        best = _local_search(best, deadline, rng)
        best_scope = _max_entry(best)
    
    # Phase 3: Try backtracking with tighter bounds if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Try to improve via backtracking with scope = best_scope - 1
        target_scope = best_scope - 1
        max_backtrack_attempts = 5
        for attempt in range(max_backtrack_attempts):
            if time.time() > deadline - 5.0:
                break
            used = [False] * (target_scope + 1)
            result = _backtrack_construct(n_rows, k, target_scope, used, 0, deadline - 5.0)
            if result is not None and _is_valid(result):
                sc = _max_entry(result)
                if sc < best_scope:
                    best_scope = sc
                    best = result
                break
            # Try with slightly larger scope
            target_scope = min(best_scope, target_scope + 2)
    
    return best

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
