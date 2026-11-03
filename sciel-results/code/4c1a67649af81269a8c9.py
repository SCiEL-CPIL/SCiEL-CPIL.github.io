"""Difference triangle set (4,7) — backtracking + beam search approach."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diff_set(row):
    """Compute the set of all positive differences in a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _scope(rows):
    """Return the maximum entry (scope) of the rows."""
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, rng, strategy='nearest'):
    """Fast greedy construction to get an initial upper bound."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _backtrack_construct(n_rows, k, max_scope, rng, deadline, beam_width=1):
    """
    Iterative deepening / beam search backtracking.
    Tries to build n_rows of (k+1) elements each, all within [0, max_scope].
    Uses MRV (minimum remaining values) to order choices.
    Returns (rows, scope) or (None, None) if no solution found within max_scope.
    """
    if beam_width <= 1:
        # Pure backtracking with MRV
        return _backtrack_single(n_rows, k, max_scope, rng, deadline)
    else:
        # Beam search
        return _beam_search(n_rows, k, max_scope, rng, deadline, beam_width)

def _backtrack_single(n_rows, k, max_scope, rng, deadline):
    """
    Backtracking search: build rows one at a time.
    For each row, pick elements greedily with backtracking on conflicts.
    """
    used = set()
    rows = []
    
    def _build_row(row_start, current_used, max_s):
        """Build a single row starting from row_start, using backtracking."""
        if time.time() > deadline:
            return None
        row = list(row_start)
        k_needed = k + 1  # total elements
        
        def _backtrack_row(pos, cur_row, cur_used):
            if time.time() > deadline:
                return None
            if len(cur_row) == k_needed:
                return list(cur_row)
            
            # Try values from smallest to largest
            start_val = cur_row[-1] + 1 if cur_row else 1
            # Limit search: try a window of candidates
            # Use a randomized window for diversity
            max_try = min(max_s, start_val + 30)  # try up to 30 candidates
            
            # Collect valid candidates
            candidates = []
            for v in range(start_val, max_try + 1):
                new_diffs = {v - a for a in cur_row}
                if not (new_diffs & cur_used) and not (new_diffs & used):
                    candidates.append((v, new_diffs))
            
            if not candidates:
                # Try a wider window
                for v in range(start_val, max_s + 1):
                    new_diffs = {v - a for a in cur_row}
                    if not (new_diffs & cur_used) and not (new_diffs & used):
                        candidates.append((v, new_diffs))
                        if len(candidates) > 20:
                            break
            
            if not candidates:
                return None
            
            # Shuffle for diversity (but keep small values preferred)
            # Weighted shuffle: smaller values more likely
            if rng.random() < 0.3:
                rng.shuffle(candidates)
            
            for v, new_diffs in candidates:
                cur_row.append(v)
                cur_used_new = cur_used | new_diffs
                result = _backtrack_row(pos + 1, cur_row, cur_used_new)
                if result is not None:
                    return result
                cur_row.pop()
            
            return None
        
        return _backtrack_row(0, row, set())
    
    def _backtrack_rows(row_idx):
        if time.time() > deadline:
            return None
        if row_idx == n_rows:
            return list(rows)
        
        # Try to build this row
        row = _build_row([0], used, max_scope)
        if row is None:
            return None
        
        row_diffs = _row_diff_set(row)
        rows.append(row)
        used.update(row_diffs)
        
        result = _backtrack_rows(row_idx + 1)
        if result is not None:
            return result
        
        # Backtrack
        rows.pop()
        used.difference_update(row_diffs)
        return None
    
    result = _backtrack_rows(0)
    if result is not None:
        return result, max_scope
    return None, None

def _beam_search(n_rows, k, max_scope, rng, deadline, beam_width):
    """
    Beam search: maintain multiple partial assignments.
    Each state: (rows_so_far, used_diffs, current_row_being_built)
    """
    # State: (rows_list, used_set, current_row)
    # We build row by row, and within a row, element by element
    # Beam: list of (rows, used, current_row)
    
    beam = [([], set(), [0])]  # Start: no rows, no used diffs, current row starts with 0
    
    for row_idx in range(n_rows):
        new_beam = []
        for rows, used, cur_row in beam:
            if time.time() > deadline:
                break
            # Complete the current row
            completed = _complete_row_beam(cur_row, used, max_scope, k, rng, deadline)
            if completed is None:
                continue
            row_diffs = _row_diff_set(completed)
            new_rows = rows + [completed]
            new_used = used | row_diffs
            new_beam.append((new_rows, new_used, [0]))
        
        if not new_beam:
            return None, None
        
        # Prune beam: keep best by scope
        if len(new_beam) > beam_width:
            # Score by max element so far
            def score(state):
                rows, used, _ = state
                if not rows:
                    return 0
                return max(max(r) for r in rows)
            new_beam.sort(key=score)
            new_beam = new_beam[:beam_width]
        
        beam = new_beam
    
    if not beam:
        return None, None
    
    # Return the best (smallest scope)
    best = min(beam, key=lambda s: max(max(r) for r in s[0]) if s[0] else 0)
    return best[0], max_scope

def _complete_row_beam(cur_row, used, max_scope, k, rng, deadline):
    """Complete a row using greedy with limited backtracking."""
    row = list(cur_row)
    k_needed = k + 1
    
    def _try_complete(pos, cur_row, cur_used):
        if time.time() > deadline:
            return None
        if len(cur_row) == k_needed:
            return list(cur_row)
        
        start_val = cur_row[-1] + 1 if cur_row else 1
        # Try candidates
        found = None
        for v in range(start_val, min(max_scope + 1, start_val + 40)):
            new_diffs = {v - a for a in cur_row}
            if not (new_diffs & cur_used) and not (new_diffs & used):
                cur_row.append(v)
                result = _try_complete(pos + 1, cur_row, cur_used | new_diffs)
                if result is not None:
                    return result
                cur_row.pop()
        
        # If greedy window failed, try wider
        for v in range(start_val, max_scope + 1):
            new_diffs = {v - a for a in cur_row}
            if not (new_diffs & cur_used) and not (new_diffs & used):
                cur_row.append(v)
                result = _try_complete(pos + 1, cur_row, cur_used | new_diffs)
                if result is not None:
                    return result
                cur_row.pop()
        
        return None
    
    return _try_complete(0, row, set())

def _local_compact(rows, n_rows, k, rng, time_budget=10.0):
    """Compact rows by trying to reduce the maximum entry of each row."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start = time.time()
    
    row_diffs = [_row_diff_set(r) for r in best_rows]
    
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        for r_idx in range(n_rows):
            if time.time() - start > time_budget:
                break
            other_diffs = set()
            for i in range(n_rows):
                if i != r_idx:
                    other_diffs |= row_diffs[i]
            
            current_max = best_rows[r_idx][-1]
            # Try to reduce scope of this row
            for new_max in range(current_max - 1, max(k, 1) - 1, -1):
                new_row = _rebuild_row_greedy(best_rows[r_idx], other_diffs, new_max, k)
                if new_row is not None:
                    new_diff = _row_diff_set(new_row)
                    if not (new_diff & other_diffs):
                        best_rows[r_idx] = new_row
                        row_diffs[r_idx] = new_diff
                        new_scope = _scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        improved = True
                        break
    
    return best_rows, best_scope

def _rebuild_row_greedy(old_row, other_diffs, max_scope, k):
    """Rebuild a row greedily within max_scope, avoiding other_diffs."""
    row = [0]
    k_needed = k + 1
    used = set()
    
    while len(row) < k_needed:
        found = False
        for v in range(row[-1] + 1, max_scope + 1):
            new_diffs = {v - a for a in row}
            if not (new_diffs & other_diffs) and not (new_diffs & used):
                row.append(v)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _hill_climb(rows, n_rows, k, rng, time_budget=20.0):
    """
    Hill climbing: try to reduce the maximum entry by moving individual elements.
    Focus on the row with the largest entry.
    """
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    start = time.time()
    
    current_row_diffs = [_row_diff_set(r) for r in current_rows]
    
    no_improve_count = 0
    max_no_improve = 500
    
    while (time.time() - start) < time_budget and no_improve_count < max_no_improve:
        # Find the row with the largest entry
        max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
        row = list(current_rows[max_row_idx])
        
        # Try to reduce elements
        improved_this_iter = False
        
        # Try reducing the last element first
        if len(row) > 1:
            for new_last in range(row[-1] - 1, row[-2], -1):
                test_row = list(row)
                test_row[-1] = new_last
                test_diffs = _row_diff_set(test_row)
                
                # Check against other rows
                conflict = False
                for i in range(n_rows):
                    if i != max_row_idx:
                        if test_diffs & current_row_diffs[i]:
                            conflict = True
                            break
                if not conflict:
                    # Check internal consistency (should be fine since we only changed last)
                    current_rows[max_row_idx] = test_row
                    current_row_diffs[max_row_idx] = test_diffs
                    new_scope = _scope(current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                    current_scope = new_scope
                    improved_this_iter = True
                    break
        
        if not improved_this_iter:
            # Try moving other elements
            for pos in range(len(row) - 1, 0, -1):
                if time.time() - start > time_budget:
                    break
                for delta in range(1, 6):
                    new_val = row[pos] - delta
                    if new_val <= row[pos - 1]:
                        continue
                    if pos < len(row) - 1 and new_val >= row[pos + 1]:
                        continue
                    
                    test_row = list(row)
                    test_row[pos] = new_val
                    test_diffs = _row_diff_set(test_row)
                    
                    conflict = False
                    for i in range(n_rows):
                        if i != max_row_idx:
                            if test_diffs & current_row_diffs[i]:
                                conflict = True
                                break
                    if not conflict:
                        current_rows[max_row_idx] = test_row
                        current_row_diffs[max_row_idx] = test_diffs
                        new_scope = _scope(current_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in current_rows]
                        current_scope = new_scope
                        improved_this_iter = True
                        break
                if improved_this_iter:
                    break
        
        if improved_this_iter:
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct difference triangle set with minimal scope."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    rng = random.Random(12345)
    
    # Phase 1: Get initial upper bound via greedy
    initial_rows = _greedy_construct(n_rows, k, rng)
    initial_scope = _scope(initial_rows)
    
    best_rows = [list(r) for r in initial_rows]
    best_scope = initial_scope
    
    # Phase 2: Local compaction on initial solution
    time_left = deadline - time.time()
    compact_budget = min(15.0, time_left * 0.15)
    if compact_budget > 2:
        best_rows, best_scope = _local_compact(best_rows, n_rows, k, rng, time_budget=compact_budget)
    
    # Phase 3: Iterative deepening backtracking
    # Try to find a solution with scope < best_scope
    time_left = deadline - time.time()
    bt_budget = min(40.0, time_left * 0.35)
    
    if bt_budget > 3:
        bt_deadline = time.time() + bt_budget
        # Try decreasing scopes
        for target_scope in range(best_scope - 1, max(k, 1) - 1, -1):
            if time.time() > bt_deadline:
                break
            # Try multiple random seeds for diversity
            for attempt in range(3):
                if time.time() > bt_deadline:
                    break
                rng_attempt = random.Random(1000 + attempt * 7919 + target_scope)
                rows, scope = _backtrack_construct(n_rows, k, target_scope, rng_attempt, bt_deadline)
                if rows is not None:
                    if target_scope < best_scope:
                        best_scope = target_scope
                        best_rows = [list(r) for r in rows]
                    break  # Found a solution at this scope, stop trying lower
    
    # Phase 4: Beam search for additional diversity
    time_left = deadline - time.time()
    beam_budget = min(25.0, time_left * 0.25)
    if beam_budget > 3:
        beam_deadline = time.time() + beam_budget
        for target_scope in range(best_scope - 1, max(k, 1) - 1, -1):
            if time.time() > beam_deadline:
                break
            for bw in [3, 5]:
                if time.time() > beam_deadline:
                    break
                rng_beam = random.Random(2000 + target_scope * 31 + bw)
                rows, scope = _beam_search(n_rows, k, target_scope, rng_beam, beam_deadline, bw)
                if rows is not None:
                    if target_scope < best_scope:
                        best_scope = target_scope
                        best_rows = [list(r) for r in rows]
                    break
    
    # Phase 5: Hill climbing to squeeze out more
    time_left = deadline - time.time()
    hc_budget = min(20.0, time_left * 0.3)
    if hc_budget > 2:
        best_rows, best_scope = _hill_climb(best_rows, n_rows, k, rng, time_budget=hc_budget)
    
    # Phase 6: Final local compaction
    time_left = deadline - time.time()
    if time_left > 3:
        best_rows, best_scope = _local_compact(best_rows, n_rows, k, rng, time_budget=time_left - 2)
    
    # Final safety: verify validity
    used = set()
    for row in best_rows:
        rd = _row_diff_set(row)
        if rd & used:
            # Shouldn't happen, but fallback to greedy
            return _greedy_construct(n_rows, k, rng)
        used |= rd
    
    return best_rows

# EVOLVE-BLOCK-END
