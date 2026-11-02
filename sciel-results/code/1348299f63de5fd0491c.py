import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, then randomized local search
    with simulated annealing to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_used(rows):
        all_used = set()
        for row in rows:
            all_used |= _row_diffs(row)
        return all_used

    def _global_max(rows):
        return max(max(row) for row in rows)

    def local_search_init(n_rows, k, seed=None):
        """Greedy initialization with randomization."""
        if seed is not None:
            random.seed(seed)
        
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        """SA-based local search with CSP-style constraint propagation to minimize the maximum entry."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 500.0
        T_min = 0.5
        cool = 0.99998
        
        no_improve = 0
        max_no_improve = 25000
        restarts = 0
        max_restarts = 30
        
        # Find which row(s) have the max value
        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]
        
        # CSP-style: compute for each position in each row the set of forbidden values
        # due to cross-row conflicts. This helps generate better candidates.
        def _compute_forbidden(ri, pi, current_rows, all_used, row_diffs, k):
            """Compute forbidden values for position pi in row ri due to cross-row conflicts."""
            forbidden = set()
            row = current_rows[ri]
            old_diffs = row_diffs[ri]
            
            # For each candidate value v at position pi, check if it creates new cross-row conflicts
            # We'll sample some values to estimate the forbidden set
            min_allowed = row[pi - 1] + 1
            max_allowed = (row[pi + 1] - 1) if pi < k else current_val + 50
            
            # Sample a few values to identify common conflicts
            sample_vals = []
            if max_allowed - min_allowed > 10:
                step = (max_allowed - min_allowed) // 10
                for v in range(min_allowed, max_allowed + 1, step):
                    sample_vals.append(v)
            else:
                sample_vals = list(range(min_allowed, max_allowed + 1))
            
            for v in sample_vals:
                # Compute new diffs if we put v at position pi
                new_diffs = set()
                for i in range(k + 1):
                    if i == pi:
                        continue
                    if i < pi:
                        d = v - row[i]
                    else:
                        d = row[i] - v
                    new_diffs.add(d)
                
                # Check which new diffs conflict with other rows
                for d in new_diffs:
                    if d not in old_diffs and d in all_used:
                        forbidden.add(v)
                        break
            
            return forbidden, min_allowed, max_allowed
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Bias towards rows with the maximum value
            max_r = _max_rows()
            if max_r and random.random() < 0.6:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates - prioritize decreasing moves
            candidates = []
            
            # Decreasing candidates (primary optimization direction)
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(60, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            # Increasing candidates (exploration)
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
            max_increase = min(20, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            
            # Random jumps
            if random.random() < 0.25:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            # Swap move: try swapping two elements in the row
            if random.random() < 0.15 and k >= 2:
                pj = random.randrange(1, k + 1)
                if pj != pi:
                    # Only consider if it maintains ordering
                    if pj > pi:
                        if current_rows[ri][pi] < current_rows[ri][pj] and \
                           (pi == 1 or current_rows[ri][pi-1] < current_rows[ri][pj]) and \
                           (pj == k or current_rows[ri][pj-1] < current_rows[ri][pi]):
                            candidates.append(-current_rows[ri][pj])  # negative sentinel for swap
                    else:
                        if current_rows[ri][pj] < current_rows[ri][pi] and \
                           (pj == 1 or current_rows[ri][pj-1] < current_rows[ri][pi]) and \
                           (pi == k or current_rows[ri][pi-1] < current_rows[ri][pj]):
                            candidates.append(-current_rows[ri][pi])  # negative sentinel for swap
            
            # CSP-style: filter out obviously bad candidates using forbidden set
            if candidates and random.random() < 0.3:
                forbidden, f_min, f_max = _compute_forbidden(ri, pi, current_rows, all_used, row_diffs, k)
                if forbidden:
                    filtered = [c for c in candidates if c > 0 and c not in forbidden]
                    if filtered:
                        candidates = filtered
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            # Sort candidates: decreasing moves first (lower values preferred)
            candidates.sort(key=lambda v: (v > old_val, v))
            
            moved = False
            for new_val in candidates:
                # Handle swap moves (negative sentinel)
                if new_val < 0:
                    swap_target = -new_val
                    # Find which position has this value
                    pj = None
                    for idx in range(1, k+1):
                        if current_rows[ri][idx] == swap_target:
                            pj = idx
                            break
                    if pj is None or pj == pi:
                        continue
                    
                    # Perform swap
                    cand_row = current_rows[ri][:]
                    cand_row[pi], cand_row[pj] = cand_row[pj], cand_row[pi]
                    
                    # Check ordering
                    valid = True
                    for idx in range(len(cand_row) - 1):
                        if cand_row[idx] >= cand_row[idx+1]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Validate
                    new_diffs = set()
                    n = len(cand_row)
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = cand_row[j] - cand_row[i]
                            if d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    # Check cross-row conflicts
                    for d in new_diffs:
                        if d in all_used and d not in row_diffs[ri]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    cand_row_max = max(cand_row)
                    old_row_max = row_maxes[ri]
                    row_maxes[ri] = cand_row_max
                    new_global_max = max(row_maxes)
                    row_maxes[ri] = old_row_max
                    
                    delta = new_global_max - current_val
                    
                    accept = False
                    if delta < 0:
                        accept = True
                    elif delta == 0:
                        if random.random() < 0.3:
                            accept = True
                    else:
                        if T > T_min:
                            prob = math.exp(-delta / max(T, 0.1))
                            if random.random() < prob:
                                accept = True
                    
                    if accept:
                        all_used -= row_diffs[ri]
                        all_used |= new_diffs
                        current_rows[ri] = cand_row
                        row_diffs[ri] = new_diffs
                        row_maxes[ri] = cand_row_max
                        current_val = max(row_maxes)
                        
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        moved = True
                        break
                    continue
                
                # Regular single-element move
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                # Validate
                new_diffs = set()
                valid = True
                n = len(cand_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = cand_row[j] - cand_row[i]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check cross-row conflicts
                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute new global max
                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                # Acceptance criterion
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
            
            T = max(T_min, T * cool)
            
            # Restart if stuck
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 300.0
                no_improve = 0
        
        return best_rows, best_val

    def _row_reopt_bt(row_init, all_used_excl, k, max_scope, max_depth=3):
        """
        Re-optimize a single row using greedy with limited backtracking.
        Tries to minimize the max entry given constraints from other rows.
        """
        def _try_fill(row, pos, used_local, depth):
            if pos == k + 1:
                return row[:]
            
            start = row[pos - 1] + 1
            end = max_scope + 1
            
            # Try smaller values first to minimize max
            for v in range(start, end):
                new_diffs = {v - row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                
                row[pos] = v
                used_local_new = used_local | new_diffs
                
                if depth < max_depth:
                    result = _try_fill(row, pos + 1, used_local_new, depth + 1)
                    if result is not None:
                        return result
                else:
                    # At max depth, just greedily fill the rest
                    rest_row = row[:]
                    rest_used = used_local_new
                    for p in range(pos + 1, k + 1):
                        found = False
                        for v2 in range(rest_row[p-1] + 1, max_scope + 1):
                            nd2 = {v2 - rest_row[i] for i in range(p)}
                            if len(nd2) != p:
                                continue
                            if any(d in all_used_excl for d in nd2):
                                continue
                            if any(d in rest_used for d in nd2):
                                continue
                            rest_row.append(v2)
                            rest_used |= nd2
                            found = True
                            break
                        if not found:
                            return None
                    return rest_row
            
            return None
        
        row = [0] + [0] * k
        result = _try_fill(row, 1, set(), 0)
        return result

    def _row_reopt(row, all_used_excl, k, max_scope=None):
        """
        Re-optimize a single row given the differences used by all other rows.
        Uses greedy with limited backtracking to find a valid row with minimal max entry.
        Returns the optimized row or None if no valid row found within max_scope.
        """
        if max_scope is None:
            max_scope = 200
        
        # First try pure greedy (fast)
        row_greedy = [0]
        used_local = set()
        for pos in range(1, k + 1):
            found = False
            start = row_greedy[pos - 1] + 1
            end = max_scope + 1
            
            for v in range(start, end):
                new_diffs = {v - row_greedy[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                
                row_greedy.append(v)
                used_local |= new_diffs
                found = True
                break
            
            if not found:
                return None
        
        # Try backtracking to see if we can do better
        bt_result = _row_reopt_bt(row_greedy, all_used_excl, k, max_scope, max_depth=2)
        if bt_result is not None:
            return bt_result
        
        return row_greedy

    def row_reopt_polish(rows, n_rows, k, deadline):
        """
        Polish by re-optimizing each row against the others.
        Iterates until no improvement or time runs out.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(n_rows):
                # Compute all_used excluding this row
                all_used_excl = set()
                for rj in range(n_rows):
                    if rj != ri:
                        all_used_excl |= row_diffs[rj]
                
                # Try to re-optimize this row with tighter scope
                new_row = _row_reopt(current_rows[ri], all_used_excl, k, max_scope=current_val - 1)
                
                if new_row is not None:
                    new_row_max = max(new_row)
                    old_row_max = max(current_rows[ri])
                    
                    if new_row_max < old_row_max:
                        # Update
                        all_used -= row_diffs[ri]
                        new_diffs = _row_diffs(new_row)
                        all_used |= new_diffs
                        current_rows[ri] = new_row
                        row_diffs[ri] = new_diffs
                        current_val = _global_max(current_rows)
                        improved = True
                        # Try again with even tighter scope
                        all_used_excl2 = set()
                        for rj in range(n_rows):
                            if rj != ri:
                                all_used_excl2 |= row_diffs[rj]
                        new_row2 = _row_reopt(current_rows[ri], all_used_excl2, k, max_scope=current_val - 1)
                        if new_row2 is not None and max(new_row2) < max(current_rows[ri]):
                            all_used -= row_diffs[ri]
                            new_diffs2 = _row_diffs(new_row2)
                            all_used |= new_diffs2
                            current_rows[ri] = new_row2
                            row_diffs[ri] = new_diffs2
                            current_val = _global_max(current_rows)
        
        return current_rows, current_val

    # Run multiple SA trials with different seeds
    for trial in range(12):
        if time.time() > deadline - 8:
            break
        
        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        # Give each trial a fair time budget
        trial_deadline = min(deadline - 8, start_time + 145.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows
    
    # Final polish with remaining time
    if time.time() < deadline - 3:
        polish_deadline = min(deadline, start_time + 145.0)
        
        # First, do SA polish
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows
        
        # Then, do row re-optimization polish
        if time.time() < deadline - 1:
            reopt_rows, reopt_scope = row_reopt_polish(
                [r[:] for r in best_solution], n_rows, k, deadline - 1
            )
            if reopt_scope < best_scope:
                best_scope = reopt_scope
                best_solution = reopt_rows
        
        # Final tightening: try to reduce scope by 1 repeatedly
        if time.time() < deadline - 0.5:
            current = [r[:] for r in best_solution]
            cur_scope = _global_max(current)
            # Try reducing each row's max by 1
            for _ in range(5):
                if time.time() > deadline - 0.5:
                    break
                progress = False
                all_used = _all_used(current)
                row_diffs = [_row_diffs(r) for r in current]
                for ri in range(n_rows):
                    if time.time() > deadline - 0.5:
                        break
                    all_used_excl = set()
                    for rj in range(n_rows):
                        if rj != ri:
                            all_used_excl |= row_diffs[rj]
                    target = max(current[ri]) - 1
                    if target <= 0:
                        continue
                    new_row = _row_reopt(current[ri], all_used_excl, k, max_scope=target)
                    if new_row is not None and max(new_row) <= target:
                        all_used -= row_diffs[ri]
                        new_diffs = _row_diffs(new_row)
                        all_used |= new_diffs
                        current[ri] = new_row
                        row_diffs[ri] = new_diffs
                        progress = True
                if not progress:
                    break
            final_scope = _global_max(current)
            if final_scope < best_scope:
                best_scope = final_scope
                best_solution = current

    return best_solution
# EVOLVE-BLOCK-END
