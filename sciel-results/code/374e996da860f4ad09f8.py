import time
import random
import math
import sys
from collections import Counter

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, then randomized local search
    with simulated annealing to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # Lower bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed  # scope must be at least this

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

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # ---- Randomized Local Search with Simulated Annealing ----
    # This is the main workhorse for finding good solutions

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

    def _count_conflicts(rows, n_rows, k):
        """Count the number of duplicate differences across all rows."""
        all_diffs = []
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    all_diffs.append(row[j] - row[i])
        cnt = Counter(all_diffs)
        conflicts = 0
        for d, c in cnt.items():
            if c > 1:
                conflicts += c - 1
        return conflicts

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
        """SA-based local search to minimize the maximum entry, with conflict-aware moves."""
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
        max_no_improve = 30000
        restarts = 0
        max_restarts = 20
        
        # Find which row(s) have the max value
        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]
        
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
            max_decrease = min(80, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            # Increasing candidates (exploration)
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
            max_increase = min(30, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            
            # Random jumps
            if random.random() < 0.3:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            # Sort candidates: decreasing moves first (lower values preferred)
            candidates.sort(key=lambda v: (v > old_val, v))
            
            moved = False
            for new_val in candidates:
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

    def conflict_aware_sa(rows, n_rows, k, start_time, deadline):
        """SA that also considers conflict count as a secondary objective to escape local minima."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        current_conflicts = _count_conflicts(current_rows, n_rows, k)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        best_conflicts = current_conflicts
        
        T = 200.0
        T_min = 0.1
        cool = 0.99995
        
        no_improve = 0
        max_no_improve = 50000
        restarts = 0
        max_restarts = 15
        
        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]
        
        while time.time() < deadline and no_improve < max_no_improve:
            max_r = _max_rows()
            if max_r and random.random() < 0.5:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(60, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
            max_increase = min(20, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            
            if random.random() < 0.2:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            candidates.sort(key=lambda v: (v > old_val, v))
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
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
                
                delta_max = new_global_max - current_val
                
                # Compute new conflict count efficiently
                # Remove old diffs from consideration, add new ones
                old_diffs = row_diffs[ri]
                # Count how many of new_diffs conflict with other rows
                new_conflict_count = 0
                for d in new_diffs:
                    if d in all_used and d not in old_diffs:
                        new_conflict_count += 1
                # Subtract old conflicts that are resolved
                old_conflict_count = 0
                for d in old_diffs:
                    if d in all_used and d not in new_diffs:
                        old_conflict_count += 1
                
                delta_conflicts = new_conflict_count - old_conflict_count
                
                # Combined acceptance: prefer lower max, then lower conflicts
                if delta_max < 0:
                    accept = True
                elif delta_max == 0 and delta_conflicts < 0:
                    accept = True
                elif delta_max == 0 and delta_conflicts == 0:
                    if random.random() < 0.1:
                        accept = True
                elif delta_max > 0:
                    # Allow worse max if it reduces conflicts significantly
                    if delta_conflicts < -2 and T > T_min:
                        prob = math.exp(-(delta_max * 2 - delta_conflicts) / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                    elif T > T_min:
                        prob = math.exp(-delta_max / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    current_conflicts += delta_conflicts
                    
                    if current_val < best_val or (current_val == best_val and current_conflicts < best_conflicts):
                        best_val = current_val
                        best_conflicts = current_conflicts
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
            
            T = max(T_min, T * cool)
            
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                current_conflicts = best_conflicts
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 150.0
                no_improve = 0
        
        return best_rows, best_val

    def _row_reopt(row, all_used_excl, k, max_scope=None):
        """
        Re-optimize a single row given the differences used by all other rows.
        Uses greedy with backtracking to find a valid row with minimal max entry.
        Returns the optimized row or None if no valid row found within max_scope.
        """
        if max_scope is None:
            max_scope = 200
        
        # Try to find a row using greedy with limited backtracking
        def _try_greedy_with_backtrack(row, pos, used_local, all_used_excl, max_scope):
            if pos == k + 1:
                return row[:]
            
            start = row[pos - 1] + 1
            end = min(max_scope, 250) + 1
            
            for v in range(start, end):
                new_diffs = {v - row[i] for i in range(pos)}
                # Check for conflicts within this row
                if len(new_diffs) != pos:
                    continue
                # Check conflicts with other rows
                if any(d in all_used_excl for d in new_diffs):
                    continue
                # Check conflicts with previously chosen diffs in this row
                if any(d in used_local for d in new_diffs):
                    continue
                
                row[pos] = v
                used_local_new = used_local | new_diffs
                
                result = _try_greedy_with_backtrack(row, pos + 1, used_local_new, all_used_excl, max_scope)
                if result is not None:
                    return result
            
            return None
        
        # Start with the current row as initial guess, but try to minimize
        # Use a simpler approach: try values in order, prefer smaller
        row = [0]
        used_local = set()
        
        for pos in range(1, k + 1):
            found = False
            start = row[pos - 1] + 1
            end = min(max_scope, 250) + 1
            
            for v in range(start, end):
                new_diffs = {v - row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                
                row.append(v)
                used_local |= new_diffs
                found = True
                break
            
            if not found:
                return None
        
        return row

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
                
                # Try to re-optimize this row
                new_row = _row_reopt(current_rows[ri], all_used_excl, k, max_scope=current_val)
                
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
        
        return current_rows, current_val

    def final_polish(rows, n_rows, k, deadline):
        """Final high-precision polish: try small perturbations on each element."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(n_rows):
                for pi in range(1, k + 1):
                    old_val = current_rows[ri][pi]
                    # Try decreasing by 1, 2, 3
                    for delta in range(1, 4):
                        new_val = old_val - delta
                        if new_val <= current_rows[ri][pi - 1]:
                            continue
                        if pi < k and new_val >= current_rows[ri][pi + 1]:
                            continue
                        
                        cand_row = current_rows[ri][:]
                        cand_row[pi] = new_val
                        
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
                        
                        for d in new_diffs:
                            if d in all_used and d not in row_diffs[ri]:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        cand_row_max = max(cand_row)
                        if cand_row_max < row_maxes[ri]:
                            all_used -= row_diffs[ri]
                            all_used |= new_diffs
                            current_rows[ri] = cand_row
                            row_diffs[ri] = new_diffs
                            row_maxes[ri] = cand_row_max
                            current_val = max(row_maxes)
                            improved = True
                            break
        
        return current_rows, current_val

    # Run multiple SA trials with different seeds
    for trial in range(8):
        if time.time() > deadline - 10:
            break
        
        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        # Give each trial a fair time budget
        remaining = deadline - 10 - time.time()
        if remaining < 5:
            break
        trial_deadline = time.time() + min(remaining * 0.8, 20.0)
        
        # Alternate between standard SA and conflict-aware SA
        if trial % 2 == 0:
            sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)
        else:
            sa_rows, sa_scope = conflict_aware_sa(init_rows, n_rows, k, start_time, trial_deadline)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows
    
    # Final polish with remaining time
    if time.time() < deadline - 5:
        polish_deadline = min(deadline - 3, time.time() + 15.0)
        
        # First, do SA polish
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows
        
        # Then, do row re-optimization polish
        if time.time() < deadline - 2:
            reopt_deadline = min(deadline - 1, time.time() + 10.0)
            reopt_rows, reopt_scope = row_reopt_polish(
                [r[:] for r in best_solution], n_rows, k, reopt_deadline
            )
            if reopt_scope < best_scope:
                best_scope = reopt_scope
                best_solution = reopt_rows
        
        # Final fine-grained polish
        if time.time() < deadline - 0.5:
            final_deadline = min(deadline, time.time() + 5.0)
            final_rows, final_scope = final_polish(
                [r[:] for r in best_solution], n_rows, k, final_deadline
            )
            if final_scope < best_scope:
                best_scope = final_scope
                best_solution = final_rows

    return best_solution
# EVOLVE-BLOCK-END
