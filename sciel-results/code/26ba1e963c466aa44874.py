import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses iterated local search with row re-optimization and SA-based refinement.
    """
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s limit

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

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _global_max(rows):
        return max(max(row) for row in rows)

    def greedy_construct(n_rows, k, seed=None):
        """Greedy construction with optional randomization."""
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

    def row_reopt_greedy(row, all_used_excl, k, max_scope=None):
        """
        Re-optimize a single row given differences used by other rows.
        Greedy: try each position from left to right, pick smallest valid value.
        """
        if max_scope is None:
            max_scope = 250
        
        new_row = [0]
        used_local = set()
        
        for pos in range(1, k + 1):
            found = False
            start = new_row[pos - 1] + 1
            end = max_scope + 1
            
            for v in range(start, end):
                new_diffs = {v - new_row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                
                new_row.append(v)
                used_local |= new_diffs
                found = True
                break
            
            if not found:
                return None
        
        return new_row

    def row_reopt_backtrack(row, all_used_excl, k, max_scope=None, depth_limit=2):
        """
        Re-optimize a row with limited backtracking for better results.
        """
        if max_scope is None:
            max_scope = 250
        
        def _backtrack(pos, current_row, used_local):
            if pos == k + 1:
                return current_row[:]
            
            if pos > depth_limit and random.random() < 0.3:
                # Limit deeper backtracking
                return None
            
            start = current_row[pos - 1] + 1
            end = min(max_scope + 1, current_row[pos - 1] + 60)
            
            for v in range(start, end):
                new_diffs = {v - current_row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                
                current_row[pos] = v
                used_local_new = used_local | new_diffs
                
                result = _backtrack(pos + 1, current_row, used_local_new)
                if result is not None:
                    return result
            
            return None
        
        # Try multiple starting points
        best_result = None
        best_max = max_scope + 1
        
        for attempt in range(3):
            current_row = [0] + [0] * k
            # Initialize with greedy
            used_local = set()
            valid = True
            for pos in range(1, k + 1):
                found = False
                for v in range(current_row[pos - 1] + 1, max_scope + 1):
                    new_diffs = {v - current_row[i] for i in range(pos)}
                    if len(new_diffs) != pos:
                        continue
                    if any(d in all_used_excl for d in new_diffs):
                        continue
                    if any(d in used_local for d in new_diffs):
                        continue
                    current_row[pos] = v
                    used_local |= new_diffs
                    found = True
                    break
                if not found:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Try backtracking from this starting point
            result = _backtrack(1, current_row, set())
            if result is not None:
                rmax = max(result)
                if rmax < best_max:
                    best_max = rmax
                    best_result = result
        
        if best_result is not None:
            return best_result
        
        # Fallback to pure greedy
        return row_reopt_greedy(row, all_used_excl, k, max_scope)

    def local_search(rows, n_rows, k, start_time, deadline, max_iters=50000):
        """
        Local search: iteratively re-optimize rows to reduce global max.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        all_used = set()
        for rd in row_diffs:
            all_used |= rd
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        no_improve = 0
        max_no_improve = 2000
        
        iter_count = 0
        while time.time() < deadline and no_improve < max_no_improve and iter_count < max_iters:
            iter_count += 1
            
            # Pick a row to re-optimize: bias toward rows with high max
            row_maxes = [max(r) for r in current_rows]
            max_row_val = max(row_maxes)
            
            # With probability 0.7, pick a row that has the current max
            if random.random() < 0.7:
                candidates = [i for i in range(n_rows) if row_maxes[i] == max_row_val]
                ri = random.choice(candidates)
            else:
                ri = random.randrange(n_rows)
            
            # Compute all_used excluding this row
            all_used_excl = set()
            for rj in range(n_rows):
                if rj != ri:
                    all_used_excl |= row_diffs[rj]
            
            # Try to re-optimize this row
            old_row_max = max(current_rows[ri])
            new_row = row_reopt_backtrack(current_rows[ri], all_used_excl, k, max_scope=current_val)
            
            if new_row is not None:
                new_row_max = max(new_row)
                
                if new_row_max < old_row_max:
                    # Update
                    all_used -= row_diffs[ri]
                    new_diffs = _row_diffs(new_row)
                    all_used |= new_diffs
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    current_val = _global_max(current_rows)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        return best_rows, best_val

    def sa_refine(rows, n_rows, k, start_time, deadline):
        """
        Simulated annealing refinement for fine-tuning.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        row_diffs = [_row_diffs(r) for r in current_rows]
        all_used = set()
        for rd in row_diffs:
            all_used |= rd
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 200.0
        T_min = 0.1
        cool = 0.99999
        no_improve = 0
        max_no_improve = 15000
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Pick row and position
            if random.random() < 0.6:
                max_r = [i for i in range(n_rows) if row_maxes[i] == current_val]
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            
            # Decreasing moves (preferred)
            for d in range(1, min(50, old_val - min_allowed) + 1):
                candidates.append(old_val - d)
            
            # Small increasing moves
            for d in range(1, min(15, max_allowed - old_val) + 1):
                candidates.append(old_val + d)
            
            # Random jump
            if random.random() < 0.2:
                lo = max(min_allowed, old_val - 30)
                hi = min(max_allowed, old_val + 30)
                if lo <= hi:
                    candidates.append(random.randint(lo, hi))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            # Sort: decreasing first, then by value
            candidates.sort(key=lambda v: (v > old_val, v))
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                # Validate row
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
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Compute new global max
                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                # Acceptance
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.1
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        accept = random.random() < prob
                
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
        
        return best_rows, best_val

    def perturb(rows, n_rows, k, strength=3):
        """Perturb a solution by making random valid moves."""
        perturbed = [r[:] for r in rows]
        row_diffs = [_row_diffs(r) for r in perturbed]
        all_used = set()
        for rd in row_diffs:
            all_used |= rd
        
        for _ in range(strength):
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = perturbed[ri][pi]
            min_allowed = perturbed[ri][pi - 1] + 1
            max_allowed = (perturbed[ri][pi + 1] - 1) if pi < k else old_val + 20
            
            candidates = []
            for d in range(1, min(10, max_allowed - old_val) + 1):
                candidates.append(old_val + d)
            for d in range(1, min(10, old_val - min_allowed) + 1):
                candidates.append(old_val - d)
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            
            for new_val in candidates:
                if new_val <= perturbed[ri][pi - 1]:
                    continue
                if pi < k and new_val >= perturbed[ri][pi + 1]:
                    continue
                
                cand_row = perturbed[ri][:]
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
                
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Accept perturbation
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                perturbed[ri] = cand_row
                row_diffs[ri] = new_diffs
                break
        
        return perturbed

    # ---- Main search loop ----
    
    # Initial solutions
    best_solution = None
    best_scope = float('inf')
    
    # Try a few greedy inits with different seeds
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        init_rows = greedy_construct(n_rows, k, seed=seed * 31 + 7)
        init_scope = _global_max(init_rows)
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
    
    # ILS loop
    current_solution = [r[:] for r in best_solution]
    current_scope = best_scope
    
    il_iter = 0
    max_il_iters = 50
    
    while time.time() < deadline - 20 and il_iter < max_il_iters:
        il_iter += 1
        
        # Local search
        ls_rows, ls_scope = local_search(
            [r[:] for r in current_solution], n_rows, k, start_time, deadline - 15
        )
        
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_solution = ls_rows
        
        # SA refinement
        if time.time() < deadline - 10:
            sa_rows, sa_scope = sa_refine(
                [r[:] for r in best_solution], n_rows, k, start_time, deadline - 5
            )
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_solution = sa_rows
        
        # Perturb for next iteration
        if time.time() < deadline - 25:
            strength = 2 + random.randint(0, 3)
            current_solution = perturb(
                [r[:] for r in best_solution], n_rows, k, strength=strength
            )
            current_scope = _global_max(current_solution)
            
            # Accept perturbation if not too bad
            if current_scope > best_scope + 15:
                current_solution = [r[:] for r in best_solution]
                current_scope = best_scope
    
    # Final polish with remaining time
    if time.time() < deadline - 3:
        polish_deadline = deadline - 1
        ls_rows, ls_scope = local_search(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_solution = ls_rows
        
        if time.time() < deadline - 2:
            sa_rows, sa_scope = sa_refine(
                [r[:] for r in best_solution], n_rows, k, start_time, deadline - 1
            )
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_solution = sa_rows
    
    # Validate final solution
    if not _validate(best_solution, n_rows, k):
        # Fallback to greedy
        best_solution = greedy_construct(n_rows, k)
    
    return best_solution
# EVOLVE-BLOCK-END
