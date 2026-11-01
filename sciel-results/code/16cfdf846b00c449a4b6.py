import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: exact-cover (DLX) for small scopes, and
    randomized local search for larger scopes.
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

    # ---- DLX exact cover for small scopes ----
    # Each column = difference value 1..M that must be covered exactly once
    # Each row in DLX = a valid (k+1)-term Sidon set (subset of {0..M} starting at 0)
    # We need to pick n_rows such that their difference sets partition {1..M}
    # This only works when n_rows * k*(k+1)/2 == M exactly

    def dlx_exact_cover(M, n_rows, k):
        """
        Exact cover via DLX: find n_rows Sidon sets (each of size k+1, starting at 0)
        whose difference sets partition {1..M}.
        Returns list of rows or None.
        """
        if n_rows * k * (k + 1) // 2 != M:
            return None  # exact cover only when total diffs == M

        # Generate all valid Sidon sets of size k+1 from {0, 1, ..., M} starting at 0
        # A Sidon set has all pairwise differences distinct
        # This is expensive for large M, so we limit
        if M > 200:
            return None

        # Generate candidate rows (Sidon sets)
        # Each candidate: sorted tuple of k+1 values starting at 0
        # Its difference set: all pairwise differences
        candidates = []
        
        # Enumerate all k-element subsets of {1, ..., M} (since 0 is always included)
        # For k+1 elements including 0, we choose k from {1..M}
        from itertools import combinations
        
        if k > 6:  # too many combinations
            return None
        
        max_val = M
        # For efficiency, enumerate combinations of k values from 1..max_val
        # But this is C(M, k) which can be huge
        # For M=105, k=5: C(105,5) ≈ 96 million - too many
        # We need a smarter enumeration
        
        # Alternative: enumerate Sidon sets directly by building them
        # Start with [0], then greedily add elements while checking Sidon property
        # But we need ALL Sidon sets, not just one
        
        # For small M and k, we can enumerate
        # For (7,5) with M=105: C(105,5) is too large
        # So DLX exact cover is only practical for very small instances
        
        if M <= 50 and k <= 4:
            for combo in combinations(range(1, max_val + 1), k):
                row = (0,) + combo
                # Check Sidon property
                diffs = set()
                valid = True
                for i in range(k + 1):
                    for j in range(i + 1, k + 1):
                        d = row[j] - row[i]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                if valid:
                    candidates.append((row, diffs))
        
        if not candidates:
            return None

        # DLX setup
        # Columns: 1..M (each difference value)
        # Each candidate row covers its difference set
        # We need to select exactly n_rows candidates such that their diff sets partition {1..M}
        
        # Build DLX structure
        # For efficiency with Python, use a simplified backtracking with constraint propagation
        
        # Map difference -> list of candidate indices that cover it
        diff_to_candidates = {d: [] for d in range(1, M + 1)}
        for idx, (row, diffs) in enumerate(candidates):
            for d in diffs:
                if 1 <= d <= M:
                    diff_to_candidates[d].append(idx)
        
        # DLX backtracking
        selected = []
        covered = set()
        
        def dlx_solve(remaining_rows, deadline_check):
            if time.time() > deadline_check:
                return None
            
            if len(selected) == n_rows:
                if len(covered) == M and covered == set(range(1, M + 1)):
                    return list(selected)
                return None
            
            if remaining_rows <= 0:
                return None
            
            # Find the uncovered difference with fewest candidate options
            uncovered = [d for d in range(1, M + 1) if d not in covered]
            if not uncovered:
                if len(selected) == n_rows:
                    return list(selected)
                return None
            
            # Choose column with minimum remaining candidates
            best_diff = min(uncovered, key=lambda d: len([c for c in diff_to_candidates[d] 
                                                        if c not in selected and 
                                                        not (diff_to_candidates[d] and 
                                                            any(d2 in covered for d2 in candidates[c][1] if d2 != d))]))
            
            # Simpler: just pick first uncovered with fewest valid candidates
            valid_candidates = []
            for d in uncovered:
                vc = []
                for c in diff_to_candidates[d]:
                    if c in selected:
                        continue
                    # Check if candidate c's diffs are all either in covered or will be added
                    cand_diffs = candidates[c][1]
                    conflict = False
                    for dd in cand_diffs:
                        if dd in covered and dd != d:
                            # Already covered by another selected row
                            conflict = True
                            break
                    if not conflict:
                        vc.append(c)
                if vc:
                    valid_candidates.append((d, vc))
            
            if not valid_candidates:
                return None
            
            # Pick the difference with fewest valid candidates
            valid_candidates.sort(key=lambda x: len(x[1]))
            best_diff, best_cands = valid_candidates[0]
            
            for c in best_cands:
                cand_diffs = candidates[c][1]
                # Check no conflicts
                conflict = False
                for dd in cand_diffs:
                    if dd in covered:
                        conflict = True
                        break
                if conflict:
                    continue
                
                selected.append(c)
                covered_new = covered | cand_diffs
                
                result = dlx_solve(remaining_rows - 1, deadline_check)
                if result is not None:
                    return result
                
                selected.pop()
                covered -= cand_diffs
            
            return None
        
        result = dlx_solve(n_rows, deadline)
        if result is not None:
            return [list(candidates[i][0]) for i in result]
        
        return None

    # Try DLX for small scopes where exact cover is possible
    for target_scope in range(lb, lb + 15):
        if time.time() > deadline:
            break
        
        # Exact cover only works when total_diffs == target_scope
        if n_rows * k * (k + 1) // 2 == target_scope:
            solution = dlx_exact_cover(target_scope, n_rows, k)
            if solution is not None and _validate(solution, n_rows, k):
                actual_scope = max(max(row) for row in solution)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_solution = solution
                break

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
        """SA-based local search to minimize the maximum entry."""
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
        
        while time.time() < deadline and no_improve < max_no_improve:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates
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
            
            random.shuffle(candidates)
            
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

    # Run multiple SA trials with different seeds
    for trial in range(8):
        if time.time() > deadline - 5:
            break
        
        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, deadline - 5)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows
    
    # Final polish
    if time.time() < deadline - 3:
        polish_deadline = min(deadline, start_time + 143.0)
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows

    return best_solution
# EVOLVE-BLOCK-END
