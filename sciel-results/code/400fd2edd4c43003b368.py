import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Helper functions
    def max_entry(rows):
        return max(max(row) for row in rows)

    def all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs

    def compute_cost(rows):
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        violations = sum(c - 1 for c in counts.values() if c > 1)
        m = max_entry(rows)
        # We want to minimize max_entry primarily, but must be valid.
        # If invalid, we penalize heavily so local search tries to fix validity first.
        # However, for scope reduction, we only care about valid solutions.
        # For local search, we want to minimize max_entry among valid solutions,
        # or minimize violations if invalid.
        # A combined cost: violations * 10000 + m
        return violations * 10000 + m

    def is_valid(rows):
        diffs = all_diffs(rows)
        return len(diffs) == len(set(diffs))

    # Phase 1: Greedy Initialization
    def greedy_init():
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

    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    
    # Phase 2: Tabu Search for Local Optimization
    def tabu_search(initial_rows, time_limit):
        current_rows = [r[:] for r in initial_rows]
        best_rows_local = [r[:] for r in initial_rows]
        best_cost = compute_cost(best_rows_local)
        current_cost = best_cost
        
        indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
        tabu_list = []  # (row, pos, old_val, expiration_time)
        tabu_tenure = 100  # Number of iterations to keep in tabu
        
        def get_neighbors(rows):
            neighbors = []
            for r, i in indices:
                old_val = rows[r][i]
                lower = rows[r][i - 1] + 1
                upper = rows[r][i + 1] - 1 if i < k else rows[r][k] + 20
                
                # Try a few random values in the range
                # To be efficient, we don't generate ALL neighbors, just a sample
                for _ in range(5):
                    if upper <= lower:
                        new_val = lower
                    else:
                        new_val = random.randint(lower, upper)
                    if new_val == old_val:
                        continue
                    new_rows = [row[:] for row in rows]
                    new_rows[r][i] = new_val
                    # Check strict increase
                    if any(new_rows[r][j] <= new_rows[r][j-1] for j in range(1, len(new_rows[r]))):
                        continue
                    cost = compute_cost(new_rows)
                    neighbors.append((cost, new_rows, r, i, old_val))
            return neighbors

        iter_count = 0
        while time.time() < start_time + time_limit:
            neighbors = get_neighbors(current_rows)
            if not neighbors:
                break
            
            # Filter out tabu moves
            non_tabu = []
            for cost, n_rows, r, i, old_val in neighbors:
                is_tabu = False
                for (tr, ti, tv, exp) in tabu_list:
                    if tr == r and ti == i and tv == old_val and exp > iter_count:
                        is_tabu = True
                        break
                if not is_tabu:
                    non_tabu.append((cost, n_rows, r, i, old_val))
            
            # Aspiration: if a tabu move is better than best so far, allow it
            if not non_tabu:
                # Pick the best neighbor overall
                non_tabu = neighbors
            
            # Select best neighbor
            non_tabu.sort(key=lambda x: x[0])
            best_cost_n, best_rows_n, r, i, old_val = non_tabu[0]
            
            # Update current
            current_rows = best_rows_n
            current_cost = best_cost_n
            
            # Update tabu
            tabu_list.append((r, i, old_val, iter_count + tabu_tenure))
            if len(tabu_list) > 500:
                tabu_list.pop(0)
            
            # Update best
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows_local = [row[:] for row in current_rows]
            
            iter_count += 1
            if iter_count % 100 == 0:
                # Clean up expired tabu entries
                tabu_list = [(tr, ti, tv, exp) for (tr, ti, tv, exp) in tabu_list if exp > iter_count]
                
        return best_rows_local

    # Run Tabu for 20 seconds
    if time.time() < deadline - 5:
        best_rows = tabu_search(best_rows, 20.0)
        best_max = max_entry(best_rows)

    # Phase 3: Scope Reduction via Backtracking
    def try_backtracking(scope, timeout, guide_rows=None):
        """
        Try to find a valid configuration with max entry <= scope.
        Uses guide_rows to order candidates.
        """
        start_bt = time.time()
        used_diffs = 0
        rows = [[0] for _ in range(n_rows)]
        
        # Pre-compute guide values if available
        guides = [r[:] for r in guide_rows] if guide_rows else None
        
        def dfs(row_idx, pos_idx):
            if time.time() - start_bt > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos_idx == k + 1:
                return dfs(row_idx + 1, 1)
            
            row = rows[row_idx]
            prev_val = row[pos_idx - 1]
            
            # Generate candidates
            candidates = []
            # If we have a guide, try the guide value first if it fits
            if guides:
                guide_val = guides[row_idx][pos_idx]
                if prev_val < guide_val <= scope:
                    candidates.append(guide_val)
            
            # Add other candidates: neighbors of guide, then random/small values
            # We want to try values that are likely to work.
            # Since we are reducing scope, values close to the previous best solution are good candidates.
            
            # Simple heuristic: try values from prev_val+1 to scope, but ordered by proximity to guide
            if guides:
                guide_val = guides[row_idx][pos_idx]
                # Create a list of values sorted by distance to guide_val
                # But we can't generate all values if scope is large.
                # Let's just try a window around the guide value and some randoms.
                if guide_val > prev_val:
                    # Try values near guide_val
                    for d in range(0, 5):
                        v1 = guide_val - d
                        v2 = guide_val + d
                        if prev_val < v1 <= scope and v1 not in candidates:
                            candidates.append(v1)
                        if prev_val < v2 <= scope and v2 not in candidates:
                            candidates.append(v2)
            
            # Add a few random candidates
            for _ in range(5):
                v = random.randint(prev_val + 1, scope)
                if v not in candidates:
                    candidates.append(v)
            
            # Also try the smallest possible values to ensure we don't miss tight fits
            for v in range(prev_val + 1, min(prev_val + 5, scope + 1)):
                if v not in candidates:
                    candidates.append(v)
            
            # Shuffle to break symmetry for non-guide candidates
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() - start_bt > timeout:
                    return False
                
                new_diffs = []
                conflict = False
                new_used = used_diffs
                for i in range(pos_idx):
                    d = x - row[i]
                    if (used_diffs >> d) & 1:
                        conflict = True
                        break
                    new_used |= (1 << d)
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                row.append(x)
                
                if dfs(row_idx, pos_idx + 1):
                    return True
                
                row.pop()
            
            return False
        
        if dfs(0, 1):
            return [r[:] for r in rows]
        
        return None

    # Scope Reduction Loop
    while time.time() < deadline - 5:
        current_max = max_entry(best_rows)
        found_better = False
        
        # Try to reduce scope by 1, then 2, etc.
        # We limit the number of attempts to avoid wasting time on impossible scopes
        for scope in range(current_max - 1, max(100, current_max - 15), -1):
            if time.time() > deadline - 5:
                break
            
            result = try_backtracking(scope, timeout=2.0, guide_rows=best_rows)
            if result is not None:
                best_rows = result
                best_max = scope
                found_better = True
                break
        
        if not found_better:
            # If we couldn't reduce scope, run a short Tabu search to try to find a different local minimum
            # that might allow a lower scope.
            if time.time() < deadline - 10:
                # Perturb the current best slightly
                perturbed = [r[:] for r in best_rows]
                # Randomly change a few values
                for _ in range(5):
                    r = random.randint(0, n_rows - 1)
                    i = random.randint(1, k)
                    old_val = perturbed[r][i]
                    lower = perturbed[r][i-1] + 1
                    upper = perturbed[r][i+1] - 1 if i < k else perturbed[r][k] + 10
                    if upper > lower:
                        new_val = random.randint(lower, upper)
                        # Ensure validity of the row structure (strictly increasing)
                        # This might break validity, but Tabu will fix it
                        perturbed[r][i] = new_val
                
                best_rows = tabu_search(perturbed, 5.0)
                best_max = max_entry(best_rows)
            else:
                break

    return best_rows
