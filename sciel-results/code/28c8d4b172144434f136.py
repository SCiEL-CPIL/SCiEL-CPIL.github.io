"""Difference triangle set (4,7): Optimized CSP solver using bitsets for difference tracking."""
import random
import sys
import time
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a bitset-based backtracking solver with strong pruning to minimize the maximum element (scope).
    """
    # We don't use a simple greedy baseline for the initial scope anymore.
    # We will search downwards from a known good upper bound.
    # Best known for (4,7) is 134. We start from 140 to be safe.
    initial_upper_bound = 140
    lower_bound = n_rows * (k + 1) * k // 2
    if lower_bound < 112:
        lower_bound = 112
    
    def solve_scope(scope, seed, max_nodes=1000000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        
        def build_row(idx):
            nonlocal used_bits, nodes
            if nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            
            if idx == n_rows:
                return True
            
            row = [0]
            row_diff_bits = 0
            
            # Symmetry breaking: 
            # We assume rows are sorted by their second element (index 1).
            # This is a valid symmetry breaking for DTS.
            if idx == 0:
                min_start = 1
            else:
                min_start = rows[idx - 1][1] + 1
            
            def rec(row, row_diff_bits, min_x):
                nonlocal used_bits, nodes
                if nodes[0] > max_nodes:
                    return False
                nodes[0] += 1
                
                if len(row) == k + 1:
                    # Check conflict
                    if used_bits & row_diff_bits:
                        return False
                    
                    used_bits |= row_diff_bits
                    rows.append(row)
                    if build_row(idx + 1):
                        return True
                    else:
                        rows.pop()
                        used_bits ^= row_diff_bits
                        return False

                remaining = (k + 1) - len(row)
                max_x = scope - (remaining - 1)
                
                if max_x < min_x:
                    return False

                blocked = used_bits | row_diff_bits
                
                # Pruning: Check if enough differences are available
                # Total differences needed for a row of size m is m*(m-1)/2
                # Current row size is len(row). Current diffs formed: len(row)*(len(row)-1)/2
                # Remaining diffs needed: (k+1)*k/2 - len(row)*(len(row)-1)/2
                current_diffs_count = len(row) * (len(row) - 1) // 2
                total_diffs_needed = (k + 1) * k // 2
                remaining_diffs_needed = total_diffs_needed - current_diffs_count
                
                # Count available bits in [1, scope] not in blocked
                # This is expensive to do every step. 
                # Approximation: if scope - blocked.bit_count() < remaining_diffs_needed, fail.
                # Note: blocked.bit_count() counts bits in the whole integer, but we only care up to scope.
                # Since scope is small (< 200), we can just count.
                if scope - blocked.bit_count() < remaining_diffs_needed:
                    return False

                candidates = []
                
                # Generate candidates
                # Iterate in increasing order to prefer compact rows
                for x in range(min_x, max_x + 1):
                    ok = True
                    for a in row:
                        d = x - a
                        if (blocked >> d) & 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)
                
                if not candidates:
                    return False
                
                # Heuristic: 
                # Prefer smaller values (compactness) which tends to work well for DTS.
                # If we want diversity, we can randomize, but deterministic is often faster for finding *a* solution.
                # We'll stick to the first candidate (smallest x) for determinism and speed, 
                # unless we are doing randomized restarts.
                # Since we call solve_scope with different seeds, we can use seed to vary strategy.
                if seed % 2 == 0:
                    x = candidates[0]
                else:
                    # Random selection from the first 10 candidates
                    subset = candidates[:min(10, len(candidates))]
                    x = rng.choice(subset)
                
                new_diffs = 0
                for a in row:
                    d = x - a
                    new_diffs |= (1 << d)
                
                if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                    return True
                return False
            
            return rec(row, row_diff_bits, min_start)
        
        success = build_row(0)
        if success:
            return rows
        return None

    t0 = time.time()
    time_limit = 115
    best_found = None
    best_scope_found = initial_upper_bound + 1 # Initialize to a value higher than max search
    
    # Search downwards from initial_upper_bound to lower_bound
    # We prioritize smaller scopes (better solutions) by going down.
    # However, finding *any* solution is better than none.
    # Since 134 is known, we start there. If we fail, we go down.
    # If we find a solution at 134, we try 133, etc.
    
    scopes_to_check = list(range(initial_upper_bound, lower_bound - 1, -1))
    
    for scope in scopes_to_check:
        if time.time() - t0 > time_limit:
            break
        
        # Time slice for this scope
        scope_start_time = time.time()
        # Allow up to 3 seconds per scope to allow more scopes to be tried
        remaining_total = time_limit - (time.time() - t0)
        slice_time = min(3.0, remaining_total)
        
        if slice_time < 0.5:
            break
            
        found = None
        # Try seeds until time slice is up
        seed = 0
        while time.time() - scope_start_time < slice_time:
            # Use a generous node budget
            budget = 10000000 
            
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
            seed += 1
            
        if found:
            best_found = found
            best_scope_found = scope
            # We can continue to search for even smaller scopes
        else:
            # No solution found for this scope within the time slice.
            pass

    # If no solution was found in the search, return a valid greedy solution as fallback
    # We need to ensure we return *something* valid.
    if best_found is None:
        # Re-run greedy to ensure a valid output
        def greedy():
            used = 0
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            d = x - a
                            used |= (1 << d)
                rows.append(row)
            return rows
        best_found = greedy()

    return best_found
