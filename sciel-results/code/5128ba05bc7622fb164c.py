import time
import math
import random
import sys

def construct(n_rows, k):
    """
    Construct difference triangle set (n_rows, k).
    Returns a list of n_rows rows, each of length k+1, starting at 0, strictly increasing.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 118  # 2s safety margin for 120s limit
    
    # Total distinct differences needed
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on scope M
    # Differences are in {1, ..., M}. We need total_diffs_needed distinct values.
    # So M >= total_diffs_needed.
    # But this is a weak bound. A better bound comes from the fact that each row uses
    # k+1 elements, so the max difference in a row is at most M.
    # Actually, the counting bound is M >= total_diffs_needed.
    # For (4,7): total_diffs = 4 * 28 = 112. So M >= 112.
    
    M_lower = total_diffs_needed
    
    # For the given task (4,7), best known is 134, counting bound is 112.
    # Let's use a tighter initial upper bound.
    # A simple greedy construction gives an upper bound.
    
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows
    
    # Start with greedy construction
    greedy_rows = greedy_construct()
    best_rows = [r[:] for r in greedy_rows]
    best_scope = max(max(r) for r in greedy_rows)
    
    # If n_rows is small and k is small, we might be able to do a more targeted search.
    # But let's use a robust local search approach.
    
    # State for Local Search
    current_rows = [r[:] for r in greedy_rows]
    current_scope = best_scope
    
    # We need to track used differences efficiently.
    # Since scope can be up to ~200-300 for (4,7), we can use a bit array or list.
    # But for generality, let's use a set or dict.
    # Actually, for speed, let's use a list if scope is bounded.
    
    # Let's estimate a reasonable upper bound for scope during search.
    # We'll allow scope up to 3 * best_scope to be safe, but try to reduce it.
    
    def compute_diff_counts(rows):
        """Return a dict mapping diff -> count."""
        counts = {}
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts
    
    def num_duplicates(counts):
        return sum(1 for v in counts.values() if v > 1)
    
    # For faster updates, let's use a different approach:
    # Store for each element its contribution to diffs, and maintain a global count array.
    # But the scope can change, so we need to handle that.
    
    # Alternative: Use a fixed-size array for diff counts, sized to max_possible_scope.
    # Let's estimate max_possible_scope = 4 * best_scope + 100 to be safe.
    
    max_possible = 4 * best_scope + 100
    diff_counts = [0] * (max_possible + 1)
    
    def rebuild_diff_counts(rows):
        """Rebuild diff_counts from scratch."""
        for i in range(len(diff_counts)):
            diff_counts[i] = 0
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d <= max_possible:
                        diff_counts[d] += 1
    
    rebuild_diff_counts(current_rows)
    
    def get_num_dupes():
        return sum(1 for i in range(1, len(diff_counts)) if diff_counts[i] > 1)
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    def calc_move_delta(ri, ei, new_val):
        """Calculate change in num_dupes if we set current_rows[ri][ei] = new_val."""
        row = current_rows[ri]
        old_val = row[ei]
        if old_val == new_val:
            return 0
        
        delta = 0
        n = len(row)
        
        # Remove old diffs
        for j in range(n):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= max_possible:
                c = diff_counts[d]
                if c == 2:
                    delta -= 1  # 2 -> 1, removes a duplicate
        
        # Add new diffs
        for j in range(n):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= max_possible:
                c = diff_counts[d]
                if c == 1:
                    delta += 1  # 1 -> 2, creates a duplicate
        
        return delta
    
    def apply_move(ri, ei, new_val):
        """Apply a move and update diff_counts."""
        row = current_rows[ri]
        old_val = row[ei]
        if old_val == new_val:
            return
        
        n = len(row)
        
        # Remove old diffs
        for j in range(n):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= max_possible:
                old_c = diff_counts[d]
                diff_counts[d] -= 1
        
        row[ei] = new_val
        
        # Add new diffs
        for j in range(n):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= max_possible:
                diff_counts[d] += 1
    
    def is_valid_scope_change(ri, ei, new_val):
        """Check if setting row[ei] = new_val maintains strict increase."""
        row = current_rows[ri]
        if ei > 0 and new_val <= row[ei - 1]:
            return False
        if ei < len(row) - 1 and new_val >= row[ei + 1]:
            return False
        return True
    
    # Simulated Annealing with iterative deepening on scope
    # Strategy:
    # 1. Start with current best scope
    # 2. Try to find a valid solution with scope <= current_scope
    # 3. If found, try to reduce scope
    # 4. Use SA to minimize conflicts, then scope
    
    best_scope_found = best_scope
    best_rows_found = [r[:] for r in best_rows]
    
    def local_search_sa(max_steps, target_scope=None):
        """Run simulated annealing local search."""
        nonlocal current_rows, diff_counts, best_scope_found, best_rows_found
        
        steps = 0
        T = 10.0
        alpha = 0.995
        min_T = 0.01
        
        # Initial state
        cur_dupes = get_num_dupes()
        cur_scope = get_scope()
        
        if target_scope is None:
            target_scope = cur_scope
        
        best_local_dupes = cur_dupes
        best_local_scope = cur_scope
        best_local_rows = [r[:] for r in current_rows]
        
        while steps < max_steps and time.time() < deadline:
            # Pick a random element to move
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)  # Don't move the first element (always 0)
            
            row = current_rows[ri]
            old_val = row[ei]
            
            # Determine valid range for new_val
            lower = row[ei - 1] + 1
            upper = row[ei + 1] - 1 if ei < k else target_scope
            
            if lower > upper:
                continue
            
            # Sample new value, biased towards smaller values
            # Use exponential distribution to favor smaller values
            if upper - lower > 50:
                # Bias towards lower
                bias = random.random() ** 2
                new_val = lower + int(bias * (upper - lower))
            else:
                new_val = random.randint(lower, upper)
            
            if new_val == old_val:
                continue
            
            if not is_valid_scope_change(ri, ei, new_val):
                continue
            
            delta_dupes = calc_move_delta(ri, ei, new_val)
            
            # Acceptance criterion
            if delta_dupes < 0:
                accepted = True
            elif delta_dupes == 0:
                # Neutral in conflicts, check scope change
                new_scope_est = max(cur_scope, new_val)
                if new_scope_est <= cur_scope:
                    accepted = True
                else:
                    # Small chance to accept scope increase
                    accepted = random.random() < math.exp(-1.0 / T)
            else:
                # Increase in conflicts
                # Accept with probability based on delta and temperature
                prob = math.exp(-delta_dupes * 2.0 / T)
                accepted = random.random() < prob
            
            if accepted:
                apply_move(ri, ei, new_val)
                cur_dupes += delta_dupes
                
                # Update local best
                if cur_dupes == 0:
                    cur_scope = get_scope()
                    if cur_scope < best_local_scope:
                        best_local_scope = cur_scope
                        best_local_rows = [r[:] for r in current_rows]
                        # Update global best
                        if cur_scope < best_scope_found:
                            best_scope_found = cur_scope
                            best_rows_found = [r[:] for r in current_rows]
                elif cur_dupes < best_local_dupes:
                    best_local_dupes = cur_dupes
                    best_local_scope = get_scope()
                    best_local_rows = [r[:] for r in current_rows]
            
            T *= alpha
            if T < min_T:
                T = min_T
            
            steps += 1
        
        # Restore best local solution
        current_rows = [r[:] for r in best_local_rows]
        rebuild_diff_counts(current_rows)
        
        return best_local_rows, best_local_scope, best_local_dupes
    
    # Iterated Local Search
    # 1. Run local search
    # 2. Perturb
    # 3. Repeat
    
    # Initial local search
    _, cur_scope, cur_dupes = local_search_sa(max_steps=50000)
    
    # If we found a valid solution, try to reduce scope
    while time.time() < deadline - 5:
        # Check if we're stuck
        if cur_dupes == 0:
            # Try to reduce scope
            # Set target_scope to current_scope - 1
            target = cur_scope - 1
            if target < M_lower:
                break  # Can't go lower
            
            # Perturb: introduce some conflicts to escape local minimum
            # Pick a few random elements and move them
            num_perturb = random.randint(3, min(6, k))
            for _ in range(num_perturb):
                ri = random.randint(0, n_rows - 1)
                ei = random.randint(1, k)
                row = current_rows[ri]
                lower = row[ei - 1] + 1
                upper = row[ei + 1] - 1 if ei < k else target + 10
                if lower > upper:
                    continue
                new_val = random.randint(lower, upper)
                if new_val != row[ei]:
                    apply_move(ri, ei, new_val)
            
            # Run local search with target scope
            _, new_scope, new_dupes = local_search_sa(max_steps=30000, target_scope=target)
            
            if new_dupes == 0 and new_scope < cur_scope:
                cur_scope = new_scope
            else:
                # Revert to best found
                current_rows = [r[:] for r in best_rows_found]
                rebuild_diff_counts(current_rows)
                cur_scope = best_scope_found
                cur_dupes = 0 if best_scope_found == cur_scope else get_num_dupes()
        else:
            # Still have conflicts, try to resolve them
            # Perturb more aggressively
            num_perturb = random.randint(5, min(10, k * 2))
            for _ in range(num_perturb):
                ri = random.randint(0, n_rows - 1)
                ei = random.randint(1, k)
                row = current_rows[ri]
                lower = row[ei - 1] + 1
                upper = row[ei + 1] - 1 if ei < k else best_scope_found + 20
                if lower > upper:
                    continue
                new_val = random.randint(lower, upper)
                if new_val != row[ei]:
                    apply_move(ri, ei, new_val)
            
            _, new_scope, new_dupes = local_search_sa(max_steps=30000)
            
            if new_dupes < cur_dupes:
                cur_dupes = new_dupes
                cur_scope = new_scope
            else:
                # Revert
                current_rows = [r[:] for r in best_rows_found]
                rebuild_diff_counts(current_rows)
                cur_scope = best_scope_found
                cur_dupes = 0
    
    # Final validation and cleanup
    # Ensure all rows are strictly increasing
    for i in range(n_rows):
        r = best_rows_found[i]
        for j in range(1, len(r)):
            if r[j] <= r[j - 1]:
                r[j] = r[j - 1] + 1
    
    # Verify validity
    used_diffs = set()
    valid = True
    for row in best_rows_found:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used_diffs:
                    valid = False
                    break
                used_diffs.add(d)
        if not valid:
            break
    
    if valid:
        return best_rows_found
    else:
        # Fallback to greedy construction
        return greedy_rows
