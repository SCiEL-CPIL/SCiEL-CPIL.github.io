import time
import random
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    total_diffs_per_row = (k + 1) * k // 2

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_pressure_construct(scope_limit, rng):
        """
        Multi-start greedy with dynamic 'difference pressure' ordering.
        For each row, build it by selecting the next smallest candidate entry
        that minimizes the increase in the maximum used difference, thereby
        naturally packing differences into the lower range.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            # Precompute all currently used differences to quickly check validity
            # But we need to be careful: we are building one row at a time.
            
            while len(row) < k + 1:
                last_val = row[-1]
                min_val = last_val + 1
                
                # We want to find the smallest x in [min_val, scope_limit]
                # such that all differences x - y (for y in row) are not in used_diffs
                # AND (x - y) <= scope_limit (implied by x <= scope_limit)
                
                # To implement "pressure", we could prioritize candidates that keep
                # the maximum difference small. Since we pick the smallest valid x,
                # this naturally keeps differences small.
                
                # However, to add "pressure", we can look ahead or use a weighted score.
                # For now, let's stick to the smallest valid x, but with a twist:
                # If there are multiple valid x's, pick the one that minimizes
                # the maximum new difference introduced.
                
                best_x = None
                best_max_diff = float('inf')
                
                # Optimization: don't scan the entire range if we can prune.
                # The maximum difference introduced by x is x - row[0] = x.
                # So minimizing x minimizes the max diff.
                # Thus, the smallest valid x is the best for pressure.
                
                # Let's just find the smallest valid x.
                for x in range(min_val, scope_limit + 1):
                    valid = True
                    max_new_diff = 0
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        if d > max_new_diff:
                            max_new_diff = d
                    
                    if valid:
                        # Since we iterate x in increasing order, the first valid x
                        # has the smallest max_new_diff.
                        best_x = x
                        break
                
                if best_x is None:
                    return None # Failed to build row within scope
                
                row.append(best_x)
                for y in row[:-1]:
                    used_diffs.add(best_x - y)
            
            rows.append(row)
        
        return rows

    def greedy_construct(scope_limit, rng):
        """Standard greedy: pick smallest valid value."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                for x in range(last_val + 1, scope_limit + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def sa_optimize(rows, time_budget, rng):
        """
        Simulated annealing to reduce scope from an initial valid solution.
        Uses incremental penalty tracking for speed.
        """
        if rows is None:
            return None, None
            
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        # Precompute used diffs and counts for incremental updates
        used_diffs = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used_diffs[d] = used_diffs.get(d, 0) + 1
        
        # Penalty is the number of duplicates (count - 1 for each diff with count > 1)
        # But we want a valid solution, so penalty should be high if any diff count > 1.
        # Let's define penalty as sum of (count - 1) for all diffs.
        # If penalty is 0, it's valid.
        
        def calc_penalty(counts):
            pen = 0
            for c in counts.values():
                if c > 1:
                    pen += c - 1
            return pen

        curr_penalty = calc_penalty(used_diffs)
        
        t_end = time.time() + time_budget
        initial_temp = max(1.0, current_scope / 4.0)
        cooling_rate = 0.9995
        lambda_pen = 100.0 # High weight to ensure validity
        
        curr_energy = current_scope + lambda_pen * curr_penalty
        
        while time.time() < t_end:
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)
            
            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            # Allow expansion slightly beyond current scope to escape local minima
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else current_scope + 20
            
            if min_val > max_val:
                continue
            
            new_val = rng.randint(min_val, max_val)
            if new_val == old_val:
                continue
            
            # Remove old diffs
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)
            
            for d in old_diffs:
                used_diffs[d] -= 1
                if used_diffs[d] == 0:
                    del used_diffs[d]
            
            # Check new diffs
            new_diffs = []
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    new_diffs.append(d)
            
            # Tentatively add new diffs
            for d in new_diffs:
                used_diffs[d] = used_diffs.get(d, 0) + 1
            
            new_penalty = calc_penalty(used_diffs)
            
            # Compute new scope
            new_row = rows[row_idx][:]
            new_row[pos] = new_val
            new_row_max = max(new_row)
            other_max = 0
            for i, r in enumerate(rows):
                if i != row_idx:
                    if max(r) > other_max:
                        other_max = max(r)
            new_scope = max(new_row_max, other_max)
            
            new_energy = new_scope + lambda_pen * new_penalty
            delta_E = new_energy - curr_energy
            
            # Acceptance criterion
            if delta_E < 0 or rng.random() < pow(2.718281828, -delta_E / initial_temp):
                rows[row_idx][pos] = new_val
                curr_energy = new_energy
                current_scope = new_scope
                curr_penalty = new_penalty
                
                if curr_penalty == 0:
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
            else:
                # Revert
                for d in new_diffs:
                    used_diffs[d] -= 1
                    if used_diffs[d] == 0:
                        del used_diffs[d]
                for d in old_diffs:
                    used_diffs[d] = used_diffs.get(d, 0) + 1
            
            initial_temp *= cooling_rate
            if initial_temp < 0.1:
                initial_temp = 0.1
        
        return best_rows, best_scope

    # Strategy 1: Try greedy pressure construction first
    initial_rows = None
    rng = random.Random(42)
    
    # Try a few different scope limits and random seeds
    for scope in [134, 130, 140, 150, 200]:
        # Try multiple random seeds for diversity
        for seed in [42, 123, 777, 999, 1234]:
            rng = random.Random(seed)
            rows = greedy_pressure_construct(scope, rng)
            if rows is not None and is_valid(rows):
                initial_rows = rows
                break
        if initial_rows is not None:
            break
    
    # If greedy pressure failed, try standard greedy
    if initial_rows is None:
        for scope in [134, 140, 150, 200, 300]:
            rng = random.Random(42)
            rows = greedy_construct(scope, rng)
            if rows is not None and is_valid(rows):
                initial_rows = rows
                break
    
    if initial_rows is None:
        # Last resort: very loose greedy
        rng = random.Random(42)
        initial_rows = greedy_construct(1000, rng)
        if initial_rows is None:
            # Absolute fallback
            used_diffs = set()
            initial_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                initial_rows.append(row)
            return initial_rows

    current_scope = get_scope(initial_rows)

    # Strategy 2: Use SA to reduce scope
    remaining = deadline - time.time()
    if remaining > 10:
        rng = random.Random(42)
        sa_rows, sa_scope = sa_optimize(initial_rows, remaining - 5, rng)
        if sa_rows is not None and is_valid(sa_rows):
            if sa_scope < current_scope:
                initial_rows = sa_rows
                current_scope = sa_scope

    return initial_rows
