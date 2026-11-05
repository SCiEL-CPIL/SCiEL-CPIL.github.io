import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    diffs_per_row = (k + 1) * k // 2

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

    def backtrack_with_scope(scope_limit, time_budget, random_seed=42):
        """
        Depth-first backtracking with forward checking to construct rows
        within a given scope limit. Uses randomization to explore different
        solution spaces.
        """
        rng = random.Random(random_seed)
        used_diffs = set()
        rows = []
        
        def try_build_row(pos, current_row, time_limit):
            """Recursively build a single row, trying values in randomized order."""
            if time.time() > time_limit:
                return False
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                return True
            
            last_val = current_row[-1]
            # Determine range of valid next values
            min_val = last_val + 1
            max_val = min(scope_limit, last_val + scope_limit)
            
            if min_val > max_val:
                return False
            
            # Generate candidate values: mix of greedy (small) and random
            candidates = list(range(min_val, max_val + 1))
            
            # Use a mixed strategy: first try smaller values (greedy), 
            # then randomize for diversity
            if len(candidates) > 50:
                # Take some small values and some random ones
                small_vals = candidates[:20]
                random_vals = rng.sample(candidates[20:], min(30, len(candidates) - 20))
                candidates = small_vals + random_vals
                rng.shuffle(candidates[20:])  # Shuffle the random part
            else:
                rng.shuffle(candidates)
            
            for val in candidates:
                if time.time() > time_limit:
                    return False
                
                # Check if val creates valid differences
                new_diffs = []
                valid = True
                for y in current_row:
                    d = val - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Forward check: can we still complete the row?
                # Simple heuristic: check that there's enough "room"
                remaining_positions = k + 1 - len(current_row) - 1
                if remaining_positions > 0:
                    # The maximum possible value for the last position
                    # must be at least val + remaining_positions (minimum gap of 1)
                    if val + remaining_positions > scope_limit:
                        continue
                
                # Tentatively add
                current_row.append(val)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if try_build_row(pos + 1, current_row, time_limit):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        time_limit = time.time() + time_budget
        for _ in range(n_rows):
            row = [0]
            if not try_build_row(0, row, time_limit):
                return None
        
        return rows

    def iterative_deepening_backtrack(time_budget, start_scope=120):
        """
        Try increasing scope limits with backtracking.
        """
        for scope in range(start_scope, start_scope + 50):
            remaining = time_budget - (time.time() - start_time)
            if remaining < 2:
                break
            
            # Allocate time for this scope attempt
            attempt_time = min(remaining * 0.3, 30)
            
            # Try a few random seeds for diversity
            for seed in [42, 123, 777]:
                rows = backtrack_with_scope(scope, attempt_time, seed)
                if rows is not None:
                    if is_valid(rows):
                        return rows, scope
            
            # If we couldn't find a solution at this scope, try next
            if time.time() > start_time + time_budget:
                break
        
        return None, None

    def sa_optimize(rows, time_budget):
        """
        Simulated annealing to reduce scope from an initial valid solution.
        """
        if rows is None:
            return None, None
            
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        # Precompute used diffs
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        t_end = time.time() + time_budget
        initial_temp = max(1.0, current_scope / 4.0)
        cooling_rate = 0.999
        
        while time.time() < t_end:
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 10
            
            if min_val > max_val:
                continue
            
            # Try multiple random perturbations
            for _ in range(random.randint(2, 5)):
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
                    continue
                
                # Collect old diffs
                old_diffs = set()
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = old_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - old_val
                        old_diffs.add(d)
                
                # Remove old diffs
                for d in old_diffs:
                    used_diffs.discard(d)
                
                # Check new diffs
                new_diffs = set()
                valid = True
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = new_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - new_val
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                
                if not valid:
                    used_diffs.update(old_diffs)
                    continue
                
                # Compute new scope
                new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
                other_max = 0
                for i, r in enumerate(rows):
                    if i != row_idx:
                        other_max = max(other_max, max(r))
                new_scope = max(new_row_max, other_max)
                
                # Acceptance criterion
                delta = current_scope - new_scope
                if delta > 0 or random.random() < pow(2.718281828, delta / initial_temp):
                    rows[row_idx][pos] = new_val
                    used_diffs.update(new_diffs)
                    current_scope = new_scope
                    
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                    break
                else:
                    used_diffs.update(old_diffs)
            
            initial_temp *= cooling_rate
        
        return best_rows, best_scope

    def greedy_construct(scope_limit):
        """Simple greedy construction as a fast initial solution."""
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

    # Strategy 1: Try greedy first for a quick valid solution
    initial_rows = None
    for scope in [130, 140, 150, 160, 200, 300]:
        initial_rows = greedy_construct(scope)
        if initial_rows is not None and is_valid(initial_rows):
            break
    
    # If greedy failed, try backtracking
    if initial_rows is None:
        remaining = deadline - time.time()
        if remaining > 10:
            bt_rows, bt_scope = iterative_deepening_backtrack(remaining - 20, 120)
            if bt_rows is not None:
                initial_rows = bt_rows
    
    if initial_rows is None:
        # Last resort: very loose greedy
        initial_rows = greedy_construct(1000)
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
        sa_rows, sa_scope = sa_optimize(initial_rows, remaining - 5)
        if sa_rows is not None and is_valid(sa_rows):
            if sa_scope < current_scope:
                initial_rows = sa_rows
                current_scope = sa_scope

    # Strategy 3: Try backtracking from the improved scope
    remaining = deadline - time.time()
    if remaining > 10 and current_scope > 120:
        target_scope = max(120, current_scope - 5)
        for seed in [42, 123]:
            bt_rows = backtrack_with_scope(target_scope, remaining * 0.4, seed)
            if bt_rows is not None and is_valid(bt_rows):
                bt_scope = get_scope(bt_rows)
                if bt_scope < current_scope:
                    initial_rows = bt_rows
                    current_scope = bt_scope
                break

    return initial_rows
