# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses simulated annealing over the full configuration with two move types:
    1) single-element perturbation (shift one element of one row)
    2) full row rebuild with a tighter bound
    Objective: minimize max_entry + penalty * num_conflicts
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    time_limit = 110.0  # Safe margin under 120s limit
    
    # Greedy baseline
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    # Compute all differences for a set of rows
    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def count_conflicts(rows):
        """Count how many differences are duplicated using a list for speed."""
        # Max entry is likely < 200 for this problem size. 
        # We use a fixed size list to avoid dict overhead.
        max_possible = 500 
        diff_count = [0] * (max_possible + 1)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_possible:
                        diff_count[d] += 1
                    else:
                        # Fallback for very large diffs (unlikely in optimal solutions)
                        # We can ignore or handle separately, but for counting conflicts,
                        # if d > max_possible, it's unique unless another row has same d.
                        # Given the small search space, we assume d <= max_possible.
                        pass
        conflicts = 0
        for cnt in diff_count:
            if cnt > 1:
                conflicts += cnt - 1
        return conflicts
    
    def objective(rows):
        """Objective: max_entry + large_penalty * conflicts. Lower is better."""
        mx = max(max(r) for r in rows)
        conflicts = count_conflicts(rows)
        return mx + 1000 * conflicts
    
    # Simulated annealing
    def simulated_annealing(initial_rows, n_rows, k, timeout, seed=0):
        deadline = time.time() + timeout
        rng = random.Random(seed)
        
        current_rows = [list(r) for r in initial_rows]
        current_obj = objective(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_obj = current_obj
        
        # Temperature schedule
        T0 = max(10.0, greedy_max * 0.5)
        T_end = 0.1
        num_steps = 50000
        cooling_rate = (T_end / T0) ** (1.0 / max(1, num_steps - 1))
        
        # Precompute: for each row, which positions are "movable" (not position 0)
        # Position 0 is always 0.
        
        step = 0
        while time.time() < deadline:
            step += 1
            T = T0 * (cooling_rate ** step)
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.7:
                # Move type 1: shift one element of one row
                row_idx = rng.randint(0, n_rows - 1)
                row = current_rows[row_idx]
                # Choose a non-zero position
                pos = rng.randint(1, len(row) - 1)
                old_val = row[pos]
                
                # Try a new value: random in a window around old_val
                window = max(3, int(greedy_max * 0.05))
                delta = rng.randint(-window, window)
                new_val = old_val + delta
                
                # Constraints: must be > row[pos-1] and < row[pos+1] (if exists)
                if pos > 0 and new_val <= row[pos - 1]:
                    new_val = row[pos - 1] + 1
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    new_val = row[pos + 1] - 1
                
                if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                    continue
                
                if new_val == old_val:
                    continue
                
                # Apply move
                row[pos] = new_val
                
                # Check validity of this row (strictly increasing)
                valid_row = all(row[i] < row[i+1] for i in range(len(row)-1))
                if not valid_row:
                    row[pos] = old_val
                    continue
                
                new_obj = objective(current_rows)
                delta_obj = new_obj - current_obj
                
                # Acceptance
                if delta_obj < 0 or rng.random() < math.exp(-delta_obj / max(T, 1e-10)):
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_rows = [list(r) for r in current_rows]
                else:
                    row[pos] = old_val
                    
            elif move_type < 0.85:
                # Move type 2: swap two elements from different rows
                if n_rows < 2:
                    continue
                r1, r2 = rng.sample(range(n_rows), 2)
                p1 = rng.randint(1, len(current_rows[r1]) - 1)
                p2 = rng.randint(1, len(current_rows[r2]) - 1)
                
                v1 = current_rows[r1][p1]
                v2 = current_rows[r2][p2]
                
                # Swap
                current_rows[r1][p1] = v2
                current_rows[r2][p2] = v1
                
                # Check both rows are still valid
                valid1 = all(current_rows[r1][i] < current_rows[r1][i+1] for i in range(len(current_rows[r1])-1))
                valid2 = all(current_rows[r2][i] < current_rows[r2][i+1] for i in range(len(current_rows[r2])-1))
                
                if not (valid1 and valid2):
                    current_rows[r1][p1] = v1
                    current_rows[r2][p2] = v2
                    continue
                
                new_obj = objective(current_rows)
                delta_obj = new_obj - current_obj
                
                if delta_obj < 0 or rng.random() < math.exp(-delta_obj / max(T, 1e-10)):
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_rows = [list(r) for r in current_rows]
                else:
                    current_rows[r1][p1] = v1
                    current_rows[r2][p2] = v2
                    
            else:
                # Move type 3: rebuild one row from scratch with a target max
                row_idx = rng.randint(0, n_rows - 1)
                old_row = current_rows[row_idx]
                
                # Compute diffs used by other rows
                other_diffs = set()
                for ri in range(n_rows):
                    if ri == row_idx:
                        continue
                    row = current_rows[ri]
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            other_diffs.add(row[j] - row[i])
                
                # Try to build a new row with max <= current best max
                target_max = min(best_obj - 1000 * count_conflicts(best_rows), old_row[-1])
                if target_max < k:
                    target_max = k
                
                new_row = build_row_greedy(other_diffs, k, target_max, rng)
                if new_row is not None:
                    current_rows[row_idx] = new_row
                    new_obj = objective(current_rows)
                    delta_obj = new_obj - current_obj
                    
                    if delta_obj < 0 or rng.random() < math.exp(-delta_obj / max(T, 1e-10)):
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_rows = [list(r) for r in current_rows]
                    else:
                        current_rows[row_idx] = old_row
        
        return best_rows, best_obj
    
    def build_row_greedy(used_diffs, k, max_entry, rng):
        """Build a single row greedily avoiding used_diffs, with randomization."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_entry:
                return None
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
            # Add some randomness: sometimes skip a valid value
            if rng.random() < 0.1:
                continue
            row.append(x)
            used_diffs = used_diffs | new_diffs
        return row
    
    # Run multiple SA instances with different seeds
    best_rows_overall = greedy_rows
    best_max_overall = greedy_max
    
    num_sa_instances = 2
    time_per_sa = (time_limit - 5.0) / num_sa_instances
    
    for sa_seed in range(num_sa_instances):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 3.0
        if remaining < 2.0:
            break
        
        sa_timeout = min(time_per_sa, remaining)
        
        # Vary initial solution slightly for diversity
        if sa_seed == 0:
            initial = greedy_rows
        else:
            # Perturb greedy solution
            initial = [list(r) for r in greedy_rows]
            rng_init = random.Random(sa_seed * 1000)
            for _ in range(10):
                ri = rng_init.randint(0, n_rows - 1)
                pi = rng_init.randint(1, len(initial[ri]) - 1)
                # Try to shift
                delta = rng_init.randint(-5, 5)
                new_v = initial[ri][pi] + delta
                if pi > 0 and new_v > initial[ri][pi-1] and (pi == len(initial[ri])-1 or new_v < initial[ri][pi+1]):
                    initial[ri][pi] = new_v
        
        sa_result, sa_obj = simulated_annealing(initial, n_rows, k, sa_timeout, seed=sa_seed * 7 + 3)
        
        sa_max = max(max(r) for r in sa_result)
        sa_conflicts = count_conflicts(sa_result)
        
        # Only accept if valid (no conflicts) or if it reduces max significantly
        if sa_conflicts == 0 and sa_max < best_max_overall:
            best_rows_overall = sa_result
            best_max_overall = sa_max
        elif sa_conflicts == 0 and sa_max <= best_max_overall:
            # Same max but different structure might be useful; keep best
            pass
        elif sa_conflicts > 0:
            # Invalid - don't use, but check if partial improvement exists
            # Try to fix conflicts by rebuilding conflicting rows
            pass
    
    # Final polish: try to reduce max by rebuilding the row with the largest entry
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 2.0
    if remaining > 1.0 and best_max_overall > 0:
        # Find row with max entry
        max_row_idx = max(range(n_rows), key=lambda i: best_rows_overall[i][-1])
        max_row = best_rows_overall[max_row_idx]
        
        # Compute diffs from other rows
        other_diffs = set()
        for ri in range(n_rows):
            if ri == max_row_idx:
                continue
            row = best_rows_overall[ri]
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    other_diffs.add(row[j] - row[i])
        
        # Try to rebuild with smaller max
        for shrink in range(1, min(20, max_row[-1] // 5 + 1)):
            target = max_row[-1] - shrink
            if target < k:
                break
            new_row = build_row_greedy(other_diffs, k, target, random.Random(12345 + shrink))
            if new_row is not None:
                best_rows_overall[max_row_idx] = new_row
                best_max_overall = max(max(r) for r in best_rows_overall)
                break
    
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows_overall, n_rows, k):
        best_rows_overall = greedy_construct(n_rows, k)
    
    return best_rows_overall

# EVOLVE-BLOCK-END
