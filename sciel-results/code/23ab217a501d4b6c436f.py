"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper: get all positive differences in a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: check if a row is valid (strictly increasing, starts at 0)
    def is_valid_row(row):
        if not row or row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        return True
    
    # Helper: compute max value across all rows
    def get_max_val(rows):
        return max(max(r) for r in rows)
    
    # Greedy construction of a single row given used_diffs and a limit
    def greedy_row(used_diffs, limit=None):
        row = [0]
        local_used = set()
        x = 1
        while len(row) < k + 1:
            if limit is not None and x > limit:
                return None
            new_diffs = set(x - a for a in row)
            if not new_diffs & (used_diffs | local_used):
                row.append(x)
                local_used |= new_diffs
            x += 1
        return row
    
    # Randomized greedy: with some probability, skip the first few valid x's
    def randomized_greedy_row(used_diffs, limit=None, skip_prob=0.3):
        row = [0]
        local_used = set()
        x = 1
        while len(row) < k + 1:
            if limit is not None and x > limit:
                return None
            new_diffs = set(x - a for a in row)
            if not new_diffs & (used_diffs | local_used):
                # With some probability, skip this x to explore alternatives
                if random.random() < skip_prob and x + 1 <= (limit if limit else x + 50):
                    x += 1
                    continue
                row.append(x)
                local_used |= new_diffs
            x += 1
        return row
    
    # Initial greedy construction
    rows = []
    used_diffs = set()
    for _ in range(n_rows):
        row = greedy_row(used_diffs)
        if row is None:
            # Fallback: extend limit
            row = greedy_row(used_diffs, limit=500)
        rows.append(row)
        used_diffs |= get_diffs(row)
    
    best_rows = [r[:] for r in rows]
    best_cost = get_max_val(best_rows)
    
    # Simulated Annealing with row-replacement moves
    T = 30.0
    cooling = 0.9995
    min_T = 0.05
    
    def try_replace_row(rows, used_diffs, idx, best_cost, limit_extra=20):
        """Try to replace rows[idx] with a better row. Returns (new_row, new_used_diffs) or None."""
        old_row = rows[idx]
        old_diffs = get_diffs(old_row)
        
        # Remove old differences
        new_used_diffs = used_diffs - old_diffs
        
        # Try multiple strategies to build a new row
        best_new_row = None
        best_new_max = float('inf')
        
        limit = best_cost + limit_extra
        
        # Strategy 1: Standard greedy
        for attempt in range(5):
            new_row = greedy_row(new_used_diffs, limit=limit)
            if new_row is not None:
                new_max = max(new_row)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_new_row = new_row
        
        # Strategy 2: Randomized greedy
        for attempt in range(10):
            new_row = randomized_greedy_row(new_used_diffs, limit=limit, skip_prob=0.3 + 0.05 * attempt)
            if new_row is not None:
                new_max = max(new_row)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_new_row = new_row
        
        # Strategy 3: Local search on the old row - try small perturbations
        if old_max < best_new_max:
            # Try to shrink the old row
            candidate = old_row[:]
            improved = True
            while improved:
                improved = False
                for pos in range(1, len(candidate)):
                    orig_val = candidate[pos]
                    for new_val in range(candidate[pos - 1] + 1, orig_val):
                        test_row = candidate[:pos] + [new_val] + candidate[pos + 1:]
                        if not is_valid_row(test_row):
                            continue
                        test_diffs = get_diffs(test_row)
                        if not test_diffs & new_used_diffs:
                            candidate[pos:] = test_row[pos:]
                            improved = True
                            break
                if improved:
                    continue
            
            cand_max = max(candidate)
            if cand_max < best_new_max:
                best_new_max = cand_max
                best_new_row = candidate
        
        if best_new_row is None:
            return None
        
        # Verify
        new_diffs = get_diffs(best_new_row)
        if new_diffs & new_used_diffs:
            return None
        
        return best_new_row, new_used_diffs
    
    # Main SA loop
    current_rows = [r[:] for r in rows]
    current_used_diffs = used_diffs
    current_cost = best_cost
    
    while time.time() < deadline and T > min_T:
        # Pick a random row to replace
        idx = random.randint(0, n_rows - 1)
        
        result = try_replace_row(current_rows, current_used_diffs, idx, current_cost)
        if result is None:
            T *= cooling
            continue
        
        new_row, new_used_diffs = result
        new_cost = get_max_val(current_rows[:idx] + [new_row] + current_rows[idx + 1:])
        
        # Acceptance criterion
        if new_cost <= current_cost:
            # Always accept improvement or equal
            current_rows[idx] = new_row
            current_used_diffs = new_used_diffs
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [r[:] for r in current_rows]
        else:
            # Accept with probability exp(-delta/T)
            delta = new_cost - current_cost
            if random.random() < pow(2.718281828, -delta / T):
                current_rows[idx] = new_row
                current_used_diffs = new_used_diffs
                current_cost = new_cost
        
        T *= cooling
    
    # Final polish: try to improve each row one more time
    for idx in range(n_rows):
        if time.time() > deadline:
            break
        result = try_replace_row(current_rows, current_used_diffs, idx, best_cost, limit_extra=5)
        if result is not None:
            new_row, new_used_diffs = result
            new_cost = get_max_val(current_rows[:idx] + [new_row] + current_rows[idx + 1:])
            if new_cost < current_cost:
                current_rows[idx] = new_row
                current_used_diffs = new_used_diffs
                current_cost = new_cost
    
    # Update best
    if current_cost < best_cost:
        best_cost = current_cost
        best_rows = [r[:] for r in current_rows]
    
    return best_rows
# EVOLVE-BLOCK-END
