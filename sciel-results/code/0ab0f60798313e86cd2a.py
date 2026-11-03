import time
import random
import numpy as np

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 135.0
    row_len = k + 1
    # Number of differences per row: row_len * (row_len - 1) / 2
    n_diffs_per_row = row_len * (row_len - 1) // 2
    
    def get_diffs(row):
        """Compute all differences in a row."""
        diffs = []
        for i in range(row_len):
            for j in range(i + 1, row_len):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid_row(row, used_diffs):
        """Check if a row is valid given used differences."""
        if row[0] != 0:
            return False
        for i in range(row_len - 1):
            if row[i + 1] <= row[i]:
                return False
        diffs = get_diffs(row)
        for d in diffs:
            if d in used_diffs:
                return False
        return True

    def greedy_row(used_diffs, max_val=200):
        """Greedily construct a row starting at 0 with differences not in used_diffs."""
        row = [0]
        current_max = 0
        while len(row) < row_len:
            # Try to find the next number
            found = False
            for x in range(row[-1] + 1, max_val + 1):
                # Check differences with existing row elements
                valid = True
                for a in row:
                    if (x - a) in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    # Update used_diffs locally for this row construction
                    # But we should be careful, greedy is for initialization
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        return row

    def random_row(rng, used_diffs, max_val=200):
        """Try to construct a random valid row."""
        # Simple backtracking
        def backtrack(row, last_val):
            if len(row) == row_len:
                return row[:]
            start = last_val + 1
            end = min(max_val, row[-1] + 50) # heuristic limit
            candidates = list(range(start, end + 1))
            rng.shuffle(candidates)
            for x in candidates:
                valid = True
                for a in row:
                    if (x - a) in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    res = backtrack(row, x)
                    if res:
                        return res
                    row.pop()
            return None
        
        res = backtrack([0], 0)
        return res

    def fast_greedy(used_diffs_set):
        """Fast greedy using set operations."""
        row = [0]
        while len(row) < row_len:
            x = row[-1] + 1
            while True:
                # Check if x is valid
                valid = True
                for a in row:
                    if (x - a) in used_diffs_set:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    break
                x += 1
        return row

    def build_diff_counts(rows):
        """Build an array of difference counts."""
        # Max possible diff is roughly scope
        # Let's use a dict or large array
        # For (7,5), scope ~112, max diff ~112. Let's allocate 500 to be safe.
        max_diff = 500
        counts = np.zeros(max_diff, dtype=np.int8)
        for row in rows:
            diffs = get_diffs(row)
            for d in diffs:
                if d < max_diff:
                    counts[d] += 1
        return counts

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def run_sa(seed, deadline):
        rng = random.Random(seed)
        
        # 1. Initialization: Greedy
        used_set = set()
        rows = []
        for _ in range(n_rows):
            row = fast_greedy(used_set)
            if row is None:
                return None
            rows.append(row)
            for d in get_diffs(row):
                used_set.add(d)
        
        # 2. Simulated Annealing
        # State: rows
        # Objective: Minimize scope (max entry)
        # Constraint: All differences unique across rows
        
        current_scope = scope_of(rows)
        best_scope = current_scope
        best_rows = [r[:] for r in rows]
        
        # Temperature schedule
        T = 100.0
        cooling = 0.995
        steps = 0
        max_steps = 50000
        
        # Pre-compute difference counts for fast conflict checking
        # We need to handle removals and additions
        # diff_count[d] = number of rows containing difference d
        # If diff_count[d] > 1, it's a conflict (invalid)
        # Initially valid, so all counts are 0 or 1
        
        max_diff = 500
        diff_counts = np.zeros(max_diff, dtype=np.int8)
        for row in rows:
            for d in get_diffs(row):
                if d < max_diff:
                    diff_counts[d] += 1
        
        # Conflict count: number of d where diff_counts[d] > 1
        conflict_count = np.sum(diff_counts > 1)
        
        while time.time() < deadline and steps < max_steps:
            steps += 1
            # Pick a random row to modify
            idx = rng.randrange(n_rows)
            old_row = rows[idx]
            old_max = old_row[-1]
            old_diffs = get_diffs(old_row)
            
            # Remove old row from diff_counts
            for d in old_diffs:
                if d < max_diff:
                    diff_counts[d] -= 1
            
            # Generate a new row
            # Strategy: Try to compress the row
            # Pick a new target max
            # If we want to reduce scope, try target < old_max
            # Otherwise, random around old_max
            if rng.random() < 0.5 and current_scope > 100:
                target_max = rng.randint(100, old_max - 1)
            else:
                target_max = rng.randint(old_max, old_max + 10)
            
            new_row = random_row(rng, set(), target_max)
            if new_row is None:
                # Failed to generate, revert
                for d in old_diffs:
                    if d < max_diff:
                        diff_counts[d] += 1
                continue
            
            # Check if new row is valid with current other rows
            # i.e., new diffs don't intersect with used diffs of others
            # Since we removed old_row, diff_counts reflects others
            # If any d in new_row has diff_counts[d] > 0, it conflicts
            new_diffs = get_diffs(new_row)
            valid = True
            for d in new_diffs:
                if d < max_diff and diff_counts[d] > 0:
                    valid = False
                    break
            
            if not valid:
                for d in old_diffs:
                    if d < max_diff:
                        diff_counts[d] += 1
                continue
            
            # Calculate new score
            # Score = scope
            new_scope_candidate = max([r[-1] for r in rows if i != idx] + [new_row[-1]])
            
            # Update diff_counts for new row
            for d in new_diffs:
                if d < max_diff:
                    diff_counts[d] += 1
            
            # Delta in scope
            delta_scope = new_scope_candidate - current_scope
            
            # Acceptance
            if delta_scope <= 0 or (T > 0 and rng.random() < np.exp(-delta_scope / T)):
                rows[idx] = new_row
                current_scope = new_scope_candidate
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                for d in new_diffs:
                    if d < max_diff:
                        diff_counts[d] -= 1
                for d in old_diffs:
                    if d < max_diff:
                        diff_counts[d] += 1
            
            T *= cooling
        
        return best_rows

    try:
        overall_best = None
        seed = 0
        # Run multiple SA instances
        # Give first one more time
        first_dl = t0 + 60
        while time.time() < DEADLINE - 5:
            if seed == 0:
                dl = first_dl
            else:
                dl = min(DEADLINE - 2, time.time() + 20)
            
            result = run_sa(seed, dl)
            if result is not None:
                sc = scope_of(result)
                if overall_best is None or sc < overall_best[0]:
                    overall_best = (sc, result)
            
            seed += 1
        
        if overall_best is not None:
            return overall_best[1]
            
    except Exception:
        pass

    # Fallback: Greedy
    used_set = set()
    rows = []
    for _ in range(n_rows):
        row = fast_greedy(used_set)
        if row:
            rows.append(row)
            for d in get_diffs(row):
                used_set.add(d)
    return rows

