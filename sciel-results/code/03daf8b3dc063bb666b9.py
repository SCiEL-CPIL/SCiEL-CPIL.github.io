"""Difference triangle set: find n_rows rows of k+1 increasing ints starting at 0
minimizing the largest entry, subject to all pairwise differences within each row
being distinct across all rows."""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def all_diffs(row):
        """Return set of all pairwise differences in a row."""
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
    
    def greedy_row(used_diffs, max_val):
        """Build a single row greedily: start at 0, pick smallest next value
        such that all new differences are unused."""
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while x <= max_val:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
                    break
                x += 1
            else:
                return None
        return row
    
    def random_row(used_diffs, max_val, rng):
        """Try to build a row with some randomization in mark selection."""
        row = [0]
        while len(row) < k + 1:
            # Try candidates in random order near the minimum
            base = row[-1] + 1
            candidates = list(range(base, min(base + 20, max_val + 1)))
            if len(candidates) < 10:
                candidates = list(range(base, max_val + 1))
            rng.shuffle(candidates)
            found = False
            for x in candidates:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
            if not found:
                # Fallback: scan from base
                x = base
                while x <= max_val:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                    x += 1
            if not found:
                return None
        return row
    
    def build_rows_greedy(max_val, rng, randomize_frac=0.0):
        """Build all rows greedily up to max_val. Returns list of rows or None."""
        used = set()
        rows = []
        for i in range(n_rows):
            if randomize_frac > 0 and rng.random() < randomize_frac:
                row = random_row(used, max_val, rng)
            else:
                row = greedy_row(used, max_val)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    def score_rows(rows):
        """Return the maximum entry across all rows."""
        return max(max(r) for r in rows)
    
    def hill_climb(rows, used_diffs, max_val, rng, time_budget=5.0):
        """Local search: try to reduce max entry by rebuilding a random row
        with a tighter bound, or swapping marks within a row."""
        start = time.time()
        current_max = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_max = current_max
        best_used = set(used_diffs)
        
        while time.time() - start < time_budget:
            # Pick a random row to rebuild
            idx = rng.randint(0, n_rows - 1)
            # Remove its differences
            removed = all_diffs(rows[idx])
            for d in removed:
                best_used.discard(d)
            
            # Try to rebuild this row with a smaller max
            target_max = max(best_rows[idx][0], best_max - 1)
            if target_max < k:
                target_max = best_max
            
            # Try a few times to build a smaller row
            improved = False
            for attempt in range(30):
                new_row = greedy_row(best_used, target_max)
                if new_row is not None:
                    best_rows[idx] = new_row
                    new_max = max(max(r) for r in best_rows)
                    if new_max < best_max:
                        best_max = new_max
                        best_used |= all_diffs(new_row)
                        improved = True
                        break
                    # Even if same max, accept if it frees up differences
                    # for potential future improvement
                    # For now, only accept strict improvement
                    best_used.discard(0)  # clean up
                    for d in all_diffs(new_row):
                        if d not in best_used:
                            pass
                    # Revert: put back old row's diffs
                    best_used |= removed
                    break
                # Try random row
                new_row = random_row(best_used, target_max, rng)
                if new_row is not None:
                    new_max = max(max(r) for r in best_rows)
                    if max(new_row) < best_rows[idx][0] or new_max < best_max:
                        best_rows[idx] = new_row
                        best_max = max(max(r) for r in best_rows)
                        best_used |= all_diffs(new_row)
                        improved = True
                        break
                    best_used |= removed
                    break
            
            if not improved:
                # Put back the original row's differences
                best_used |= removed
        
        return best_rows, best_used, best_max
    
    # Phase 1: Initial greedy solution
    rng = random.Random(42)
    
    # Find a feasible max_val by doubling
    max_val = k * (k + 1) // 2 * n_rows + 10
    initial_rows = None
    while True:
        initial_rows = build_rows_greedy(max_val, rng)
        if initial_rows is not None:
            break
        max_val = int(max_val * 1.5) + 5
        if time.time() > deadline - 30:
            # Last resort: just return what we can
            # Try without bound
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
    
    current_rows = initial_rows
    current_max = score_rows(current_rows)
    current_used = set()
    for r in current_rows:
        current_used |= all_diffs(r)
    
    # Phase 2: Iterative improvement via local search
    # Try to reduce the scope by rebuilding rows with tighter bounds
    attempt = 0
    while time.time() < deadline - 20:
        attempt += 1
        # Occasionally try a fresh greedy with different randomization
        if attempt % 5 == 0:
            fresh = build_rows_greedy(current_max - 1, rng, randomize_frac=0.3)
            if fresh is not None:
                fresh_max = score_rows(fresh)
                if fresh_max < current_max:
                    current_rows = fresh
                    current_max = fresh_max
                    current_used = set()
                    for r in current_rows:
                        current_used |= all_diffs(r)
                    continue
        
        # Local search on current solution
        time_budget = min(8.0, deadline - time.time() - 15)
        if time_budget < 1.0:
            break
        new_rows, new_used, new_max = hill_climb(
            current_rows, current_used, current_max, rng, time_budget
        )
        if new_max < current_max:
            current_rows = new_rows
            current_max = new_max
            current_used = new_used
    
    # Phase 3: Final attempt to squeeze with very tight bounds
    if time.time() < deadline - 10:
        for try_max in range(current_max - 1, max(k, current_max - 5), -1):
            if time.time() > deadline - 10:
                break
            # Try to find a complete solution with this max
            for trial in range(50):
                if time.time() > deadline - 10:
                    break
                used = set()
                rows = []
                success = True
                for i in range(n_rows):
                    if trial % 3 == 0:
                        row = greedy_row(used, try_max)
                    else:
                        row = random_row(used, try_max, rng)
                    if row is None:
                        success = False
                        break
                    rows.append(row)
                    used |= all_diffs(row)
                if success:
                    current_rows = rows
                    current_max = try_max
                    break
            if current_max == try_max:
                break
    
    return current_rows
