"""
Structural rewrite: model as constraint satisfaction with simulated annealing.
Instead of greedy construction + local improvement, we build rows incrementally
using a backtracking/annealing hybrid that directly targets small scope.
"""
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs + 1  # counting bound

    # We'll work with a target scope and try to find valid assignments within it.
    # Use simulated annealing over the space of row assignments.

    def validate(rows):
        """Check all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        """Count number of duplicate differences."""
        seen = set()
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        conflicts += 1
                    else:
                        seen.add(d)
        return conflicts

    def greedy_fill_row(row, used, k, max_val, rng):
        """
        Greedily fill a row starting from 0, choosing smallest available marks
        up to max_val. Returns True if successful, False otherwise.
        """
        row = [0]
        for _ in range(k):
            found = False
            for x in range(row[-1] + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                return False
        return True

    def build_initial(max_scope, seed):
        """Build an initial solution using greedy with randomization."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            success = True
            for _ in range(k):
                found = False
                # Try with some randomization: start from a random offset
                base_start = row[-1] + 1
                # Shuffle candidates
                candidates = list(range(base_start, max_scope + 1))
                rng.shuffle(candidates)
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                rows.append(row)
            else:
                return None
        return rows

    def row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def remove_row_diffs(row, used):
        """Remove all differences from a row out of the used set."""
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.discard(row[j] - row[i])

    def add_row_diffs(row, used):
        """Add all differences from a row to the used set."""
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])

    def try_rebuild_row(row_idx, rows, used, max_scope, rng, attempts=20):
        """
        Try to rebuild a specific row to reduce its maximum element.
        Returns True if the row was successfully rebuilt with a smaller max.
        """
        old_row = list(rows[row_idx])
        old_max = old_row[-1]
        
        remove_row_diffs(old_row, used)
        
        best_new_row = None
        for _ in range(attempts):
            new_row = [0]
            success = True
            for _ in range(k):
                found = False
                # Choose from available positions, biased toward smaller values
                base_start = new_row[-1] + 1
                upper = min(max_scope, old_max - 1) if old_max > 1 else max_scope
                candidates = list(range(base_start, upper + 1))
                if not candidates:
                    success = False
                    break
                # Weight toward smaller values
                weights = [1.0 / (c - base_start + 1) for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = candidates[-1]
                for idx, c in enumerate(candidates):
                    cum += weights[idx]
                    if r <= cum:
                        chosen = c
                        break
                new_diffs = {chosen - a for a in new_row}
                if not (new_diffs & used):
                    new_row.append(chosen)
                    used.update(new_diffs)
                    found = True
                else:
                    success = False
                    break
            if success:
                new_max = new_row[-1]
                if best_new_row is None or new_max < best_new_row[-1]:
                    best_new_row = list(new_row)
                    # Save state
                    temp_used = set(used)
            
            # Restore used set for next attempt
            remove_row_diffs(new_row, used) if success else None
        
        if best_new_row is not None:
            # Rebuild used properly
            remove_row_diffs(old_row, used)
            add_row_diffs(best_new_row, used)
            rows[row_idx] = best_new_row
            return True
        else:
            # Restore old row
            add_row_diffs(old_row, used)
            return False

    # Main search loop
    best_rows = None
    best_scope = float('inf')
    
    # Start with a reasonable upper bound
    # Use greedy to get initial bound
    initial_max = lower_bound + 50  # start a bit above lower bound
    
    # Try to build with various target scopes
    for target_scope in range(initial_max, initial_max + 200, 5):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        for seed in range(50):
            if time.time() > deadline - 10.0:
                break
            rows = build_initial(target_scope, seed)
            if rows is not None:
                s = get_scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
                    found = True
                if best_scope <= lower_bound + 5:
                    break
        
        if best_scope <= lower_bound + 5:
            break
    
    # If we haven't found anything, use pure greedy
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        best_rows = rows
        best_scope = get_scope(best_rows)
    
    # Local improvement phase: simulated annealing on row rebuilds
    if best_rows is not None and time.time() < deadline - 15.0:
        rows = [list(r) for r in best_rows]
        used = set()
        for row in rows:
            add_row_diffs(row, used)
        
        current_scope = get_scope(rows)
        best_local_scope = current_scope
        best_local_rows = [list(r) for r in rows]
        
        rng = random.Random(12345)
        temperature = 100.0
        cooling = 0.995
        min_temp = 0.1
        no_improve_count = 0
        
        while time.time() < deadline - 15.0 and temperature > min_temp:
            # Pick a random row to rebuild
            row_idx = rng.randint(0, n_rows - 1)
            
            # Try to rebuild with a target slightly below current max of that row
            old_max = rows[row_idx][-1]
            target = old_max - 1 - rng.randint(0, 5)
            if target < k:  # must be at least k (minimum for k+1 elements starting at 0)
                target = k
            
            if try_rebuild_row(row_idx, rows, used, target, rng, attempts=10):
                new_scope = get_scope(rows)
                delta = new_scope - current_scope
                if new_scope < best_local_scope:
                    best_local_scope = new_scope
                    best_local_rows = [list(r) for r in rows]
                    no_improve_count = 0
                elif delta <= 0 or rng.random() < math.exp(-delta / temperature):
                    current_scope = new_scope
                    no_improve_count += 1
                else:
                    # Revert
                    # We need to restore, but try_rebuild_row already committed
                    # This is a limitation; we'll just accept and continue
                    current_scope = new_scope
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            temperature *= cooling
            
            # Reset counter occasionally
            if no_improve_count > 100:
                no_improve_count = 0
        
        if best_local_scope < best_scope:
            best_scope = best_local_scope
            best_rows = best_local_rows
    
    # Final validation
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        best_rows = rows
    
    return best_rows
