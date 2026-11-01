"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization (refinement machinery) ----------
    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Randomized greedy for this row to escape local minima in row construction
            jitter = 0.1 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                for trial_x in range(x + 1, x + 50):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Apply jitter: sometimes skip a valid candidate to explore
                        if jitter > 0 and random.random() < jitter and trial_x < 200:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        # Limit the number of full passes to avoid spending too much time here
        # if we are in an ILS loop
        max_passes = 5 
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=50)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima using improved rebuild."""
        return kick_improved(rows)

    # ---------- Improved kick: destroy and repair with target scope ----------
    def rebuild_row_with_target(used_other, target_max, attempts=30):
        """Try to build a row with max <= target_max, avoiding used_other diffs."""
        for attempt in range(attempts):
            row, x = [0], 0
            row_diffs = set()
            success = True
            jitter = 0.05 * (attempt + 1)
            while len(row) < k + 1:
                found = False
                # Look ahead for a valid position
                for trial_x in range(x + 1, min(x + 80, target_max + 1)):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs):
                        if jitter > 0 and random.random() < jitter:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) <= target_max:
                return row
        return None

    def kick_improved(rows):
        """Perturb by rebuilding 1-2 rows with a slightly relaxed target."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            current_max = max(rows[idx])
            # Try to rebuild with same or slightly smaller target
            target = current_max
            new_row = rebuild_row_with_target(used_other, target, attempts=20)
            
            if new_row is None:
                # Try with slightly larger target
                target = current_max + 5
                new_row = rebuild_row_with_target(used_other, target, attempts=20)
            
            if new_row is not None:
                rows[idx] = new_row
            else:
                # Fallback: greedy rebuild
                row, x = [0], 0
                row_diffs = set()
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs):
                        row.append(x)
                        row_diffs |= new
                rows[idx] = row
        
        return rows

    # ---------- Row swap move: try to find a better row for a specific index ----------
    def try_swap_row(rows, row_idx, used_other, target_max, max_attempts=50):
        """Try to find a row with max <= target_max to replace rows[row_idx]."""
        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            row_diffs = set()
            success = True
            jitter = 0.03 * (attempt + 1)
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, min(x + 60, target_max + 1)):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs):
                        if jitter > 0 and random.random() < jitter:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) <= target_max:
                return row
        return None

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search with improved kicks
    while time_left() > 25:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick (improved)
        if time_left() < 18:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
        else:
            pass

    # Final high-precision polish: try to reduce scope by 1 using targeted row swaps
    if time_left() > 15:
        for _ in range(3):
            if time_left() < 10:
                break
            current_scope = best_scope
            improved = False
            # Try to find a row that can be replaced with one having smaller max
            order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
            for idx in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != idx:
                        used_other |= row_diffs_set(best_rows[j])
                
                current_row_max = max(best_rows[idx])
                # Try target = current_row_max - 1
                target = current_row_max - 1
                if target >= k:  # minimum possible max is k
                    new_row = try_swap_row(best_rows, idx, used_other, target, max_attempts=30)
                    if new_row is not None:
                        test_rows = [list(r) for r in best_rows]
                        test_rows[idx] = new_row
                        if is_valid(test_rows):
                            best_rows = test_rows
                            best_scope = scope(best_rows)
                            improved = True
                            break
            
            if not improved:
                break

    # Final polish to ensure we squeeze out any last drops
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
