"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        arr = np.array(row)
        n = len(arr)
        if n < 2:
            return set()
        # Compute all pairwise differences using broadcasting
        diff_matrix = arr[np.newaxis, :] - arr[:, np.newaxis]
        # Take upper triangle (i < j means row[i] < row[j], so diff > 0)
        iu = np.triu_indices(n, k=1)
        diffs = diff_matrix[iu]
        return set(diffs.tolist())

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_set(row):
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

    # ---------- Bin-packing inspired row construction ----------
    # The idea: when building a row, we want to choose entries such that the
    # resulting set of differences is "balanced" - not too many small diffs
    # that could be used by other rows, but also not forced to use huge gaps.
    # We use a scoring function to evaluate partial rows.

    def score_partial_row(row, used_other, current_diffs):
        """Score a partial row: lower is better.
        Penalize:
        - Using very small differences (wastes cheap diffs)
        - Having a large gap to the next needed entry
        - Creating differences that are hard to fit elsewhere
        """
        if not row:
            return 0
        max_entry = row[-1]
        # Number of differences used so far
        n_diffs = len(current_diffs)
        # Expected total diffs for a full row
        total_diffs = (k + 1) * k // 2
        # Penalty for using small diffs (they're more contested)
        small_penalty = sum(1 for d in current_diffs if d <= 20) * 0.5
        # Penalty for the span relative to what's needed
        # A row of k+1 entries needs at least k*(k+1)/2 distinct diffs
        # The minimal span for m diffs is roughly sqrt(2*m)
        min_span_est = int(math.sqrt(2 * total_diffs)) + 1
        span_penalty = max(0, max_entry - min_span_est) * 0.1
        # Bonus for using a good mix
        if n_diffs > 0:
            avg_diff = sum(current_diffs) / n_diffs
            # We want avg_diff to be around min_span_est/2
            mix_penalty = abs(avg_diff - min_span_est / 2) * 0.05
        else:
            mix_penalty = 0
        return small_penalty + span_penalty + mix_penalty

    def build_row_smart(used_other, max_span, num_restarts=5):
        """Build a row using a smarter heuristic that considers difference distribution.
        Returns the best row found, or None if no valid row within max_span.
        """
        best_row = None
        best_score = float('inf')

        for restart in range(num_restarts):
            row = [0]
            current_diffs = set()
            x = 0
            success = True

            while len(row) < k + 1:
                # Find all valid next positions
                candidates = []
                for trial_x in range(x + 1, max_span + 1):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_diffs):
                        # Score this candidate
                        trial_diffs = current_diffs | new
                        trial_row = row + [trial_x]
                        sc = score_partial_row(trial_row, used_other, trial_diffs)
                        candidates.append((sc, trial_x, new))

                if not candidates:
                    success = False
                    break

                # Sort by score and pick from the top few with some randomness
                candidates.sort(key=lambda x: x[0])
                top_n = min(5, len(candidates))
                # Weighted random selection from top candidates
                weights = [1.0 / (i + 1) for i in range(top_n)]
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                chosen = candidates[0]
                for i in range(top_n):
                    cum += weights[i]
                    if r <= cum:
                        chosen = candidates[i]
                        break

                sc, trial_x, new = chosen
                row.append(trial_x)
                current_diffs |= new
                x = trial_x

            if success:
                row_score = score_partial_row(row, used_other, current_diffs)
                if row_score < best_score:
                    best_score = row_score
                    best_row = list(row)

        return best_row

    def try_reduce_row_v2(rows, row_idx, used_other, max_attempts=100):
        """Try to find a row at row_idx with smaller max entry using smart construction."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        # First pass: try to reduce by 1, 2, 3... entries
        for delta in range(1, 15):
            if time_left() < 3:
                break
            target_max = current_max - delta
            if target_max < k:
                break

            # Try multiple restarts at this target
            for restart in range(5):
                if time_left() < 3:
                    break
                row = build_row_smart(used_other, target_max, num_restarts=2)
                if row is not None and max(row) <= target_max:
                    if max(row) < best_max:
                        best_row = list(row)
                        best_max = max(row)
                        if best_max <= current_max - delta:
                            break  # Got the target, move to next delta

        # Second pass: fine-grained search around best found
        if best_max < current_max:
            for target_max in range(best_max - 1, max(k, best_max - 5), -1):
                if time_left() < 3:
                    break
                row = build_row_smart(used_other, target_max, num_restarts=3)
                if row is not None and max(row) < best_max:
                    best_row = list(row)
                    best_max = max(row)

        return best_row, best_max

    def local_polish(rows, max_passes=8, time_threshold=8):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time_left() > time_threshold and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < time_threshold:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row_v2(rows, i, used_other, max_attempts=80)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    def final_high_precision_polish(rows):
        """Final polish pass with tighter constraints and more iterations."""
        rows = [list(r) for r in rows]
        current_scope = max(max(r) for r in rows)

        # Try to reduce each row by 1 more entry with aggressive search
        for _ in range(3):
            if time_left() < 5:
                break
            improved = False
            for i in range(n_rows):
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])

                # Try to reduce by exactly 1
                target = max(rows[i]) - 1
                if target >= k:
                    for attempt in range(10):
                        row = build_row_smart(used_other, target, num_restarts=2)
                        if row is not None and max(row) <= target:
                            rows[i] = row
                            improved = True
                            break

            if not improved:
                break

        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)

        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])

            # Try smart rebuild with high randomness
            success = False
            for attempt in range(15):
                # Use a larger max_span for kicks to allow exploration
                max_span = max(rows[idx]) + 30 + attempt * 10
                row = build_row_smart(used_other, max_span, num_restarts=2)
                if row is not None:
                    rows[idx] = row
                    success = True
                    break

            if not success:
                # Fallback to greedy
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # Initial polish
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    while time_left() > 25:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        # 2. Kick
        if time_left() < 20:
            break

        kicked_rows = kick(best_rows)

        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)

        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final high-precision polish
    if time_left() > 10:
        polished = final_high_precision_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
