"""Difference triangle set (4,7) solver: structured ILS with fast delta evaluation.

Combines:
  - Fast conflict-delta bookkeeping (from current program) so row swaps are O(row_diffs).
  - Structured per-row local search with lookahead + jitter (from inspiration) that
    specifically drives scope down, plus ILS kicks to escape local minima.
"""
import time
import random
import sys


def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin under 150s kill

    size = k + 1
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]

    def get_diffs(row):
        return [row[j] - row[i] for i, j in pairs]

    def get_diffs_set(row):
        s = set()
        for i, j in pairs:
            s.add(row[j] - row[i])
        return s

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    def get_scope(rows):
        return max(max(r) for r in rows)

    # ---------- Greedy baseline ----------
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < size:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # ---------- Fast delta bookkeeping ----------
    # diff_counts[d] = number of rows currently using difference d.
    # total_conflicts = number of d with diff_counts[d] >= 2.
    max_diff_est = 400
    diff_counts = [0] * (max_diff_est + 1)

    def ensure_size(d):
        nonlocal diff_counts
        if d >= len(diff_counts):
            diff_counts.extend([0] * (d - len(diff_counts) + 1))

    total_conflicts = 0

    def apply_remove(row):
        nonlocal total_conflicts
        for d in get_diffs(row):
            if diff_counts[d] == 1:
                total_conflicts -= 1
            diff_counts[d] -= 1

    def apply_add(row):
        nonlocal total_conflicts
        for d in get_diffs(row):
            ensure_size(d)
            if diff_counts[d] == 1:
                total_conflicts += 1
            diff_counts[d] += 1

    def swap_row(rows, idx, new_row):
        """Replace rows[idx] with new_row, updating diff_counts and total_conflicts.
        Returns (old_row, old_scope, new_scope)."""
        nonlocal total_conflicts
        old_row = rows[idx]
        apply_remove(old_row)
        apply_add(new_row)
        rows[idx] = new_row
        new_scope = get_scope(rows)
        old_scope = max(old_row)
        return old_row, old_scope, new_scope

    def revert_swap(rows, idx, old_row):
        """Undo a swap: remove current rows[idx], re-add old_row."""
        nonlocal total_conflicts
        apply_remove(rows[idx])
        apply_add(old_row)
        rows[idx] = old_row

    # ---------- Row generation ----------
    def generate_row(used_other_set, max_val_hint, jitter_prob=0.0, max_x=None):
        """Greedy row construction avoiding used_other_set and internal conflicts.
        Returns a list of size elements starting at 0, or None on failure."""
        if max_x is None:
            max_x = max_val_hint + 60
        row = [0]
        x = 0
        internal = set()
        while len(row) < size:
            found = False
            # Lookahead window
            for trial_x in range(x + 1, min(x + 90, max_x + 1)):
                new = {trial_x - a for a in row}
                if (new & used_other_set) or (new & internal):
                    continue
                if jitter_prob > 0 and random.random() < jitter_prob and trial_x < max_x - 10:
                    continue
                row.append(trial_x)
                internal |= new
                x = trial_x
                found = True
                break
            if not found:
                # Widen window
                for trial_x in range(x + 1, max_x + 1):
                    new = {trial_x - a for a in row}
                    if (new & used_other_set) or (new & internal):
                        continue
                    if jitter_prob > 0 and random.random() < jitter_prob and trial_x < max_x - 10:
                        continue
                    row.append(trial_x)
                    internal |= new
                    x = trial_x
                    found = True
                    break
            if not found:
                return None
        return row

    def used_other(rows, idx):
        s = set()
        for i in range(n_rows):
            if i != idx:
                s |= get_diffs_set(rows[i])
        return s

    # ---------- Per-row local search: try to reduce scope of one row ----------
    def try_reduce_row_scope(rows, idx, budget=0.5):
        """Try multiple randomized reconstructions of row idx to find one with
        smaller max entry (and no new conflicts). Returns True if improved."""
        nonlocal total_conflicts
        start = time.time()
        current_row = rows[idx]
        current_max = max(current_row)
        used_others = used_other(rows, idx)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(200):
            if time.time() - start > budget or time.time() > deadline - 5:
                break
            # Jitter increases with attempt to explore
            jitter = 0.0 if attempt < 10 else min(0.3, 0.05 + attempt * 0.005)
            # Target: try to beat current_max
            hint = current_max - 1 if attempt < 50 else current_max + 10
            new_row = generate_row(used_others, hint, jitter_prob=jitter,
                                   max_x=current_max + 30)
            if new_row is None:
                continue
            new_max = max(new_row)
            if new_max >= best_max:
                continue
            # Check: does this row introduce any conflict?
            # Since used_others is fixed and generate_row avoided it, and internal
            # diffs are distinct by construction, the row is conflict-free.
            best_row = new_row
            best_max = new_max

        if best_max < current_max:
            # Apply the swap
            old_row = rows[idx]
            apply_remove(old_row)
            apply_add(best_row)
            rows[idx] = best_row
            return True
        return False

    # ---------- Local polish: iterate rows in scope order ----------
    def local_polish(rows, max_passes=8):
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 5:
            improved = False
            passes += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 5:
                    break
                if try_reduce_row_scope(rows, i, budget=0.4):
                    improved = True
        return rows

    # ---------- Kick: perturb to escape local minima ----------
    def kick(rows):
        nonlocal total_conflicts
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_others = used_other(rows, idx)
            # Rebuild with high jitter
            new_row = None
            for attempt in range(30):
                jitter = 0.35 + 0.02 * attempt
                hint = max(rows[idx]) + 15
                new_row = generate_row(used_others, hint, jitter_prob=jitter,
                                       max_x=hint + 40)
                if new_row is not None:
                    break
            if new_row is None:
                # Fallback: greedy
                new_row = generate_row(used_others, max(rows[idx]) + 50,
                                       jitter_prob=0.0, max_x=max(rows[idx]) + 100)
            if new_row is not None:
                old_row = rows[idx]
                apply_remove(old_row)
                apply_add(new_row)
                rows[idx] = new_row
        return rows

    # ---------- Main search ----------
    best_rows = greedy_construct()
    # Initialize diff_counts
    for r in best_rows:
        apply_add(r)

    best_scope = get_scope(best_rows)
    best_rows_copy = [r[:] for r in best_rows]

    # Initial polish
    if time.time() < deadline - 10:
        polished = [r[:] for r in best_rows]
        # We need to track state for polished separately; simpler: polish in place
        # and keep best.
        local_polish(best_rows)
        cur_scope = get_scope(best_rows)
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows_copy = [r[:] for r in best_rows]

    # ILS loop
    while time.time() < deadline - 10:
        # Kick
        kicked = [r[:] for r in best_rows]
        kick(kicked)

        # Polish the kicked solution
        local_polish(kicked)

        cur_scope = get_scope(kicked)
        if cur_scope < best_scope and is_valid(kicked):
            best_scope = cur_scope
            best_rows_copy = [r[:] for r in kicked]
            # Sync diff_counts to best state
            # Reset diff_counts
            for i in range(len(diff_counts)):
                diff_counts[i] = 0
            total_conflicts = 0
            for r in best_rows_copy:
                apply_add(r)
            # Update working rows
            for i in range(n_rows):
                best_rows[i] = best_rows_copy[i][:]

        # Early exit at theoretical lower bound
        if best_scope <= 112:
            break

    # Final validation
    if is_valid(best_rows_copy):
        return best_rows_copy
    return best_rows


if __name__ == "__main__":
    rows = construct(4, 7)
    for r in rows:
        print(r)
    diffs = []
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.append(r[j] - r[i])
    print("Valid:", len(diffs) == len(set(diffs)))
    print("Scope:", max(max(r) for r in rows))
