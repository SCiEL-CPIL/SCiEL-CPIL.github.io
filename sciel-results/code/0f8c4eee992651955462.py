"""Difference triangle set (7,5) – improved greedy with lookahead & backtracking."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # For (7,5) each row has 6 entries -> 15 differences per row, 105 total.
    # We want to minimize the maximum entry (the "scope").

    # Strategy:
    # 1. Try to construct rows with a greedy that looks ahead.
    # 2. Use a priority on which differences to "reserve" – prefer small differences
    #    early to keep the row compact.
    # 3. If time permits, do local search / restarts with randomization to find
    #    a smaller scope.

    best_rows = None
    best_scope = float('inf')

    def elapsed():
        return time.time() - start_time

    def build_greedy(used, n_rows_remaining, k, randomize=False, max_scope=None):
        """
        Greedily build n_rows_remaining rows given current 'used' difference set.
        Returns (rows, used) or None if impossible within max_scope.
        """
        rows = []
        for _ in range(n_rows_remaining):
            row = [0]
            # Try to build the row
            x = 1
            max_x = max_scope if max_scope is not None else 200
            success = False
            while len(row) < k + 1:
                if x > max_x:
                    break
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    success = True
                x += 1
            if not success:
                return None
            rows.append(row)
        return rows, used

    def scope_of(rows):
        if not rows:
            return float('inf')
        return max(max(r) for r in rows)

    # --- Phase 1: Deterministic greedy (baseline) ---
    used = set()
    rows, used = build_greedy(used, n_rows, k)
    if rows:
        best_rows = rows
        best_scope = scope_of(rows)

    if elapsed() > deadline - 2.0:
        return best_rows

    # --- Phase 2: Greedy with "smallest unused difference" preference ---
    # Instead of just incrementing x, try to find the smallest x such that
    # all new differences are unused. This is what the above does, but let's
    # also try a variant where we prioritize rows that can be built with
    # smaller max values first.

    # Actually, let's try a different approach: build all rows simultaneously
    # using a "water-filling" style – at each step, pick the row with the
    # smallest current max and extend it.

    def build_interleaved(n_rows, k, randomize=False, max_scope=None):
        """
        Build rows interleaved: at each step, extend the row with the smallest
        current maximum. This tends to keep all rows balanced and compact.
        """
        rows = [[0] for _ in range(n_rows)]
        used = set()
        max_x = max_scope if max_scope is not None else 200

        for _ in range(n_rows * k):  # total number of additions
            # Find the row with smallest current max (and still needs more)
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            # Pick the one with smallest max
            i = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[i]
            x = row[-1] + 1
            found = False
            while x <= max_x:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                return None

        # Check all rows are complete
        for r in rows:
            if len(r) != k + 1:
                return None
        return rows, used

    rows, used = build_interleaved(n_rows, k)
    if rows:
        sc = scope_of(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = rows

    if elapsed() > deadline - 2.0:
        return best_rows

    # --- Phase 3: Local search / hill climbing ---
    # Try to reduce the scope by adjusting individual entries.

    def try_improve(rows, n_rows, k, time_limit):
        """
        Local search: try to reduce the max entry by moving individual points.
        """
        best = [r[:] for r in rows]
        best_scope = scope_of(best)
        used = set()
        for r in best:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])

        deadline_local = time.time() + time_limit
        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            # Try to move each point in each row to a smaller position
            for ri in range(n_rows):
                row = best[ri]
                for pi in range(1, len(row)):  # skip 0
                    old_val = row[pi]
                    # Try smaller values
                    for new_val in range(old_val - 1, 0, -1):
                        # Check if moving row[pi] to new_val is valid
                        # Need to remove old diffs and add new diffs
                        old_diffs = {old_val - a for a in row[:pi]} | {b - old_val for b in row[pi+1:]}
                        new_diffs = {new_val - a for a in row[:pi]} | {b - new_val for b in row[pi+1:]}
                        
                        # Also need to ensure new_val > row[pi-1] and new_val < row[pi+1] if exists
                        if pi > 0 and new_val <= row[pi - 1]:
                            continue
                        if pi < len(row) - 1 and new_val >= row[pi + 1]:
                            continue
                        if new_val <= 0:
                            continue

                        # Check no conflicts
                        # Remove old_diffs from used, check new_diffs don't conflict with remaining
                        remaining = used - old_diffs
                        if not new_diffs & remaining:
                            # Also ensure new_diffs don't include 0
                            if 0 in new_diffs:
                                continue
                            # Valid move
                            used = remaining | new_diffs
                            row[pi] = new_val
                            new_scope = scope_of(best)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                improved = True
                            break  # Move to the smallest valid new_val

        return best, best_scope

    if best_rows is not None:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 1.0:
            improved_rows, improved_scope = try_improve(best_rows, n_rows, k, remaining_time)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows

    if elapsed() > deadline - 2.0:
        return best_rows

    # --- Phase 4: Randomized restarts with interleaved construction ---
    # Try different random orderings of rows to see if we get a better scope.
    rng = random.Random(42)
    num_restarts = 50
    for restart in range(num_restarts):
        if elapsed() > deadline - 2.0:
            break

        # Randomly shuffle the order in which we build rows
        order = list(range(n_rows))
        rng.shuffle(order)

        rows = [[0] for _ in range(n_rows)]
        used = set()
        max_x = best_scope - 1 if best_scope < float('inf') else 200
        success = True

        # Build in shuffled order, but still use interleaved logic
        # Actually, let's just do a randomized greedy
        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            # Pick randomly from candidates, weighted toward smaller max
            weights = [1.0 / (rows[i][-1] + 1) for i in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for idx, i in enumerate(candidates):
                cum += weights[idx]
                if r <= cum:
                    chosen = i
                    break
            row = rows[chosen]
            x = row[-1] + 1
            found = False
            while x <= max_x:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                success = False
                break

        if success and all(len(r) == k + 1 for r in rows):
            sc = scope_of(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = rows

    return best_rows
