import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Total number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    def scope_of(rows):
        return max(max(r) for r in rows)

    def valid(rows):
        """Check all positive differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_construct(rng=None, randomize=False):
        """Greedy construction with optional randomization in candidate ordering."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Try x; if randomize, sometimes skip to a larger value
                if randomize and rng is not None and rng.random() < 0.15:
                    skip = rng.randint(1, 3)
                    x += skip
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def local_search(rows, time_limit=5.0):
        """Local search via single-entry moves that preserve validity and reduce scope."""
        rows = [list(r) for r in rows]
        best_scope = scope_of(rows)
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])

        ls_start = time.time()
        improved = True
        while improved and (time.time() - ls_start) < time_limit:
            improved = False
            # Try to reduce each entry (except index 0 which is fixed at 0)
            for ri in range(n_rows):
                row = rows[ri]
                for idx in range(1, len(row)):
                    old_val = row[idx]
                    # Try decreasing the value
                    for new_val in range(old_val - 1, -1, -1):
                        # Must maintain strictly increasing
                        if idx > 0 and new_val <= row[idx - 1]:
                            break
                        if idx < len(row) - 1 and new_val >= row[idx + 1]:
                            continue
                        # Check differences with other entries in this row
                        row_diffs_old = set()
                        for j in range(len(row)):
                            if j != idx:
                                row_diffs_old.add(abs(row[idx] - row[j]))
                        row_diffs_new = set()
                        for j in range(len(row)):
                            if j != idx:
                                row_diffs_new.add(abs(new_val - row[j]))
                        # Differences that are removed
                        removed = row_diffs_old - row_diffs_new
                        # Differences that are added
                        added = row_diffs_new - row_diffs_old
                        # Added diffs must not conflict with global used (minus removed)
                        global_available = used - removed
                        if added & global_available:
                            continue
                        # Also check within-row: new diffs must not duplicate existing
                        # (already handled by row_diffs_new being a set, but check against
                        # other diffs in the row that remain)
                        # Actually we need to check: for the new value, all pairwise diffs
                        # must be unique within the row
                        row_copy = list(row)
                        row_copy[idx] = new_val
                        row_ok = True
                        seen_row = set()
                        for a in range(len(row_copy)):
                            for b in range(a + 1, len(row_copy)):
                                d = row_copy[b] - row_copy[a]
                                if d in seen_row:
                                    row_ok = False
                                    break
                                seen_row.add(d)
                        if not row_ok:
                            continue
                        # Valid move! Apply it
                        used = (used - removed) | added
                        row[idx] = new_val
                        best_scope = min(best_scope, scope_of(rows))
                        improved = True
                        break  # restart from this row
                    if improved:
                        break
                if improved:
                    break
        return rows, best_scope

    def randomized_greedy_with_refinement(rng, time_budget):
        """Generate a randomized greedy solution and refine it."""
        rows = greedy_construct(rng=rng, randomize=True)
        if not valid(rows):
            return None, scope_of(rows)
        refined, scope = local_search(rows, time_limit=min(3.0, max(0.5, time_budget / 3)))
        return refined, scope

    # Phase 1: Pure greedy baseline
    baseline_rows = greedy_construct(randomize=False)
    if not valid(baseline_rows):
        # Fallback: should not happen
        return baseline_rows
    best_rows = [list(r) for r in baseline_rows]
    best_scope = scope_of(best_rows)

    # Phase 2: Local search on baseline
    refined, scope = local_search(baseline_rows, time_limit=5.0)
    if scope < best_scope:
        best_rows = refined
        best_scope = scope

    # Phase 3: Randomized restarts
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        # Time budget for this attempt
        time_budget = min(remaining - 1.0, 8.0)
        rows, scope = randomized_greedy_with_refinement(rng, time_budget)
        if rows is not None and scope < best_scope:
            best_rows = rows
            best_scope = scope
            # Try to further refine the best
            if time.time() < deadline - 3.0:
                refined2, scope2 = local_search(best_rows, time_limit=min(3.0, deadline - time.time() - 2.0))
                if scope2 < best_scope:
                    best_rows = refined2
                    best_scope = scope2

        # Check if we've hit the counting bound
        if best_scope <= 112:
            break

    return best_rows
# EVOLVE-BLOCK-END
