import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 112.0  # Safety margin

    # Each row has k*(k+1)/2 differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Greedy fallback
    def greedy_init():
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

    def validate(rows):
        """Check if rows are valid. Return scope if valid, None if not."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return None
                    seen.add(d)
            scope = max(scope, max(row))
        return scope

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = validate(rows) or max(max(r) for r in rows)

    rng = random.Random(12345)

    # --- Core: Backtracking solver with MRV ---
    # We build rows one element at a time.
    # State: for each row, the list of chosen elements (starts with [0])
    # and the set of used differences.
    # MRV: choose the row with the fewest valid next values.
    # Value ordering: prefer smaller values (tighter packing).

    def backtrack_solve(scope_limit, time_budget, rng_seed):
        """
        Try to construct n_rows rows of k+1 elements (starting at 0)
        with all differences distinct and max element <= scope_limit.
        Returns rows if found, None otherwise.
        """
        r = random.Random(rng_seed)
        start_t = time.time()
        # We'll track: rows[i] = list of chosen elements for row i (always starts with [0])
        # used_diffs = global set of used differences
        # For MRV: for each row i, compute the set of valid next values
        #   given current row[i] elements and used_diffs.

        rows_state = [[0] for _ in range(n_rows)]
        used_diffs = set()

        # We need to decide which row to extend next and what value to add.
        # A row is "complete" when len(rows_state[i]) == k+1.
        # We pick the incomplete row with fewest valid candidates (MRV).

        def get_candidates(row_idx):
            """Return sorted list of valid next values for row_idx."""
            row = rows_state[row_idx]
            last = row[-1]
            cands = []
            # Next value must be > last and <= scope_limit
            # For each candidate x, the new differences {x - a for a in row} must not intersect used_diffs
            # Also x - a > 0 for all a in row (guaranteed since x > last >= all a)
            lo = last + 1
            hi = scope_limit
            # Optimization: if row has many elements, the constraints are tighter
            for x in range(lo, hi + 1):
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    cands.append(x)
            return cands

        # We'll use a simple backtracking with MRV.
        # To avoid exponential blowup, we add:
        #  - Randomization in value ordering (not always smallest first)
        #  - A node limit to prevent infinite loops
        #  - Forward checking (implicit in candidate computation)

        node_count = [0]
        node_limit = 500000  # Max nodes to explore

        def solve(depth):
            """
            depth = number of elements placed so far (total across all rows, minus n_rows for the initial 0s)
            Actually, let's track: total elements placed = sum(len(rows_state[i]) for i)
            We need total == n_rows * (k+1) to be done.
            """
            if time.time() - start_t > time_budget:
                return False
            node_count[0] += 1
            if node_count[0] > node_limit:
                return False

            # Find incomplete rows
            incomplete = [i for i in range(n_rows) if len(rows_state[i]) < k + 1]
            if not incomplete:
                return True  # All rows complete!

            # MRV: pick row with fewest candidates
            best_row = -1
            best_cands = None
            best_count = float('inf')
            for i in incomplete:
                cands = get_candidates(i)
                if len(cands) == 0:
                    return False  # Dead end
                if len(cands) < best_count:
                    best_count = len(cands)
                    best_row = i
                    best_cands = cands
                    if best_count == 1:
                        break  # Can't do better than 1

            if best_row == -1:
                return False

            # Value ordering: with some probability, pick from the smaller values first
            # Otherwise, randomize to explore different structures
            if r.random() < 0.7:
                # Prefer smaller values (tighter)
                order = best_cands[:min(8, len(best_cands))]
                if len(best_cands) > 8:
                    # Add a few random ones from the rest
                    rest = best_cands[8:]
                    order.extend(r.sample(rest, min(4, len(rest))))
            else:
                order = best_cands[:]
                r.shuffle(order)

            for x in order:
                # Place x
                rows_state[best_row].append(x)
                new_diffs = set()
                for a in rows_state[best_row]:
                    if a == x:
                        continue
                    d = x - a
                    new_diffs.add(d)
                # Check no conflict (should be guaranteed by get_candidates, but safety)
                if new_diffs & used_diffs:
                    rows_state[best_row].pop()
                    continue
                used_diffs |= new_diffs

                if solve(depth + 1):
                    return True

                # Undo
                used_diffs -= new_diffs
                rows_state[best_row].pop()

            return False

        if solve(0):
            return rows_state
        return None

    # --- Main search loop ---
    # Strategy:
    # 1. Start with greedy solution
    # 2. Try backtracking with decreasing scope limits
    # 3. Also do local search on the greedy result as a fallback

    # Local search (simple) to improve greedy
    def local_search(rows, budget, seed):
        r = random.Random(seed)
        start = time.time()
        cur = [row[:] for row in rows]
        cur_scope = validate(cur)
        if cur_scope is None:
            cur_scope = max(max(row) for row in cur)
        best = [row[:] for row in cur]
        best_sc = cur_scope

        while time.time() - start < budget:
            # Pick a random row and position
            ri = r.randrange(n_rows)
            pos = r.randint(1, len(cur[ri]) - 1)
            old = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_sc + 10
            if hi <= lo:
                continue
            new_val = r.randint(lo, hi)
            if new_val == old:
                continue
            cur[ri][pos] = new_val
            sc = validate(cur)
            if sc is not None:
                if sc < best_sc:
                    best_sc = sc
                    best = [row[:] for row in cur]
                # Accept with probability based on improvement
                delta = sc - cur_scope
                if delta <= 0 or r.random() < 0.5:
                    cur_scope = sc
                else:
                    cur[ri][pos] = old
            else:
                cur[ri][pos] = old

        return best, best_sc

    # First: try backtracking from the greedy scope downward
    current_scope = best_scope
    max_backtracks = 30
    backtrack_time_each = 3.0

    for attempt in range(max_backtracts):
        if time.time() > deadline - 30.0:
            break
        # Try to find a solution with scope = current_scope - 1
        target = current_scope - 1
        if target < total_diffs:  # Can't be less than total number of diffs
            break
        result = backtrack_solve(target, backtrack_time_each, 1000 + attempt * 7)
        if result is not None:
            sc = validate(result)
            if sc is not None and sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in result]
                current_scope = sc
            # Try to go lower
        else:
            # Couldn't find one with target; try local search on current best
            if time.time() < deadline - 15.0:
                ls_rows, ls_scope = local_search(best_rows, 5.0, 5000 + attempt * 13)
                if ls_scope < best_scope:
                    best_scope = ls_scope
                    best_rows = [r[:] for r in ls_rows]
                    current_scope = best_scope

    # Final: more local search if time remains
    if time.time() < deadline - 10.0:
        ls_rows, ls_scope = local_search(best_rows, deadline - time.time() - 12.0, 9999)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = [r[:] for r in ls_rows]

    return best_rows
