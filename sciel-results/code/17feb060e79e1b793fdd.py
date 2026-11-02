"""Unified simulated-annealing / random-walk search for difference-triangle sets.

Instead of building rows independently with greedy + local compaction, we treat the
entire collection of rows as a single state and perform random-walk moves: pick a
random element (row r, position p>0), try moving it to a random nearby value, and
accept if the resulting state is valid and improves (or occasionally equals) the
current best scope. We seed with a few greedy solutions and run the walk until the
deadline, always tracking the best valid state found.
"""

import time
import random
import sys
from array import array


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1  # entries per row

    def valid_scope(rows):
        """Return (valid, scope, used_set) for a list of rows."""
        used = set()
        scope = 0
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False, None, None
            for i in range(m):
                for j in range(i + 1, m):
                    d = row[j] - row[i]
                    if d in used:
                        return False, None, None
                    used.add(d)
            if row[-1] > scope:
                scope = row[-1]
        return True, scope, used

    def greedy_seed(rng, mode=0):
        """Generate a greedy seed solution. mode varies the strategy."""
        u = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                if mode == 1 and rng.random() < 0.15:
                    continue  # random skip
                ok = True
                for a in row:
                    if (x - a) in u:
                        ok = False
                        break
                if ok:
                    for a in row:
                        u.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    # Generate initial seeds
    rng = random.Random(42)
    best_rows = None
    best_scope = float('inf')

    for mode in range(3):
        rows = greedy_seed(rng, mode)
        valid, scope, _ = valid_scope(rows)
        if valid and scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        # Fallback: plain deterministic greedy
        u = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in u:
                        ok = False
                        break
                if ok:
                    for a in row:
                        u.add(x - a)
                    row.append(x)
            rows.append(row)
        best_rows = rows
        best_scope = max(max(r) for r in best_rows)

    # Simulated annealing / random walk
    # State: current rows, current used set, current scope
    cur_rows = [list(r) for r in best_rows]
    cur_scope = best_scope
    cur_used = set()
    for row in cur_rows:
        for i in range(m):
            for j in range(i + 1, m):
                cur_used.add(row[j] - row[i])

    # Precompute all positions (r, p) where p > 0
    positions = [(r, p) for r in range(n_rows) for p in range(1, m)]

    step = 0
    no_improve = 0
    max_no_improve = 200000

    while time.time() < deadline:
        step += 1

        # Pick a random position
        r, p = positions[rng.randrange(len(positions))]
        row = cur_rows[r]
        old_val = row[p]

        # Choose a new value: biased toward smaller values to reduce scope
        # Range: (row[p-1], old_val] or sometimes larger
        low = row[p - 1] + 1
        # 70% of the time try to reduce, 30% try random nearby
        if rng.random() < 0.70:
            # Try a value in [low, old_val]
            if low >= old_val:
                continue
            new_val = rng.randint(low, old_val)
        else:
            # Random nearby: old_val +/- random(1, max(5, old_val//10))
            span = max(5, old_val // 10)
            new_val = old_val + rng.randint(-span, span)
            if new_val <= row[p - 1] or new_val >= row[p + 1] if p + 1 < m else new_val <= row[p - 1]:
                if p + 1 < m and new_val >= row[p + 1]:
                    continue
                if new_val <= row[p - 1]:
                    continue

        # Check if new_val equals old_val
        if new_val == old_val:
            continue

        # Remove old differences involving position p
        old_diffs = []
        for i in range(p):
            d = old_val - row[i]
            old_diffs.append((d, i))
        for j in range(p + 1, m):
            d = row[j] - old_val
            old_diffs.append((d, j))

        # Tentatively remove them
        for d, _ in old_diffs:
            cur_used.discard(d)

        # Check new differences
        new_diffs = []
        conflict = False
        for i in range(p):
            d = new_val - row[i]
            if d in cur_used:
                conflict = True
                break
            new_diffs.append(d)
        if not conflict:
            for j in range(p + 1, m):
                d = row[j] - new_val
                if d in cur_used:
                    conflict = True
                    break
                new_diffs.append(d)

        # Also check internal consistency: new differences must be distinct
        if not conflict and len(set(new_diffs)) != len(new_diffs):
            conflict = True

        if conflict:
            # Restore
            for d, _ in old_diffs:
                cur_used.add(d)
            continue

        # Apply: add new differences
        for d in new_diffs:
            cur_used.add(d)

        # Compute new scope
        new_row_max = max(row[-1], new_val) if p < m - 1 else new_val
        # Actually recompute scope properly
        new_scope = 0
        for rr in range(n_rows):
            if cur_rows[rr][-1] > new_scope:
                new_scope = cur_rows[rr][-1]
        # But we haven't updated cur_rows yet, so:
        if p == m - 1:
            # Last element changed
            if new_val > new_scope:
                new_scope = new_val
        else:
            # Not last element, scope unchanged unless it was the max
            pass

        # Actually, let's just compute scope from the modified row
        # The row's max is max of all elements
        row_max = 0
        for idx in range(m):
            if idx == p:
                val = new_val
            else:
                val = row[idx]
            if val > row_max:
                row_max = val

        # Global scope
        gscope = 0
        for rr in range(n_rows):
            if rr == r:
                rm = row_max
            else:
                rm = cur_rows[rr][-1]
            if rm > gscope:
                gscope = rm

        # Accept if improves or equal (with some probability for exploration)
        if gscope < cur_scope:
            # Accept improvement
            row[p] = new_val
            cur_scope = gscope
            no_improve = 0
            if cur_scope < best_scope:
                best_scope = cur_scope
                best_rows = [list(rr) for rr in cur_rows]
        elif gscope == cur_scope:
            # Accept with small probability to allow exploration
            if rng.random() < 0.3:
                row[p] = new_val
            else:
                # Revert
                for d, _ in old_diffs:
                    cur_used.add(d)
                for d in new_diffs:
                    cur_used.discard(d)
                no_improve += 1
        else:
            # Revert
            for d, _ in old_diffs:
                cur_used.add(d)
            for d in new_diffs:
                cur_used.discard(d)
            no_improve += 1

        if no_improve > max_no_improve:
            # Restart from best
            cur_rows = [list(rr) for rr in best_rows]
            cur_scope = best_scope
            cur_used = set()
            for row in cur_rows:
                for i in range(m):
                    for j in range(i + 1, m):
                        cur_used.add(row[j] - row[i])
            no_improve = 0

        # Periodic check
        if step % 5000 == 0:
            sys.stdout.flush()

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
