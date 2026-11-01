import time
import random
import sys

_START = None


def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0  # safety margin before 150s kill

    rng = random.Random(42)
    d = k * (k + 1) // 2  # differences per row
    n_marks = k + 1

    # ---------- fast incremental validity ----------
    def validate(rows):
        """Return (scope, used) if all positive differences distinct, else (None, None)."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(n_marks):
                a = row[i]
                for j in range(i + 1, n_marks):
                    diff = row[j] - a
                    if diff in seen:
                        return None, None
                    seen.add(diff)
            m = row[-1]
            if m > scope:
                scope = m
        return scope, seen

    def row_diffs(row):
        s = set()
        for i in range(n_marks):
            a = row[i]
            for j in range(i + 1, n_marks):
                s.add(row[j] - a)
        return s

    def row_diffs_delta(old, new, pos):
        """Return (removed, added) difference sets when row[pos] changes old->new."""
        removed = set()
        added = set()
        for i in range(n_marks):
            if i == pos:
                continue
            if i < pos:
                removed.add(old - row[i])
                added.add(new - row[i])
            else:
                removed.add(row[i] - old)
                added.add(row[i] - new)
        return removed, added

    # ---------- initial solution: greedy ----------
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_marks:
                x += 1
                ok = True
                for a in row:
                    if x - a in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row:
                        if a != x:
                            used.add(x - a)
            rows.append(row)
        return rows

    rows = greedy_init()
    best_scope, _ = validate(rows)
    best_rows = [r[:] for r in rows]

    # ---------- local search over valid states ----------
    def search():
        nonlocal best_scope, best_rows

        cur = [r[:] for r in best_rows]
        cur_scope, cur_used = validate(cur)
        if cur_scope is None:
            return
        best_local = cur_scope
        best_local_rows = [r[:] for r in cur]

        # temperature schedule
        T = 1.5
        T_floor = 0.05
        T_decay = 0.99998

        while time.time() < deadline:
            # pick a random row and a random non-first mark to move
            ri = rng.randrange(n_rows)
            pos = rng.randint(1, n_marks - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < n_marks - 1 else best_scope + 30
            if hi <= lo:
                T *= T_decay
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            # incremental validity check
            row = cur[ri]
            removed = set()
            added = set()
            for i in range(n_marks):
                if i == pos:
                    continue
                if i < pos:
                    removed.add(old_val - row[i])
                    added.add(new_val - row[i])
                else:
                    removed.add(row[i] - old_val)
                    added.add(row[i] - new_val)

            # check no added diff already in cur_used (and not in removed)
            conflict = False
            for diff in added:
                if diff in cur_used and diff not in removed:
                    conflict = True
                    break
            if conflict:
                # invalid move, reject (but still cool)
                T *= T_decay
                continue

            # valid move: compute new scope
            new_scope = cur_scope
            # only the row's max can change scope
            if pos == n_marks - 1:
                # last element changed
                new_scope = max(new_scope, new_val)
                # other rows unchanged
            else:
                # recompute this row's max (unchanged unless pos was last, already handled)
                pass
            # Actually scope = max over all rows of their last element.
            # Only row ri changed, and only if pos==last does its max change.
            if pos == n_marks - 1:
                new_scope = max(max(r[-1] for r in cur if r is not row), new_val)
            # else new_scope == cur_scope

            delta = new_scope - cur_scope
            accept = delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01)))

            if accept:
                # commit
                row[pos] = new_val
                cur_used -= removed
                cur_used |= added
                cur_scope = new_scope
                if new_scope < best_local:
                    best_local = new_scope
                    best_local_rows = [r[:] for r in cur]
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
            else:
                pass  # no change

            T *= T_decay
            if T < T_floor:
                T = T_floor

        # finalize
        if best_local < best_scope:
            best_scope = best_local
            best_rows = best_local_rows

    # ---------- targeted descent: shrink the max mark ----------
    def targeted_descent(budget):
        nonlocal best_scope, best_rows
        start = time.time()
        cur = [r[:] for r in best_rows]
        cur_scope, cur_used = validate(cur)
        if cur_scope is None:
            return
        best_local = cur_scope
        best_local_rows = [r[:] for r in cur]

        while time.time() - start < budget:
            max_val = max(r[-1] for r in cur)
            # find rows achieving the max
            max_rows = [r for r in range(n_rows) if cur[r][-1] == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            # try to shrink the last element
            pos = n_marks - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if old_val <= lower:
                continue
            # candidate values: near the bottom for max reduction
            tried = 0
            for new_val in range(lower, min(lower + 20, old_val)):
                if time.time() - start > budget:
                    break
                removed = set()
                added = set()
                for i in range(n_marks):
                    if i == pos:
                        continue
                    removed.add(row[i] - old_val)
                    added.add(row[i] - new_val)
                conflict = False
                for diff in added:
                    if diff in cur_used and diff not in removed:
                        conflict = True
                        break
                if conflict:
                    continue
                # valid: new scope
                new_scope = max(max(rr[-1] for rr in cur if rr is not row), new_val)
                if new_scope < best_local:
                    row[pos] = new_val
                    cur_used -= removed
                    cur_used |= added
                    cur_scope = new_scope
                    best_local = new_scope
                    best_local_rows = [r[:] for r in cur]
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in cur]
            # also try shrinking a middle mark
            pos2 = rng.randint(1, n_marks - 2)
            old2 = row[pos2]
            lo2 = row[pos2 - 1] + 1
            hi2 = row[pos2 + 1] - 1
            if hi2 > lo2:
                for _ in range(8):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    removed = set()
                    added = set()
                    for i in range(n_marks):
                        if i == pos2:
                            continue
                        if i < pos2:
                            removed.add(old2 - row[i])
                            added.add(nv - row[i])
                        else:
                            removed.add(row[i] - old2)
                            added.add(row[i] - nv)
                    conflict = False
                    for diff in added:
                        if diff in cur_used and diff not in removed:
                            conflict = True
                            break
                    if conflict:
                        continue
                    new_scope = cur_scope  # last element unchanged
                    if new_scope < best_local:
                        row[pos2] = nv
                        cur_used -= removed
                        cur_used |= added
                        cur_scope = new_scope
                        best_local = new_scope
                        best_local_rows = [r[:] for r in cur]
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [r[:] for r in cur]

        if best_local < best_scope:
            best_scope = best_local
            best_rows = best_local_rows

    # ---------- main loop: alternate SA and targeted descent ----------
    # Run SA for a chunk, then targeted descent, repeat
    chunk = 15.0
    while time.time() < deadline - 3.0:
        # SA chunk
        search()
        if time.time() < deadline - 5.0:
            targeted_descent(budget=min(4.0, max(1.0, deadline - time.time() - 2.0)))

    # final safety: ensure best_rows is valid
    if best_rows is None:
        best_rows = greedy_init()
    else:
        sc, _ = validate(best_rows)
        if sc is None:
            best_rows = greedy_init()

    return best_rows
