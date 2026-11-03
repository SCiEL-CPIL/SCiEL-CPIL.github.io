"""Difference triangle set (7,5): randomized greedy + local search with time guard."""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 110.0  # safety margin under 120s limit
    best_rows = None
    best_scope = float('inf')
    rng = random.Random(12345)

    def row_diffs(row):
        return frozenset(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

    def max_entry(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        seen = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_row(used, k, rng, randomize, max_try):
        """Build one row starting at 0 with k+1 entries, differences disjoint from used."""
        row = [0]
        while len(row) < k + 1:
            # find all candidate x such that new diffs don't collide
            # search in a window to allow randomization
            if randomize and rng.random() < 0.5:
                # try a random offset within a small window
                base = max(row) + 1
                window = max_try
                candidates = []
                for x in range(base, base + window):
                    new = set()
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used or d in new:
                            ok = False
                            break
                        new.add(d)
                    if ok:
                        candidates.append(x)
                if candidates:
                    x = rng.choice(candidates)
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    continue
            # deterministic smallest
            x = max(row) + 1
            while True:
                new = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in used or d in new:
                        ok = False
                        break
                    new.add(d)
                if ok:
                    break
                x += 1
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
        return row

    def try_construct(rng, randomize, max_try):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, k, rng, randomize, max_try)
            rows.append(row)
        return rows

    # Initial deterministic greedy
    rows = try_construct(rng, randomize=False, max_try=1)
    if is_valid(rows):
        best_rows = rows
        best_scope = max_entry(rows)

    # Randomized restarts
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        randomize = rng.random() < 0.7
        max_try = rng.randint(3, 12)
        rows = try_construct(rng, randomize, max_try)
        if is_valid(rows):
            scope = max_entry(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows

        # Local search: try to shrink the largest row by rebuilding it
        if best_rows is not None and time.time() < deadline:
            # pick the row with the largest entry, try to rebuild it tighter
            idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
            orig_row = best_rows[idx]
            # remove its diffs from used
            used = set()
            for i, r in enumerate(best_rows):
                if i != idx:
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            used.add(r[b] - r[a])
            # try several random rebuilds of row idx
            for _ in range(5):
                new_row = greedy_row(used, k, rng, randomize=True, max_try=rng.randint(3, 15))
                if max(new_row) < max(orig_row):
                    candidate = list(best_rows)
                    candidate[idx] = new_row
                    if is_valid(candidate):
                        best_rows = candidate
                        best_scope = max_entry(best_rows)
                        orig_row = new_row
                        # recompute used for further tries
                        used = set()
                        for i, r in enumerate(best_rows):
                            if i != idx:
                                for a in range(len(r)):
                                    for b in range(a + 1, len(r)):
                                        used.add(r[b] - r[a])

    if best_rows is None:
        # fallback: deterministic greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            best_rows.append(greedy_row(used, k, rng, randomize=False, max_try=1))
    return best_rows
