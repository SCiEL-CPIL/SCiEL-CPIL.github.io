"""Difference triangle set (4,7): randomized greedy + local search rewrite."""
import time
import random

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 145.0

    def get_scope(rows):
        return max(max(r) for r in rows)

    def build_row_greedy(used_diffs, k, rng, jitter=0.0, cap=None):
        """
        Build a single row greedily. Returns (row, new_diffs_added) or (None, None).
        cap: if set, max entry must be <= cap.
        """
        row = [0]
        local_diffs = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None, None
            new_diffs = frozenset(x - a for a in row)
            if new_diffs & used_diffs or new_diffs & local_diffs:
                continue
            if jitter > 0 and rng.random() < jitter:
                continue
            row.append(x)
            local_diffs |= new_diffs
        return row, local_diffs

    def build_all(used_diffs, n_rows, k, rng, jitter=0.0):
        rows = []
        cur_used = set(used_diffs)
        for _ in range(n_rows):
            row, diffs = build_row_greedy(cur_used, k, rng, jitter)
            if row is None:
                return None
            rows.append(row)
            cur_used |= diffs
        return rows

    def local_search(rows, rng, max_iters=300):
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)

        for _ in range(max_iters):
            if time.time() > deadline - 8:
                break
            idx = rng.randrange(n_rows)
            old_row = best_rows[idx]
            old_max = max(old_row)
            if old_max <= k:
                continue

            # Compute diffs from all other rows
            other_diffs = set()
            for i, r in enumerate(best_rows):
                if i == idx:
                    continue
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        other_diffs.add(r[b] - r[a])

            # Try to rebuild with smaller cap
            target = old_max - 1
            found = None
            for _attempt in range(80):
                row, diffs = build_row_greedy(other_diffs, k, rng, jitter=0.0, cap=target)
                if row is not None:
                    found = row
                    break

            if found is not None:
                best_rows[idx] = found
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope

        return best_rows, best_scope

    rng = random.Random(12345)
    best_rows = None
    best_scope = float('inf')

    # Phase 1: many randomized greedy restarts
    for i in range(800):
        if time.time() > deadline - 25:
            break
        jitter = rng.random() * 0.4
        rows = build_all(set(), n_rows, k, rng, jitter)
        if rows is None:
            continue
        s = get_scope(rows)
        if s < best_scope:
            best_scope = s
            best_rows = [r[:] for r in rows]
            # Local search on improvement
            if time.time() < deadline - 12 and s < best_scope + 5:
                lr, ls = local_search(rows, rng, max_iters=400)
                if ls < best_scope:
                    best_scope = ls
                    best_rows = lr

    # Phase 2: targeted local search on best solution
    if best_rows is not None and time.time() < deadline - 10:
        lr, ls = local_search(best_rows, rng, max_iters=2000)
        if ls < best_scope:
            best_scope = ls
            best_rows = lr

    # Fallback
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = set(x - a for a in row)
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
        best_rows = rows

    return best_rows

# EVOLVE-BLOCK-END
