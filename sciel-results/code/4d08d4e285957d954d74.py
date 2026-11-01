"""Difference triangle set (4,7) - structural change to constraint-guided
multi-start local search with incremental validity checking and targeted
scope-reduction descent. Combines SA structure from current with targeted
descent from inspiration 1, plus a different initialisation and move
generation strategy."""
import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 142.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d
    row_len = k + 1

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(row_len):
                ri = row[i]
                for j in range(i + 1, row_len):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            if max(row) > scope:
                scope = max(row)
        return scope

    def greedy_init(seed_val=0):
        """Greedy construction with slight randomisation via seed_val offset."""
        rng_local = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                # Try x, but occasionally skip to add diversity
                if rng_local.random() < 0.15 and x < 200:
                    x += rng_local.randint(1, 3)
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Initial solutions: try several seeds, keep best
    rows = greedy_init(0)
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    for s in range(1, 8):
        r2 = greedy_init(s)
        sc = get_scope(r2)
        if sc < best_scope:
            best_scope = sc
            best_rows = [r[:] for r in r2]

    rng = random.Random(42)

    def targeted_descent(rows, budget):
        """Try to reduce the scope by shrinking marks in the row(s) with the max.
        Returns (improved_rows, new_scope) or (None, None)."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False

        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                # Try a non-last position in this row
                pos = rng.randint(1, len(row) - 2)
                old_val = row[pos]
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1
                if upper <= lower:
                    continue
                # Try to decrease
                candidates = set()
                for v in range(lower, min(lower + 12, old_val)):
                    candidates.add(v)
                for _ in range(8):
                    candidates.add(rng.randint(lower, old_val - 1))
                for new_val in sorted(candidates, key=lambda x: -x):
                    if new_val >= old_val:
                        continue
                    row[pos] = new_val
                    scope = full_validate_scope(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos] = old_val
                if not improved:
                    break
                continue

            # Try decreasing the last element
            candidates = set()
            for v in range(lower, min(lower + 15, old_val)):
                candidates.add(v)
            for _ in range(10):
                candidates.add(rng.randint(lower, old_val - 1))

            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate_scope(cur)
                if scope is not None and scope < cur_scope:
                    cur_scope = scope
                    improved = True
                    break
                row[pos] = old_val

            if not improved:
                break

        if improved:
            return cur, cur_scope
        return None, None

    # Simulated annealing with restarts
    max_restarts = 250
    steps_per_restart = 60000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbed
        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb: random row, random mark, random new value
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 10
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Annealing
        T = max(0.05, 1.0 - restart / max_restarts)
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 20
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            cur[ri][pos] = new_val
            new_scope = full_validate_scope(cur)

            if new_scope is not None:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    cur[ri][pos] = old_val
            else:
                # Invalid: reject
                cur[ri][pos] = old_val

            T *= 0.99995

        # After annealing, run a short targeted descent to try to squeeze scope further
        if time.time() < deadline - 4.0:
            desc_rows, desc_scope = targeted_descent(cur, budget=1.5)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    # Final targeted descent on best
    if time.time() < deadline - 1.0:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=2.0)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    if best_rows is None:
        best_rows = greedy_init(0)

    return best_rows
