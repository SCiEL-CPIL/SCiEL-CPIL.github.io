import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 138.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

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

    def compute_all_diffs(rows):
        """Compute set of all positive differences across all rows."""
        seen = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    seen.add(row[j] - ri)
        return seen

    def validate_and_scope(rows):
        """Return (scope, all_diffs) if valid, else (None, None)."""
        seen = set()
        scope = 0
        for row in rows:
            n = len(row)
            row_max = 0
            for i in range(n):
                ri = row[i]
                if ri > row_max:
                    row_max = ri
                for j in range(i + 1, n):
                    diff = row[j] - ri
                    if diff in seen:
                        return None, None
                    seen.add(diff)
            if row_max > scope:
                scope = row_max
        return scope, seen

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope, best_diffs = validate_and_scope(best_rows)
    if best_scope is None:
        # Fallback: just return greedy
        return best_rows

    rng = random.Random(42)

    def targeted_descent(rows, budget):
        """Try to reduce scope by shrinking the largest mark. Returns (rows, scope) or (None, None)."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope, cur_diffs = validate_and_scope(cur)
        if cur_scope is None:
            return None, None
        improved = False

        while time.time() - start < budget:
            # Find rows with the current max value
            max_val = cur_scope
            max_rows = [r for r in range(n_rows) if cur[r][-1] == max_val]
            if not max_rows:
                # Find actual max
                max_val = max(max(r) for r in cur)
                max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                continue

            # Try candidates: prefer larger reductions
            candidates = set()
            for v in range(lower, min(lower + 20, old_val)):
                candidates.add(v)
            for _ in range(15):
                candidates.add(rng.randint(lower, old_val - 1))

            found = False
            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                new_scope, new_diffs = validate_and_scope(cur)
                if new_scope is not None and new_scope < cur_scope:
                    cur_scope = new_scope
                    cur_diffs = new_diffs
                    improved = True
                    found = True
                    break
                row[pos] = old_val

            if not found:
                # Try non-last element
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    break
                for _ in range(8):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    new_scope, new_diffs = validate_and_scope(cur)
                    if new_scope is not None and new_scope < cur_scope:
                        cur_scope = new_scope
                        cur_diffs = new_diffs
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break

        if improved:
            return cur, cur_scope
        return None, None

    # Simulated annealing with restarts
    max_restarts = 500
    steps_per_restart = 60000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb: 1-3 random moves
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        cur_scope, cur_diffs = validate_and_scope(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]
            best_diffs = cur_diffs

        # Annealing
        T = max(0.03, 0.8 - restart / max_restarts)
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 25
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            cur[ri][pos] = new_val
            new_scope, new_diffs = validate_and_scope(cur)

            if new_scope is not None:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                    best_diffs = new_diffs
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                    cur_diffs = new_diffs
                else:
                    cur[ri][pos] = old_val
            else:
                cur[ri][pos] = old_val

            T *= 0.99998

        # Targeted descent after each restart
        if time.time() < deadline - 3.0:
            desc_rows, desc_scope = targeted_descent(cur, budget=1.5)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
                best_diffs, _ = validate_and_scope(best_rows)

    return best_rows
