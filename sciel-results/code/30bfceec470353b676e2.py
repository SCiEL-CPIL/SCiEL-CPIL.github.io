import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 140.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    # Simple counting bound
    m_min = 0
    while m_min * (m_min + 1) // 2 < total_diffs:
        m_min += 1
    lower_bound = m_min

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
        """Return True if all positive differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in seen:
                        return False
                    seen.add(diff)
        return True

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

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    # Simulated annealing with restarts
    max_restarts = 300
    steps_per_restart = 80000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbed
        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb
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
                cur[ri][pos] = old_val

            T *= 0.99995

    return best_rows
