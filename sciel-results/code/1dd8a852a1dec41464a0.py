"""Search-based constructor for the difference-triangle-set task.

Strategy: generate many *varied* greedy candidates (different row orderings and
starting marks) and keep the best one found before an explicit deadline guard.
Each candidate is built by greedy set-cover: extend each row by the smallest next
mark whose pairwise differences are still unused. A lower bound on the final
mark lets us prune hopeless rows early.
"""

import time
import random


def construct(n_rows, k):
    if n_rows <= 0 or k <= 0:
        return []

    deadline = time.time() + 145.0
    rng = random.Random(12345)
    m = k + 1

    def build(order, starts, best_scope):
        used = set()
        rows = [None] * n_rows
        for ri in order:
            row = [starts[ri]]
            x = starts[ri]
            while len(row) < m:
                if time.time() > deadline:
                    return None
                x += 1
                # Lower bound check: smallest unused difference + max(row)
                # The smallest unused diff is at least 1, but we can compute it.
                # For efficiency, just check if x >= best_scope.
                if x >= best_scope:
                    return None
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows[ri] = row
        return rows

    # Initial deterministic greedy.
    best_rows = build(list(range(n_rows)), [0] * n_rows, 10**9)
    best_scope = max(max(r) for r in best_rows)

    # Random variations.
    while time.time() < deadline:
        order = list(range(n_rows))
        rng.shuffle(order)
        starts = [rng.randint(0, 1) for _ in range(n_rows)]
        rows = build(order, starts, best_scope)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows

    return best_rows
