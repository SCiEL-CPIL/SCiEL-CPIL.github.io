"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _greedy_ruler(k, used):
    """Build one greedy ruler given the set of already-used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row


def _greedy_set(n_rows, k, seed=0):
    """Greedy construction; optionally randomize the order of candidate marks via a seeded
    tie-break to explore different greedy solutions."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try a small window of candidates to allow some non-determinism
            best_x = None
            # Standard greedy: smallest x
            while True:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    best_x = x
                    break
            # With a small probability, try to skip a few to diversify
            if seed and rng.random() < 0.15:
                # try up to 5 alternative x's, pick a random feasible one
                candidates = []
                y = best_x
                for _ in range(6):
                    new = {y - a for a in row}
                    if not new & used:
                        candidates.append(y)
                    y += 1
                if candidates:
                    best_x = rng.choice(candidates)
            row.append(best_x)
            used |= {best_x - a for a in row[:-1]}
        rows.append(row)
    return rows


def _score(rows):
    """Return the largest entry across rows (the scope)."""
    return max(r[-1] for r in rows)


def _local_search(rows, k, max_iter=200, seed=0):
    """
    Local search: try to reduce the scope by lowering the last element of a row
    (and adjusting subsequent elements) while maintaining validity.
    """
    rng = random.Random(seed)
    best = [r[:] for r in rows]
    best_scope = _score(best)
    used = set()
    for r in best:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                used.add(r[j] - r[i])

    for _ in range(max_iter):
        improved = False
        # pick a random row
        ri = rng.randrange(len(best))
        row = best[ri]
        # try to reduce the last element
        if len(row) < 2:
            continue
        last = row[-1]
        # try values from last-1 down to row[-2]+1
        for new_last in range(last - 1, row[-2], -1):
            # check differences involving new_last
            new_diffs = {new_last - a for a in row[:-1]}
            # remove old diffs involving old last
            old_diffs = {last - a for a in row[:-1]}
            # check if new_diffs conflict with used minus old_diffs
            remaining_used = used - old_diffs
            if not new_diffs & remaining_used:
                # also ensure strictly increasing (new_last > row[-2]) which is guaranteed by range
                # accept
                row[-1] = new_last
                used -= old_diffs
                used |= new_diffs
                new_scope = _score(best)
                if new_scope < best_scope:
                    best_scope = new_scope
                improved = True
                break
        if not improved:
            # try to reduce a middle element
            pos = rng.randrange(1, len(row) - 1) if len(row) > 2 else None
            if pos is None:
                continue
            cur = row[pos]
            lower = row[pos - 1] + 1
            upper = row[pos + 1] - 1
            if lower >= upper:
                continue
            # try values from cur-1 down to lower
            for new_val in range(cur - 1, lower - 1, -1):
                # compute old diffs involving row[pos]
                old_diffs = set()
                for j in range(len(row)):
                    if j == pos:
                        continue
                    if j < pos:
                        old_diffs.add(row[pos] - row[j])
                    else:
                        old_diffs.add(row[j] - row[pos])
                # compute new diffs
                new_diffs = set()
                for j in range(len(row)):
                    if j == pos:
                        continue
                    if j < pos:
                        new_diffs.add(new_val - row[j])
                    else:
                        new_diffs.add(row[j] - new_val)
                remaining_used = used - old_diffs
                if not new_diffs & remaining_used:
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    new_scope = _score(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
    return best


def _hill_climb(n_rows, k, seed=0, max_local_iters=50):
    """Generate a greedy solution and apply local search."""
    rows = _greedy_set(n_rows, k, seed)
    rows = _local_search(rows, k, max_iter=max_local_iters, seed=seed)
    return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0, minimizing the
    largest entry, subject to all positive differences within rows being distinct
    across all rows.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    best_rows = None
    best_scope = float('inf')

    # Start with a few greedy seeds
    seeds_to_try = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rng = random.Random(42)
    seeds_to_try += [rng.randrange(10000) for _ in range(20)]

    for seed in seeds_to_try:
        if time.time() > deadline:
            break
        rows = _greedy_set(n_rows, k, seed)
        rows = _local_search(rows, k, max_iter=100, seed=seed)
        scope = _score(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    # If we have time, do a few more local search passes on the best solution
    if best_rows is not None and time.time() < deadline:
        for extra_seed in range(5):
            if time.time() > deadline:
                break
            rows = _local_search(best_rows, k, max_iter=200, seed=1000 + extra_seed)
            scope = _score(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback: pure greedy
        best_rows = _greedy_set(n_rows, k, 0)

    return best_rows

# EVOLVE-BLOCK-END
