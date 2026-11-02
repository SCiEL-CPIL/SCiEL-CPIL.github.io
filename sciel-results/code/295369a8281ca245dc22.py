"""
Difference triangle set (7,5) constructor.

We aim for n_rows rows of k+1 strictly increasing integers starting at 0,
all pairwise differences within rows globally unique, minimising the largest entry.
"""

import time
import random
import math
from collections import defaultdict

# ---- helpers ----

def _get_scope(rows):
    return max(max(r) for r in rows)

def _is_valid(rows, k):
    diffs = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _fast_greedy(n_rows, k):
    """Fast greedy: pick smallest next mark for each row that doesn't collide."""
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

# ---- simulated annealing optimizer ----

def _sa_optimize(initial_rows, k, deadline):
    """
    SA over the vector of all row entries (excluding 0).
    Move: pick a random row and a random position in that row, change the value
    to a nearby value, ensuring strictly increasing.
    Objective: minimise scope + penalty * (number of collisions).
    """
    n_rows = len(initial_rows)
    k1 = k + 1  # number of entries per row

    # Flatten: rows[i][j] for i in range(n_rows), j in range(1, k1)
    # We keep row[0] = 0 always.
    def _flatten(rows):
        flat = []
        for row in rows:
            flat.extend(row[1:])
        return flat

    def _unflatten(flat):
        rows = []
        for i in range(n_rows):
            row = [0] + list(flat[i * k:(i + 1) * k])
            rows.append(row)
        return rows

    def _collisions_and_scope(rows):
        diff_count = defaultdict(int)
        max_val = 0
        for row in rows:
            for j in range(len(row)):
                max_val = max(max_val, row[j])
                for i in range(j):
                    d = row[j] - row[i]
                    diff_count[d] += 1
        coll = sum(1 for c in diff_count.values() if c > 1)
        # Better: count pairs of duplicated diffs
        coll = sum(c - 1 for c in diff_count.values() if c > 1)
        return coll, max_val

    def _objective(flat):
        rows = _unflatten(flat)
        # Check strictly increasing
        for row in rows:
            for j in range(1, len(row)):
                if row[j] <= row[j - 1]:
                    return float('inf')
        coll, max_val = _collisions_and_scope(rows)
        return max_val + 1000 * coll

    current_flat = _flatten(initial_rows)
    current_obj = _objective(current_flat)
    best_flat = list(current_flat)
    best_obj = current_obj

    T = max(_get_scope(initial_rows) * 0.5, 50.0)
    cooling = 0.995
    min_T = 1.0

    while T > min_T and time.time() < deadline:
        # Pick random row and position
        i = random.randrange(n_rows)
        j = random.randrange(1, k1)  # position 1..k

        # Current value
        idx = i * k + (j - 1)
        old_val = current_flat[idx]

        # Neighbors: j-1 is the previous position (0-indexed in flat for this row), j+1 is next
        # In flat: row i occupies indices i*k .. i*k + k - 1
        # Position j in row (1-indexed, 0 is fixed) corresponds to flat index i*k + (j-1)
        # Previous value: if j > 1, flat[i*k + j - 2]; else 0
        # Next value: if j < k, flat[i*k + j]; else None

        prev_val = 0 if j == 1 else current_flat[i * k + j - 2]
        next_val = current_flat[i * k + j] if j < k else None

        # Try a few random nearby values
        improved = False
        for _ in range(3):
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            if new_val <= prev_val:
                new_val = prev_val + 1
            if next_val is not None and new_val >= next_val:
                new_val = next_val - 1
            if new_val <= prev_val:
                continue
            if new_val == old_val:
                continue

            current_flat[idx] = new_val
            new_obj = _objective(current_flat)
            delta_obj = new_obj - current_obj

            if new_obj < current_obj or random.random() < math.exp(-delta_obj / T):
                current_flat[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_flat = list(current_flat)
                improved = True
                break
            else:
                current_flat[idx] = old_val

        T *= cooling

    return best_obj, _unflatten(best_flat)

# ---- main constructor ----

def construct(n_rows, k):
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin

    # Generate several initial candidates with greedy (fast)
    candidates = []
    for _ in range(3):
        candidates.append(_fast_greedy(n_rows, k))

    # Sort by scope
    candidates.sort(key=_get_scope)
    best_scope = _get_scope(candidates[0])
    best_rows = candidates[0]

    # Run SA on the best candidate
    if time.time() < deadline - 5:
        obj, rows = _sa_optimize(candidates[0], k, deadline - 5)
        if _is_valid(rows, k):
            s = _get_scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows

    # If time remains, try SA on second best
    if time.time() < deadline - 5:
        obj2, rows2 = _sa_optimize(candidates[1], k, deadline - 5)
        if _is_valid(rows2, k):
            s2 = _get_scope(rows2)
            if s2 < best_scope:
                best_scope = s2
                best_rows = rows2

    # Final validation
    if not _is_valid(best_rows, k):
        for c in candidates:
            if _is_valid(c, k):
                return c
        return candidates[0]

    return best_rows
