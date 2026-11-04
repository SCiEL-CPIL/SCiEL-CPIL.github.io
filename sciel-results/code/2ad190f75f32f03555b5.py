"""Difference triangle set (4,7) - fresh design with randomized greedy + simulated annealing."""
import time
import random
import sys
from collections import defaultdict

_DEADLINE = 150.0
_START_TIME = None

def _elapsed():
    return time.time() - _START_TIME

def _compute_differences(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return list of per-row diff sets and the global set."""
    per_row = []
    global_set = set()
    for row in rows:
        d = _compute_differences(row)
        per_row.append(d)
        global_set |= d
    return per_row, global_set

def _count_duplicates(per_row):
    """Count how many differences are duplicated across rows."""
    cnt = defaultdict(int)
    for d in per_row:
        for x in d:
            cnt[x] += 1
    return sum(c - 1 for c in cnt.values() if c > 1)

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_random(n_rows, k, rng, bias=0.5, max_scan=20000):
    """Greedy construction with randomization.
    At each step, pick from a window of candidate marks, weighted toward small ones.
    Returns rows or None if a row couldn't be completed within max_scan.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Scan a window of candidate marks
            best_x = None
            best_score = None
            # Consider candidates in a window [x+1, x+window]
            window = 12
            candidates = []
            for cand in range(x + 1, x + 1 + window):
                new = {cand - a for a in row}
                if not (new & used):
                    candidates.append(cand)
            if not candidates:
                # No valid candidate in window; extend scan
                found = False
                for cand in range(x + 1 + window, x + 1 + max_scan):
                    new = {cand - a for a in row}
                    if not (new & used):
                        candidates.append(cand)
                        found = True
                        break
                if not found:
                    return None
            # Choose: with probability bias, pick smallest; else random from candidates
            if rng.random() < bias:
                chosen = candidates[0]
            else:
                # Weight toward smaller candidates
                weights = [1.0 / (i + 1) for i in range(len(candidates))]
                chosen = rng.choices(candidates, weights=weights, k=1)[0]
            x = chosen
            row.append(x)
            used |= {x - a for a in row}
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, rng, max_iters=200000, deadline_fn=None):
    """Simulated annealing local search.
    Moves: pick a random row, pick a random position (not 0), perturb the value
    by a small random delta, and check if the move improves or is accepted.
    Objective: minimize (duplicates * BIG + max_entry).
    """
    BIG = 100000  # heavy penalty for duplicates
    rows = [list(r) for r in rows]
    per_row, global_set = _all_diffs(rows)
    cur_dup = _count_duplicates(per_row)
    cur_max = _max_entry(rows)
    cur_obj = cur_dup * BIG + cur_max

    best_rows = [list(r) for r in rows]
    best_obj = cur_obj
    best_dup = cur_dup
    best_max = cur_max

    T = 50.0
    cool = 0.9999
    it = 0
    while it < max_iters:
        if deadline_fn is not None and deadline_fn():
            break
        # Pick a random row and position
        ri = rng.randrange(n_rows)
        row = rows[ri]
        # Positions 1..len(row)-1 (position 0 is fixed at 0)
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        # Perturbation size depends on current value
        delta = rng.randint(-3, 3)
        if delta == 0:
            delta = rng.choice([-1, 1])
        new_val = old_val + delta
        # Constraints: must stay strictly increasing and > 0
        if new_val <= 0:
            continue
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            continue

        # Compute change in differences for this row
        old_diffs = per_row[ri]
        # Remove old diffs involving position pos
        for j in range(len(row)):
            if j != pos:
                d = abs(row[j] - old_val)
                old_diffs.discard(d)
        # Apply new value
        row[pos] = new_val
        # Add new diffs involving position pos
        for j in range(len(row)):
            if j != pos:
                d = abs(row[j] - new_val)
                old_diffs.add(d)

        # Recompute duplicates and max for this row
        # For efficiency, recompute global duplicates only approximately
        # Recompute this row's max
        new_row_max = max(row)
        # Recompute global max
        new_max = new_max_entry = max(max(r) for r in rows)
        # Recompute duplicates (expensive but necessary for correctness)
        # We'll recompute per_row for this row only and track global counts
        # For simplicity and correctness, recompute full duplicates
        new_dup = _count_duplicates(per_row)
        new_obj = new_dup * BIG + new_max

        delta_obj = new_obj - cur_obj
        if delta_obj <= 0 or rng.random() < pow(2.718281828, -delta_obj / T):
            cur_obj = new_obj
            cur_dup = new_dup
            cur_max = new_max
            if new_dup == 0 and new_max < best_max:
                best_max = new_max
                best_dup = new_dup
                best_obj = new_obj
                best_rows = [list(r) for r in rows]
        else:
            # Revert
            row[pos] = old_val
            for j in range(len(row)):
                if j != pos:
                    d = abs(row[j] - new_val)
                    old_diffs.discard(d)
                    d2 = abs(row[j] - old_val)
                    old_diffs.add(d2)

        T *= cool
        it += 1

    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(12345)

    def deadline_fn():
        return _elapsed() > _DEADLINE - 5.0

    # Phase 1: Generate diverse greedy candidates
    best_rows = None
    best_max = float('inf')

    # First, a pure greedy (deterministic) baseline
    pure = _greedy_random(n_rows, k, rng, bias=1.0)
    if pure is not None:
        best_rows = pure
        best_max = _max_entry(pure)

    # Then randomized greedy restarts
    attempts = 0
    while not deadline_fn():
        bias = rng.uniform(0.3, 0.8)
        cand = _greedy_random(n_rows, k, rng, bias=bias)
        attempts += 1
        if cand is None:
            continue
        mx = _max_entry(cand)
        if mx < best_max:
            best_max = mx
            best_rows = cand
        # Occasionally run local search on promising candidates
        if mx <= best_max + 5 and attempts % 3 == 0 and not deadline_fn():
            ls_rows, ls_max = _local_search(
                cand, n_rows, k, rng,
                max_iters=50000, deadline_fn=deadline_fn
            )
            if ls_max < best_max:
                best_max = ls_max
                best_rows = ls_rows

    # Phase 2: More aggressive local search on the best candidate
    if best_rows is not None and not deadline_fn():
        ls_rows, ls_max = _local_search(
            best_rows, n_rows, k, rng,
            max_iters=300000, deadline_fn=deadline_fn
        )
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    if best_rows is None:
        # Fallback: should not happen, but return something valid
        # Use a simple construction
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

    return best_rows
