"""Improved program: local search / simulated annealing over residue choices."""
import time
import numpy as np
import random

BASE = 51975
# All divisors > 1 of BASE
_DIVS = []
for _d in range(2, BASE + 1):
    if BASE % _d == 0:
        _DIVS.append(_d)
_DIVS = sorted(_DIVS)
# Precompute residue index arrays for each divisor
_RES_IDX = {m: np.arange(r, BASE, m) for m in _DIVS for r in range(m)}
# Actually build properly
_RES_IDX = {}
for _m in _DIVS:
    for _r in range(_m):
        _RES_IDX[(_m, _r)] = np.arange(_r, BASE, _m)


def _score(assignment):
    """assignment: list of (m, r) pairs. Returns number of uncovered residues."""
    count = np.zeros(BASE, dtype=np.int16)
    for m, r in assignment:
        idx = _RES_IDX[(m, r)]
        count[idx] += 1
    return int(np.sum(count == 0))


def _delta_add(m, r, count):
    idx = _RES_IDX[(m, r)]
    return int(np.sum(count[idx] == 0))


def _delta_remove(m, r, count):
    idx = _RES_IDX[(m, r)]
    return int(np.sum(count[idx] == 1))


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s

    # Start with greedy assignment: for each divisor, pick residue covering most uncovered
    count = np.zeros(BASE, dtype=np.int16)
    assignment = {}  # m -> r
    # Process divisors in a reasonable order (smaller first gives more residues)
    for m in _DIVS:
        best_r = 0
        best_gain = -1
        for r in range(m):
            idx = _RES_IDX[(m, r)]
            gain = int(np.sum(count[idx] == 0))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        assignment[m] = best_r
        count[_RES_IDX[(m, best_r)]] += 1

    current_score = int(np.sum(count == 0))
    best_score = current_score
    best_assignment = dict(assignment)

    # Local search: 1-move swaps
    # Order divisors by some heuristic (larger moduli first may be more impactful)
    div_order = sorted(_DIVS, key=lambda x: -x)

    # Precompute neighbor candidates for each divisor (all residues)
    # We'll iterate over divisors and try changing their residue

    iteration = 0
    no_improve = 0
    max_no_improve = 200000

    while time.time() < DEADLINE:
        iteration += 1
        if no_improve > max_no_improve:
            break

        # Pick a random divisor to try changing
        m = random.choice(div_order)
        current_r = assignment[m]

        # Try a random new residue for this divisor
        new_r = random.randint(0, m - 1)
        if new_r == current_r:
            continue

        # Compute delta: remove old, add new
        idx_old = _RES_IDX[(m, current_r)]
        idx_new = _RES_IDX[(m, new_r)]

        # Remove old: residues that go from 1 to 0 become uncovered (bad if they were only covered by this)
        # Actually: removing decreases count by 1 at idx_old. Uncovered increases by 1 where count was 1.
        # Adding increases count by 1 at idx_new. Uncovered decreases by 1 where count was 0.
        # But there can be overlap between idx_old and idx_new.

        # Net change in uncovered:
        # After remove: +1 for positions where count[idx_old] == 1
        # After add: -1 for positions where (count_after_remove)[idx_new] == 0
        # Let's compute carefully.

        # Positions in idx_old where count == 1: these become uncovered after removal
        removed_uncover = int(np.sum(count[idx_old] == 1))

        # For idx_new: after removal, count at those positions is count[idx_new] - (1 if in idx_old else 0)
        # This is getting complex. Let's just recompute score for the change.
        # But that's expensive. Let's do a smarter approach.

        # Alternative: compute the change directly.
        # New count = count - 1 at idx_old + 1 at idx_new
        # Uncovered change = (new_uncovered_count) - (old_uncovered_count)
        # = |{x: new_count[x]==0}| - |{x: count[x]==0}|

        # Let's just do it the straightforward way but vectorized:
        # temp_count = count.copy()  # too expensive
        # Instead:
        # The change in uncovered is:
        # - For x in idx_old \ idx_new: if count[x]==1, uncovered +1; if count[x]==0, uncovered -1 (but count can't be 0 before removal since we're removing)
        #   Actually if count[x]==1, after removal count[x]=0, so +1 uncovered. If count[x]>1, no change.
        # - For x in idx_new \ idx_old: if count[x]==0, after addition count[x]=1, so -1 uncovered. If count[x]>0, no change.
        # - For x in idx_old ∩ idx_new: net change is 0, no effect on uncovered.

        # So: delta = |{x in idx_old \ idx_new: count[x]==1}| - |{x in idx_new \ idx_old: count[x]==0}|

        # Computing set difference is expensive. Let's use a different approach:
        # Just recompute the score change by checking affected positions.
        # Affected positions are idx_old ∪ idx_new.
        # For efficiency, let's just do a full recompute if the moduli are small,
        # or use the formula above.

        # Let's use a simpler approach: compute the new score directly for the affected positions.
        # Actually, for large moduli this is still O(m) which is fine.

        # Let me just do the simple delta computation:
        # Save old count at affected positions
        affected = np.unique(np.concatenate([idx_old, idx_new]))
        old_counts = count[affected].copy()

        # Apply changes
        count[idx_old] -= 1
        count[idx_new] += 1

        new_score = int(np.sum(count == 0))
        delta = new_score - current_score

        if delta < 0:
            # Accept improvement
            current_score = new_score
            assignment[m] = new_r
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_assignment = dict(assignment)
        else:
            # Revert
            count[idx_old] += 1
            count[idx_new] -= 1
            no_improve += 1

    # Return the best assignment found
    result = [[m, r] for m, r in best_assignment.items()]
    return result
