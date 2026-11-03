"""
Task: Construct a finite set U of nonnegative integers containing 0 to maximize
      1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)

The seed is [0, 1, 3] with score ~1.079.
A well-known construction for "sum-free" or "sumset-efficient" sets uses
geometric progression with base 2: {0, 1, 3, 7, 15, 31, ...} = {2^k - 1}.

For U = {0, 1, 3, 7, 15}:
- |U| = 5, max(U) = 15
- U+U has 15 distinct sums
- U-U has 21 distinct differences
- Ratio = 21/15 = 1.4
- Score = 1 + log(1.4)/log(31) ≈ 1.098

We can do better with a small local search over sets of size up to 8
with elements bounded, plus a deadline guard.
"""

import time
import itertools
from math import log

def _score(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    if max(U) == 0:
        return 0.0
    # Compute |U+U| and |U-U|
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    ratio = len(minus) / len(plus)
    if ratio <= 0:
        return 0.0
    denom = log(2 * max(U) + 1)
    if denom == 0:
        return 0.0
    return 1.0 + log(ratio) / denom

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin from 150s

    # Start with known good sets
    candidates = [
        [0, 1, 3],           # seed, score ~1.079
        [0, 1, 3, 7],        # score ~1.085
        [0, 1, 3, 7, 15],    # score ~1.098
        [0, 1, 3, 7, 15, 31],
    ]

    best_U = candidates[0]
    best_score = _score(best_U)

    for U in candidates[1:]:
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U

    # Local search: try adding or replacing elements to improve score
    # We'll do a simple hill-climbing over small perturbations.
    max_iter = 5000
    current_U = list(best_U)
    current_score = best_score

    # Try extending the geometric sequence further
    # {0, 1, 3, 7, 15, 31, 63}
    geo = [0]
    val = 1
    for _ in range(12):
        current_U = list(geo)
        s = _score(current_U)
        if s > best_score:
            best_score = s
            best_U = current_U
        geo.append(2 * val + 1)
        val = 2 * val + 1

    # Try some hand-picked sets
    extra_sets = [
        [0, 1, 2, 4, 8, 16],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 4, 8],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 7, 16],
        [0, 1, 4, 9, 16],
    ]
    for U in extra_sets:
        if time.time() > DEADLINE:
            break
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # Hill climbing: try replacing each element with nearby values
    for iteration in range(max_iter):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(len(best_U)):
            old_val = best_U[i]
            # Try neighbors
            for delta in range(-3, 4):
                new_val = old_val + delta
                if new_val < 0:
                    continue
                # Don't duplicate
                if new_val in best_U and new_val != old_val:
                    continue
                test_U = list(best_U)
                test_U[i] = new_val
                test_U = sorted(set(test_U))
                s = _score(test_U)
                if s > current_score + 1e-12:
                    current_score = s
                    current_U = test_U
                    improved = True
                    break
        if not improved:
            # Try adding a new element
            max_val = max(current_U)
            for new_val in range(max_val + 1, max_val + 20):
                test_U = sorted(current_U + [new_val])
                s = _score(test_U)
                if s > current_score + 1e-12:
                    current_score = s
                    current_U = test_U
                    improved = True
                    break
            # Try removing an element
            if len(current_U) > 2:
                for i in range(len(current_U)):
                    if i == 0:  # keep 0
                        continue
                    test_U = current_U[:i] + current_U[i+1:]
                    s = _score(test_U)
                    if s > current_score + 1e-12:
                        current_score = s
                        current_U = test_U
                        improved = True
                        break
        if not improved:
            break
        if current_score > best_score:
            best_score = current_score
            best_U = list(current_U)

    return best_U

# EVOLVE-BLOCK-END
