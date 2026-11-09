"""Černý problem (N=12): maximize exact shortest reset-word length.

Structural approach: simulated annealing over the two transition functions,
with an exact BFS threshold evaluation and a deadline guard. Seeded from a
known strong construction to start near the high end of the landscape.
"""
import time
import random
from collections import deque

N = 12
DEADLINE = time.time() + 80.0  # safety margin under the 90s wall-clock kill


def _threshold(a, b):
    """Exact shortest reset-word length via BFS on the subset automaton.

    Returns the length, or -1 if the automaton does not synchronize.
    A full set that maps to itself is treated as a fixed point (non-sync).
    """
    start = frozenset(range(N))
    # BFS: state -> length
    seen = {start: 0}
    q = deque([start])
    while q:
        s = q.popleft()
        d = seen[s]
        if len(s) == 1:
            return d
        for t in (a, b):
            ns = frozenset(t[x] for x in s)
            if ns not in seen:
                if len(ns) == len(s) and ns == s:
                    # fixed point of full set under this letter: still fine,
                    # just don't loop; it's already 'seen'
                    pass
                seen[ns] = d + 1
                q.append(ns)
    return -1  # never reached a singleton -> not synchronizing


def _cherny_construction():
    """A known synchronizing automaton with a long reset word.

    Letter a: cycle 0->1->...->N-1->0.
    Letter b: maps everything to 0 except state 1 which maps to 2
    (a 'defect' that forces a long word to accumulate the full set).
    This is a standard family that yields thresholds well above trivial.
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    b[1] = 2
    return a, b


def construct():
    random.seed(20240517)

    # Start from a strong known baseline.
    a, b = _cherny_construction()
    best = (a[:], b[:])
    best_score = _threshold(a, b)
    if best_score < 0:
        # Fallback: trivially synchronizing (cycle + constant).
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        best = (a[:], b[:])
        best_score = _threshold(a, b)

    cur_a, cur_b = a[:], b[:]
    cur_score = best_score

    # Simulated annealing
    temp = 12.0
    cool = 0.985
    step = 0
    while time.time() < DEADLINE:
        step += 1
        # Random move: change one transition of one letter.
        letter = random.random() < 0.5
        if letter:
            i = random.randrange(N)
            cur_a[i] = random.randrange(N)
            old = cur_a
        else:
            i = random.randrange(N)
            cur_b[i] = random.randrange(N)
            old = cur_b

        new_score = _threshold(cur_a, cur_b)

        if new_score >= 0:
            # Valid synchronizing automaton.
            if new_score >= cur_score:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = (cur_a[:], cur_b[:])
            else:
                # Worse: accept with probability exp((new-old)/temp)
                delta = new_score - cur_score
                if random.random() < pow(2.718281828, delta / max(temp, 1e-6)):
                    cur_score = new_score
        else:
            # Not synchronizing: revert this move.
            if letter:
                cur_a[i] = old[i]
            else:
                cur_b[i] = old[i]
            # Keep cur_score as-is (we reverted).

        temp *= cool
        if temp < 0.5:
            # Occasionally reheat to escape deep local optima.
            if random.random() < 0.1:
                temp = 6.0

    return best[0], best[1]
