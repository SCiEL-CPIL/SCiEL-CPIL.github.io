import time
import random
from itertools import combinations

DEADLINE = time.time() + 80.0  # safety margin before 90s kill
MAX_N = 192
K = 3

def _triple_sum_bitset(arr):
    """Return list bs where bs[i] is a bitset over positions j of arr[i]
    (i in arr) such that arr[i] divides sum of some 3 distinct elements of arr
    (not including position i)."""
    n = len(arr)
    bs = [0] * n
    for i, a in enumerate(arr):
        others = [arr[j] for j in range(n) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                s = sum(t)
                for j, v in enumerate(arr):
                    if j != i and v == s:
                        bs[i] |= (1 << j)
    return bs

def _can_add_x(existing, x):
    """Check if x can be added to existing (list, sorted ascending) without
    violating the condition. Returns True if valid."""
    n = len(existing)
    # (1) existing elements a: does a divide (x + p + q) for distinct p,q in existing?
    for a in existing:
        # iterate pairs (i,j) i<j in existing
        for i in range(n):
            for j in range(i + 1, n):
                if (x + existing[i] + existing[j]) % a == 0:
                    return False
    # (2) x: does x divide sum of any 3 distinct existing?
    for t in combinations(existing, 3):
        if sum(t) % x == 0:
            return False
    return True

def _greedy_build(order):
    """Build a set by considering candidates in given order, adding if valid."""
    selected = []
    for x in order:
        if _can_add_x(selected, x):
            selected.append(x)
            selected.sort()
    return selected

def construct():
    global DEADLINE
    start = time.time()
    best = []
    best_len = 0

    # 1) Descending greedy (matches seed behavior, gives baseline)
    cand = _greedy_build(range(MAX_N, 0, -1))
    if len(cand) > best_len:
        best = cand
        best_len = len(cand)

    # 2) Randomized restarts with different orderings
    rng = random.Random(12345)
    attempts = 0
    while time.time() < DEADLINE:
        attempts += 1
        # Build a random-ish order: mix of descending chunks with random shuffles
        # Strategy: pick a random "bias" — some descending, some random, some ascending
        mode = rng.random()
        if mode < 0.4:
            # mostly descending with random perturbations
            order = list(range(MAX_N, 0, -1))
            for _ in range(rng.randint(5, 40)):
                i = rng.randint(0, MAX_N - 1)
                j = rng.randint(0, MAX_N - 1)
                order[i], order[j] = order[j], order[i]
        elif mode < 0.7:
            # fully random
            order = list(range(1, MAX_N + 1))
            rng.shuffle(order)
        else:
            # ascending (sometimes good to start with small numbers)
            order = list(range(1, MAX_N + 1))
            for _ in range(rng.randint(5, 40)):
                i = rng.randint(0, MAX_N - 1)
                j = rng.randint(0, MAX_N - 1)
                order[i], order[j] = order[j], order[i]

        cand = _greedy_build(order)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            # If we hit the cap, we can stop early
            if best_len >= 48:
                break

    return best

# EVOLVE-BLOCK-END
