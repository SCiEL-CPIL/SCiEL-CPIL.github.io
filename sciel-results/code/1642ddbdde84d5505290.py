"""Rewritten program: maximize subset of [1,100] with no subset summing to 200."""
import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    target = 200
    mask = (1 << (target + 1)) - 1  # keep bits 0..target
    full = list(range(1, 101))

    def greedy(order):
        """Include elements in given order if it doesn't make target reachable."""
        reachable = 1  # bit 0
        selected = []
        for x in order:
            new_sums = (reachable << x) & mask
            if not (new_sums & (1 << target)):
                selected.append(x)
                reachable |= new_sums
        return selected

    def is_valid(S):
        """Verify no subset of S sums to target."""
        reachable = 1
        for x in S:
            reachable |= (reachable << x) & mask
        return not (reachable & (1 << target))

    best = []
    best_len = 0

    def consider(S):
        nonlocal best, best_len
        if S and len(S) > best_len and is_valid(S):
            best = S
            best_len = len(S)

    # Structured orderings
    structured = [
        list(range(100, 0, -1)),          # descending
        list(range(1, 101)),              # ascending
        list(range(100, 50, -1)) + list(range(50, 0, -1)),
        list(range(1, 51)) + list(range(100, 50, -1)),
        list(range(50, 101)) + list(range(49, 0, -1)),
    ]
    # alternating ends
    lo, hi = 1, 100
    alt = []
    while lo <= hi:
        alt.append(hi)
        if lo != hi:
            alt.append(lo)
        lo += 1
        hi -= 1
    structured.append(alt)

    for order in structured:
        consider(greedy(order))
        if time.time() > deadline:
            break

    # Random orderings
    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline - 2:
        order = full[:]
        rng.shuffle(order)
        consider(greedy(order))
        attempts += 1
        # occasionally also try a biased shuffle (larger first with noise)
        if attempts % 5 == 0:
            order2 = sorted(full, key=lambda x: -x + rng.uniform(-15, 15))
            consider(greedy(order2))

    # Local improvement: try to add excluded elements back
    if best:
        best_set = set(best)
        excluded = [x for x in full if x not in best_set]
        # try adding each excluded element in various orders
        for order in [excluded, excluded[::-1], sorted(excluded), sorted(excluded, reverse=True)]:
            cur = list(best)
            cur_set = set(cur)
            reachable = 1
            for x in cur:
                reachable |= (reachable << x) & mask
            new_selected = list(cur)
            for x in order:
                if x in cur_set:
                    continue
                new_sums = (reachable << x) & mask
                if not (new_sums & (1 << target)):
                    new_selected.append(x)
                    cur_set.add(x)
                    reachable |= new_sums
            consider(new_selected)
            if time.time() > deadline:
                break

    return best if best else [100]
