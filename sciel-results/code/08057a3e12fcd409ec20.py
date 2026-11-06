"""Rewritten: fast O(K^3) validity + local search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
K = 3
LIMIT = 48

def _valid_add(s, x):
    """Check that x does not divide the sum of any 3 distinct other elements in s,
    and that no element of s divides the sum of any 3 distinct elements of s
    that include x.  s is known to already be internally valid (no element of s
    divides the sum of 3 other elements of s), so we only need interactions with x.
    """
    n = len(s)
    # 1) x must not divide sum of any 3 distinct elements of s
    if n >= 3:
        for a, b, c in combinations(s, 3):
            if (a + b + c) % x == 0:
                return False
    # 2) no element of s may divide sum of 3 distinct elements of s that include x
    if n >= 2:
        for a, b in combinations(s, 2):
            s3 = x + a + b
            for e in s:
                if s3 % e == 0:
                    return False
    return True

def _valid_full(s):
    """Full O(K^3) check that no element divides the sum of any 3 distinct others."""
    n = len(s)
    if n < 4:
        return True
    for i in range(n):
        e = s[i]
        others = [s[j] for j in range(n) if j != i]
        for a, b, c in combinations(others, 3):
            if (a + b + c) % e == 0:
                return False
    return True

def _greedy(order, start=None):
    """Greedy add in given order, keeping only valid additions."""
    selected = list(start) if start else []
    for x in order:
        if x in selected:
            continue
        if _valid_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, deadline, rng):
    """Try to grow the set via: (a) insert a missing element, (b) swap one out for
    two in, (c) random restarts of greedy with shuffled orders."""
    selected = list(selected)
    base_size = len(selected)
    best = selected

    # Precompute the universe
    universe = list(range(1, N + 1))

    def try_insert():
        nonlocal best, base_size
        if time.time() > deadline:
            return
        cur = list(best)
        # Try to insert one missing element
        missing = [x for x in universe if x not in cur]
        rng.shuffle(missing)
        for x in missing:
            if time.time() > deadline:
                break
            if _valid_add(cur, x):
                cur.append(x)
                if len(cur) > len(best):
                    best = list(cur)
                    base_size = len(best)
                # after one insert, break to re-evaluate
                break

    def try_swap():
        nonlocal best, base_size
        if time.time() > deadline:
            return
        cur = list(best)
        # remove one element, then try to insert two
        idx = rng.randrange(len(cur))
        removed = cur[idx]
        cur2 = cur[:idx] + cur[idx+1:]
        missing = [x for x in universe if x not in cur2]
        rng.shuffle(missing)
        improved = False
        for x in missing:
            if time.time() > deadline:
                return
            if _valid_add(cur2, x):
                cur3 = cur2 + [x]
                for y in missing:
                    if y == x:
                        continue
                    if y in cur3:
                        continue
                    if _valid_add(cur3, y):
                        cand = cur3 + [y]
                        if len(cand) > len(best):
                            best = cand
                            base_size = len(best)
                            improved = True
                        break
                if improved:
                    return
                cur2 = cur3  # keep the single insert, try another
                break

    # Main loop: alternate insert and swap attempts
    step = 0
    while time.time() < deadline - 2.0:
        step += 1
        if step % 5 == 0:
            try_swap()
        else:
            try_insert()

    return best

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill
    rng = random.Random(12345)

    # Phase 1: greedy from several orders to get a good starting set
    candidates = []

    # Descending
    sel = _greedy(range(N, 0, -1))
    candidates.append(sel)

    # Ascending
    sel = _greedy(range(1, N + 1))
    candidates.append(sel)

    # A few shuffled orders
    for seed in range(10):
        if time.time() > deadline - 30.0:
            break
        order = list(range(1, N + 1))
        random.Random(seed).shuffle(order)
        sel = _greedy(order)
        candidates.append(sel)

    # Pick the largest
    best = max(candidates, key=len)

    # Phase 2: local search to grow
    if time.time() < deadline - 5.0:
        best = _local_search(best, deadline, rng)

    # Final safety: ensure validity
    if not _valid_full(best):
        # fall back to greedy descending
        best = _greedy(range(N, 0, -1))

    return best
# EVOLVE-BLOCK-END
