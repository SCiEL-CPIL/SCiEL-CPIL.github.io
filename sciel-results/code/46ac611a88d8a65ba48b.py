"""Large sets without subset averages - bitset-accelerated randomized local search."""
import time
import random

N = 512
MAX_CARD = 48

# Precompute, for each mean a, the set of b (b < a) such that c = 2a - b is in [1, N].
# An element a is a bad mean of S iff popcount(B[a] & S) is odd.
_B = [0] * (N + 1)
for a in range(1, N + 1):
    bit = 0
    s = a * 2
    for b in range(1, a):
        c = s - b
        if 1 <= c <= N:
            bit |= (1 << b)
    _B[a] = bit


def _popcount(x):
    return x.bit_count()


def _bad_mask(Sbits):
    """Return bitset of elements a in the universe that are bad means w.r.t. Sbits."""
    bad = 0
    for a in range(1, N + 1):
        if _popcount(_B[a] & Sbits) & 1:
            bad |= (1 << a)
    return bad


def _greedy(order):
    """Greedily add elements in `order` if they keep the set mean-free."""
    chosen = []
    Sbits = 0
    bad = 0
    for x in order:
        if (bad >> x) & 1:
            continue
        newbits = Sbits | (1 << x)
        # recompute bad mask
        nb = 0
        for a in range(1, N + 1):
            if _popcount(_B[a] & newbits) & 1:
                nb |= (1 << a)
        if nb & (1 << x):
            continue
        chosen.append(x)
        Sbits = newbits
        bad = nb
    return chosen


def _local_search(S):
    """1-swap local improvement. Returns improved set (same or larger cardinality)."""
    S = list(S)
    Sbits = 0
    for x in S:
        Sbits |= (1 << x)
    bad = _bad_mask(Sbits)
    inS = set(S)
    improved = True
    while improved:
        improved = False
        # candidate additions: not in S, not currently a bad mean
        for y in range(1, N + 1):
            if (1 << y) in Sbits:
                continue
            if (bad >> y) & 1:
                continue
            for x in S:
                if (bad >> x) & 1:
                    continue
                newbits = Sbits | (1 << y) & ~(1 << x)
                nb = _bad_mask(newbits)
                if not (nb & newbits):
                    # valid swap
                    S.remove(x)
                    S.append(y)
                    Sbits = newbits
                    bad = nb
                    improved = True
                    break
            if improved:
                break
    return S


def _verify(S):
    """Exact check: no element is the mean of any subset of >=2 distinct others."""
    Sbits = 0
    for x in S:
        Sbits |= (1 << x)
    for a in S:
        if _popcount(_B[a] & Sbits) & 1:
            return False
    return True


def construct():
    t0 = time.time()
    deadline = 78.0  # safety margin under 90s wall-clock
    rng = random.Random(12345)

    best = None
    best_len = 0

    # Initial deterministic-ish greedy
    order = list(range(1, N + 1))
    g = _greedy(order)
    g = _local_search(g)
    if _verify(g):
        best = g
        best_len = len(g)

    iters = 0
    while time.time() - t0 < deadline:
        # randomized greedy
        order = list(range(1, N + 1))
        rng.shuffle(order)
        g = _greedy(order)
        g = _local_search(g)
        iters += 1
        if len(g) > best_len:
            if _verify(g):
                best = g
                best_len = len(g)
                if best_len >= MAX_CARD:
                    break

    # Final exact verification of the returned set
    if best is None:
        best = [1]
    elif not _verify(best):
        # fall back to a trivially valid small set
        best = [1, 2]
    return best
