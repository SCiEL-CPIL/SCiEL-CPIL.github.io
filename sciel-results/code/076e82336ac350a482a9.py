import time
import random

N = 512

# Precompute, for each element x, the set of 4-APs containing x.
# conflict[x] = list of frozensets of the OTHER 3 elements.
# x is addable to S iff no conflict[x] entry is a subset of S.
def _build_conflicts():
    conf = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            t = a + 3 * d
            s = frozenset((a, b, c, t))
            conf[a].add(frozenset((b, c, t)))
            conf[b].add(frozenset((a, c, t)))
            conf[c].add(frozenset((a, b, t)))
            conf[t].add(frozenset((a, b, c)))
    return [tuple(v) for v in conf]

CONFLICT = _build_conflicts()


def _can_add(x, S):
    for m in CONFLICT[x]:
        if m <= S:
            return False
    return True


def _base3_set(limit):
    """Salem-Spencer: numbers whose base-3 digits are only 0/1, in [1, limit]."""
    vals = set()
    for mask in range(1, 1 << 7):
        v = 0
        pw = 1
        m = mask
        while m:
            if m & 1:
                v += pw
            m >>= 1
            pw *= 3
            if v > limit:
                break
        if 1 <= v <= limit:
            vals.add(v)
    return vals


def _greedy_fill(S, order):
    """Add any addable element in order."""
    for x in order:
        if x not in S and _can_add(x, S):
            S.add(x)
    return S


def construct():
    start = time.time()
    deadline = start + 52.0  # margin under the 60s wall-clock kill

    def remaining():
        return deadline - time.time()

    # ---- Initial construction: union of block base-3 seeds, then greedy fill ----
    S = set()
    block_len = 81
    overlap = 10
    bs = 1
    while bs <= N:
        be = min(bs + block_len - 1, N)
        for v in _base3_set(be - bs + 1):
            S.add(bs + v - 1)
        bs += block_len - overlap

    order = list(range(1, N + 1))
    random.shuffle(order)
    _greedy_fill(S, order)
    # a couple of full sweeps to catch everything
    for _ in range(3):
        for x in order:
            if x not in S and _can_add(x, S):
                S.add(x)

    best = set(S)
    best_size = len(S)

    # ---- Local search: swaps (remove one, add one) + periodic fill ----
    # Maintain a list mirror of S for fast random element access.
    def list_from_set(st):
        return list(st)

    S = best
    Slist = list_from_set(S)
    size = best_size

    def try_fill():
        nonlocal S, Slist, size
        changed = False
        for x in range(1, N + 1):
            if x not in S and _can_add(x, S):
                S.add(x)
                changed = True
        if changed:
            Slist = list_from_set(S)
            size = len(S)
            if size > best_size:
                best_size = size
                best = set(S)
        return changed

    # Build a quick membership set reference
    rng = random.Random(12345)

    steps = 0
    while remaining() > 1.5:
        steps += 1
        # 1) Try to add a random element
        x = rng.randint(1, N)
        if x not in S and _can_add(x, S):
            S.add(x)
            Slist.append(x)
            size += 1
            if size > best_size:
                best_size = size
                best = set(S)
            continue

        # 2) Try a swap: remove a random y, add x
        y = Slist[rng.randrange(len(Slist))]
        # temporarily remove y
        S.discard(y)
        found = False
        for _ in range(6):
            cand = rng.randint(1, N)
            if cand == y:
                continue
            if cand not in S and _can_add(cand, S):
                S.add(cand)
                # update list: replace y with cand
                Slist[Slist.index(y)] = cand
                found = True
                break
        if not found:
            # revert
            S.add(y)
        else:
            # size unchanged; but the swap may enable new additions -> occasional fill
            if steps % 40 == 0:
                try_fill()

        # 3) Periodic full fill to escape local optima
        if steps % 150 == 0:
            try_fill()

    # ---- Final exhaustive fill ----
    for x in range(1, N + 1):
        if x not in best and _can_add(x, best):
            best.add(x)

    result = sorted(best)
    if not result:
        result = [1, 2]
    return result
