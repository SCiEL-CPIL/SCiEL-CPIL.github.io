"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 16
K = 4
UNIVERSE = (1 << N) - 1


def _precompute():
    """Build all 4-subsets as bitmasks, a safe-pair matrix, and a conflict matrix."""
    rows = list(combinations(range(N), K))
    m = len(rows)
    masks = []
    pop = []
    for r in rows:
        b = 0
        for x in r:
            b |= 1 << x
        masks.append(b)
        pop.append(bin(b).count('1'))
    # A pair (i,j) is "safe" (can never be in a sunflower triple) if the number
    # of possible distinct third sets is <= 1.
    safe = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            s = masks[i] & masks[j]
            ss = bin(s).count('1')
            if ss == 4:
                continue  # identical sets, not a distinct pair
            avail = (UNIVERSE ^ (masks[i] | masks[j]))
            a = bin(avail).count('1')
            need = K - ss
            # count C(a, need)
            if need == 0:
                c = 1
            elif need == 1:
                c = a
            elif need == 2:
                c = a * (a - 1) // 2
            elif need == 3:
                c = a * (a - 1) * (a - 2) // 6
            else:
                c = a * (a - 1) * (a - 2) * (a - 3) // 24
            if c <= 1:
                safe[i][j] = safe[j][i] = True
    # Conflict matrix: edge if the pair is NOT safe.
    conflict = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if not safe[i][j]:
                conflict[i][j] = conflict[j][i] = True
    return rows, m, masks, pop, safe, conflict


_ROWS, _M, _MASKS, _POP, _SAFE, _CONFLICT = _precompute()
_T0 = time.time()


def _greedy(order, deadline, conflict):
    """Greedy: add sets in order if they conflict with no already-chosen set."""
    chosen = []
    chosen_set = set()
    for idx in order:
        if time.time() > deadline:
            break
        ok = True
        for j in chosen:
            if conflict[idx][j]:
                ok = False
                break
        if ok:
            chosen.append(idx)
            chosen_set.add(idx)
    return chosen


def _local_improve(chosen, deadline, conflict):
    """Add non-conflicting sets, then 1-swap removals, until no gain / time out."""
    chosen_set = set(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Phase A: add any set that conflicts with nothing in the family.
        for idx in range(_M):
            if idx in chosen_set:
                continue
            ok = True
            for j in chosen_set:
                if conflict[idx][j]:
                    ok = False
                    break
            if ok:
                chosen_set.add(idx)
                improved = True
        if time.time() > deadline:
            break
        # Phase B: try removing a single set that unblocks a net gain.
        lst = list(chosen_set)
        for rem in lst:
            if time.time() > deadline:
                break
            # sets that conflict with rem (candidates to add once rem is gone)
            candidates = [i for i in range(_M)
                          if i not in chosen_set and conflict[rem][i]]
            for add in candidates:
                # add must not conflict with any of the remaining family
                ok = True
                for j in chosen_set:
                    if j == rem:
                        continue
                    if conflict[add][j]:
                        ok = False
                        break
                if ok:
                    chosen_set.discard(rem)
                    chosen_set.add(add)
                    improved = True
                    break
            if improved:
                break
    return chosen_set


def construct():
    deadline = _T0 + 80.0  # safety margin before the 90s kill
    conflict = _CONFLICT
    rng = random.Random(7)

    orders = []
    # Natural order
    orders.append(list(range(_M)))
    # Random orders
    for seed in (42, 123, 999, 2024):
        o = list(range(_M))
        random.Random(seed).shuffle(o)
        orders.append(o)
    # Popcount-ascending (small intersections first), tie-broken randomly
    oa = list(range(_M))
    rng.shuffle(oa)
    oa.sort(key=lambda i: _POP[i])
    orders.append(oa)
    # Popcount-descending
    od = list(range(_M))
    rng.shuffle(od)
    od.sort(key=lambda i: -_POP[i])
    orders.append(od)

    best = set()
    for order in orders:
        if time.time() > deadline:
            break
        fam = _greedy(order, deadline, conflict)
        if len(fam) > len(best):
            best = set(fam)
        # local improvement on this family
        if time.time() < deadline:
            imp = _local_improve(list(fam), deadline, conflict)
            if len(imp) > len(best):
                best = imp

    # Final polish on the best family
    if time.time() < deadline:
        imp = _local_improve(list(best), deadline, conflict)
        if len(imp) > len(best):
            best = imp

    result = [sorted(_ROWS[i]) for i in best]
    return result
# EVOLVE-BLOCK-END
