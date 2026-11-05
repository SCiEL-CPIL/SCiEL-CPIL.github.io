from fractions import Fraction
import time

def _subsums(elems):
    """Set of all subset sums of reciprocals of elems (including empty sum 0)."""
    sums = {Fraction(0, 1)}
    for e in elems:
        r = Fraction(1, e)
        sums = sums | {s + r for s in sums}
    return sums

def _is_valid(S):
    """S: list of distinct ints. Valid iff for all x in S, 1/x not in SubSums(S\{x})."""
    Sset = set(S)
    # cache subsums of S\{x}
    cache = {}
    for x in Sset:
        others = [y for y in Sset if y != x]
        key = frozenset(others)
        if key not in cache:
            cache[key] = _subsums(others)
        if Fraction(1, x) in cache[key]:
            return False
    return True

def _greedy(order):
    """Greedy add elements in given order, keeping valid set."""
    T = []
    Tset = set()
    subsums_T = {Fraction(0,1)}
    for x in order:
        if x in Tset:
            continue
        # check 1/x not in subsums_T
        if Fraction(1, x) in subsums_T:
            continue
        # check for each y in T: (1/y - 1/x) not in SubSums(T\{y})
        ok = True
        rx = Fraction(1, x)
        for y in T:
            others = [z for z in T if z != y]
            s = _subsums(others)
            if Fraction(1, y) - rx in s:
                ok = False
                break
        if ok:
            T.append(x)
            Tset.add(x)
            subsums_T = subsums_T | {s + rx for s in subsums_T}
    return T

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s (kill at 90 but be safe)

    best = None
    best_size = -1

    # Multiple greedy orders
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        list(range(24, 0, -2)) + list(range(23, 0, -2)),  # evens then odds desc
        list(range(2, 25, 2)) + list(range(1, 24, 2)),    # evens then odds asc
        sorted(range(1, 25), key=lambda x: -x),
    ]

    for order in orders:
        if time.time() > deadline:
            break
        T = _greedy(order)
        if len(T) > best_size:
            best_size = len(T)
            best = T

    # Local search: try adding each excluded element by removing minimal conflicts
    if best is None:
        return list(range(13, 25))

    S = list(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        Sset = set(S)
        excluded = [x for x in range(1, 25) if x not in Sset]
        # For each excluded, try adding; if invalid, find minimal removals
        for x in excluded:
            if time.time() > deadline:
                break
            cand = S + [x]
            if _is_valid(cand):
                S = cand
                improved = True
                break
            else:
                # try removing one element to fix
                for y in list(S):
                    if time.time() > deadline:
                        break
                    cand2 = [z for z in S if z != y] + [x]
                    if len(cand2) > len(S) and _is_valid(cand2):
                        S = cand2
                        improved = True
                        break
                if improved:
                    break

    return S
