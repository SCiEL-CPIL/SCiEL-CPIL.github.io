"""Suite task: subset of [1,256] with no 4-AP, maximizing number of 3-APs."""
import time
import random

N = 256
DEADLINE = 84.0  # safety margin under 90s kill


def count_3aps_in_set(selected):
    s = set(selected)
    c = 0
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            if a in s and (a + d) in s and (a + 2 * d) in s:
                c += 1
    return c


def has_4ap(selected):
    s = set(selected)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            if a in s and (a + d) in s and (a + 2 * d) in s and (a + 3 * d) in s:
                return True
    return False


def greedy_construct(order):
    selected = set()
    for x in order:
        ok = True
        # adding x creates a 4-AP iff x is the 1st, 2nd, 3rd, or 4th term
        # x as 1st: x+d, x+2d, x+3d in selected
        # x as 2nd: x-d, x+d, x+2d in selected
        # x as 3rd: x-2d, x-d, x+d in selected
        # x as 4th: x-3d, x-2d, x-d in selected
        for d in range(1, N // 3 + 1):
            if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
                ok = False
                break
            if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
                ok = False
                break
            if (x - 2 * d) in selected and (x - d) in selected and (x + d) in selected:
                ok = False
                break
            if (x - 3 * d) in selected and (x - 2 * d) in selected and (x - d) in selected:
                ok = False
                break
        if ok:
            selected.add(x)
    return selected


def local_search(selected, deadline, rng):
    selected = set(selected)
    cur = count_3aps_in_set(selected)
    best = selected
    best_c = cur
    while time.time() < deadline:
        improved = False
        # candidate additions
        adds = [x for x in range(1, N + 1) if x not in selected]
        rng.shuffle(adds)
        # candidate removals
        rems = list(selected)
        rng.shuffle(rems)
        for a in adds:
            if time.time() >= deadline:
                break
            for r in rems:
                ns = selected
                ns.add(a)
                ns.discard(r)
                if not has_4ap(ns):
                    c = count_3aps_in_set(ns)
                    if c > cur:
                        selected = ns
                        cur = c
                        improved = True
                        if cur > best_c:
                            best_c = cur
                            best = set(ns)
                        break
        if not improved:
            break
    return best, best_c


def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = None
    best_c = -1
    rng = random.Random(12345)

    # Try several random orderings for the greedy seed
    orders = []
    orders.append(list(range(1, N + 1)))
    for _ in range(30):
        o = list(range(1, N + 1))
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time.time() >= deadline:
            break
        sel = greedy_construct(o)
        c = count_3aps_in_set(sel)
        if c > best_c:
            best_c = c
            best = sel
        # local search to improve
        if time.time() < deadline - 5:
            ls, lsc = local_search(sel, deadline - 2, rng)
            if lsc > best_c:
                best_c = lsc
                best = ls

    if best is None:
        best = greedy_construct(list(range(1, N + 1)))
    return sorted(best)
