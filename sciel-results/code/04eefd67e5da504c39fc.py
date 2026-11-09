import time
import math
import random

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _spiral_order(limit):
    R = limit
    cands = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            hd = _hexdist(a,b)
            x = a + 0.5*b
            y = 0.8660254037844386*b
            ang = math.atan2(y, x)
            cands.append((hd, ang, a, b))
    cands.sort()
    out = []
    for hd, ang, a, b in cands:
        out.append((a,b))
    return out

def construct():
    deadline = time.time() + 120
    n = 32
    order = _spiral_order(12)
    best = list(order[:n])
    best_e = _count_edges(best)
    in_set = set(best)

    # Build a larger candidate pool
    pool_set = set()
    for r in range(10):
        for a in range(-r, r+1):
            for b in range(-r, r+1):
                if _hexdist(a,b) == r:
                    pool_set.add((a,b))
    pool = list(pool_set)

    rng = random.Random(0)
    it = 0
    no_improve = 0
    max_no_improve = 50000

    # Phase 1: Greedy improvement
    while time.time() < deadline and no_improve < max_no_improve:
        it += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            no_improve += 1
            continue

        edges_lost = 0
        for da, db in _NEIGH:
            if (cur[0]+da, cur[1]+db) in in_set:
                edges_lost += 1
        edges_gained = 0
        for da, db in _NEIGH:
            nb = (cand[0]+da, cand[1]+db)
            if nb in in_set and nb != cur:
                edges_gained += 1

        delta = edges_gained - edges_lost
        if delta > 0:
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            no_improve = 0
        else:
            no_improve += 1

    # Phase 2: Simulated annealing to escape local optima
    if time.time() < deadline:
        T_start = 2.0
        T_end = 0.01
        cool_rate = 0.99995
        T = T_start
        sa_no_improve = 0
        max_sa_no_improve = 100000

        while T > T_end and time.time() < deadline and sa_no_improve < max_sa_no_improve:
            it += 1
            i = rng.randrange(n)
            cur = best[i]
            cand = pool[rng.randrange(len(pool))]
            if cand == cur or cand in in_set:
                sa_no_improve += 1
                T *= cool_rate
                continue

            edges_lost = 0
            for da, db in _NEIGH:
                if (cur[0]+da, cur[1]+db) in in_set:
                    edges_lost += 1
            edges_gained = 0
            for da, db in _NEIGH:
                nb = (cand[0]+da, cand[1]+db)
                if nb in in_set and nb != cur:
                    edges_gained += 1

            delta = edges_gained - edges_lost

            if delta > 0:
                best_e += delta
                best[i] = cand
                in_set.discard(cur)
                in_set.add(cand)
                sa_no_improve = 0
            else:
                # Accept with probability exp(delta / T)
                if delta < 0 and rng.random() < math.exp(delta / T):
                    best_e += delta
                    best[i] = cand
                    in_set.discard(cur)
                    in_set.add(cand)
                    sa_no_improve = 0
                else:
                    sa_no_improve += 1

            T *= cool_rate

    # Phase 3: Final greedy polish
    if time.time() < deadline:
        no_improve = 0
        max_no_improve = 50000
        while time.time() < deadline and no_improve < max_no_improve:
            it += 1
            i = rng.randrange(n)
            cur = best[i]
            cand = pool[rng.randrange(len(pool))]
            if cand == cur or cand in in_set:
                no_improve += 1
                continue

            edges_lost = 0
            for da, db in _NEIGH:
                if (cur[0]+da, cur[1]+db) in in_set:
                    edges_lost += 1
            edges_gained = 0
            for da, db in _NEIGH:
                nb = (cand[0]+da, cand[1]+db)
                if nb in in_set and nb != cur:
                    edges_gained += 1

            delta = edges_gained - edges_lost
            if delta > 0:
                best_e += delta
                best[i] = cand
                in_set.discard(cur)
                in_set.add(cand)
                no_improve = 0
            else:
                no_improve += 1

    # Final verification and return
    return [[a,b] for (a,b) in best]
