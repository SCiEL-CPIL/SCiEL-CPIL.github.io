"""Rewritten: search-based construction of a dense 32-point hexagonal blob
on the triangular lattice, maximizing unit-distance edges (target 76)."""
import time
import random

_START = None
_DEADLINE = None

def _now():
    return time.time()

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _hexdist(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return (abs(da) + abs(da + db) + abs(db)) // 2

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    if len(Sset) != 32:
        return False
    return all(abs(a) <= 64 and abs(b) <= 64 for a, b in Sset)

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _delta_edges(Sset, p, q):
    """Return change in edge count when removing p and adding q (both not in Sset for p, q not in Sset)."""
    # Edges lost: edges from p to Sset (excluding p itself)
    lost = sum(1 for n in _neighbors(p) if n in Sset)
    # Edges gained: edges from q to Sset (excluding p if p was a neighbor, but p is being removed)
    gained = sum(1 for n in _neighbors(q) if n in Sset and n != p)
    return gained - lost

def _local_search(Sset, pool, swap_cand, deadline, rng, max_passes=1000):
    S = list(Sset)
    Sset = set(S)
    cur = _edge_count(Sset)
    for _ in range(max_passes):
        if _now() > deadline - 2.0:
            break
        improved = False
        order = list(range(len(S)))
        rng.shuffle(order)
        for i in order:
            p = S[i]
            # Find best swap for this point
            best_q = None
            best_delta = 0
            for q in swap_cand[p]:
                if q in Sset:
                    continue
                d = _delta_edges(Sset, p, q)
                if d > best_delta:
                    best_delta = d
                    best_q = q
            if best_q is not None:
                Sset.discard(p)
                Sset.add(best_q)
                S[i] = best_q
                cur += best_delta
                improved = True
        if not improved:
            break
    return Sset, cur

def construct():
    global _START, _DEADLINE
    if _START is None:
        _START = time.time()
        _DEADLINE = _START + 110.0   # safety margin before the 120 s contract limit

    rng = random.Random(20240517)

    # Candidate pool: hexagon of radius 5 (plenty of room; all coords <= 5 <= 64).
    pool = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= 5:
                pool.append((a, b))
    poolset = set(pool)

    # Precompute swap candidates: pool points within hex-distance 2 of each pool point.
    swap_cand = {}
    for p in pool:
        swap_cand[p] = [q for q in pool if q != p and _hexdist(p, q) <= 2]

    best = None
    best_edges = -1

    def consider(Sset):
        nonlocal best, best_edges
        if _valid(Sset):
            e = _edge_count(Sset)
            if e > best_edges:
                best_edges = e
                best = [list(t) for t in Sset]

    # Seed sets: pure center, hexagon radius 2 (19 pts), and random center-ish seeds.
    center = (0, 0)
    hex2 = [p for p in pool if _hexdist(p, center) <= 2]  # 19 points

    seeds = [
        [center],
        list(hex2),
        [(0, 0), (1, 0), (0, 1)],
    ]
    # a couple of random 3-point seeds near the centre
    for _ in range(4):
        seeds.append(rng.sample(hex2, 3))

    def grow_to_32(seed):
        Sset = set(seed)
        Sset = {p for p in Sset if p in poolset}
        if len(Sset) >= 32:
            return set(list(Sset)[:32])
        while len(Sset) < 32:
            if _now() > _DEADLINE - 3.0:
                return Sset
            # pick pool point (not in Sset) maximising new edges, random tie-break
            best_pts = []
            best_new = -1
            for q in pool:
                if q in Sset:
                    continue
                new_edges = sum(1 for n in _neighbors(q) if n in Sset)
                if new_edges > best_new:
                    best_new = new_edges
                    best_pts = [q]
                elif new_edges == best_new:
                    best_pts.append(q)
            if not best_pts:
                break
            Sset.add(rng.choice(best_pts))
        return Sset

    restarts = 0
    for seed in seeds:
        if _now() > _DEADLINE - 5.0:
            break
        restarts += 1
        Sset = grow_to_32(seed)
        if len(Sset) != 32:
            continue
        Sset, _ = _local_search(Sset, pool, swap_cand, _DEADLINE, rng)
        consider(Sset)

    # Extra randomized restarts while time remains, to shake out better local optima.
    while _now() < _DEADLINE - 6.0:
        restarts += 1
        # random starting blob: take a random 6-point seed around the centre
        seed = rng.sample(hex2, rng.randint(3, 7))
        Sset = grow_to_32(seed)
        if len(Sset) != 32:
            continue
        Sset, _ = _local_search(Sset, pool, swap_cand, _DEADLINE, rng)
        consider(Sset)
        if best_edges >= 76:
            # reached the Harborth optimum; stop early
            break

    if best is None:
        # Guaranteed-valid fallback (the 16x2 strip, the seed program's answer).
        best = [[a, b] for a in range(16) for b in range(2)]

    # Final sanity: ensure exactly 32 distinct points within bounds.
    assert len(best) == 32
    assert len(set(tuple(p) for p in best)) == 32
    assert all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in best)
    return best
