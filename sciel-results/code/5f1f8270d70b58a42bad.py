"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

_DEADLINE = time.time() + 85  # 5s safety margin before 90s kill

def _score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]; b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _greedy_build(values, m, seed):
    """Build m edges over 14 vertices to minimize distinct (sum,prod) pairs."""
    rng = random.Random(seed)
    n = len(values)
    # all value-pairs (a,b) with a<=b, a!=b
    pairs = []
    for a in range(n):
        for b in range(a+1, n):
            if values[a] != values[b]:
                pairs.append((a, b, values[a]+values[b], values[a]*values[b]))
    if not pairs:
        return None
    # group by (sum,prod)
    group = defaultdict(list)
    for a, b, s, p in pairs:
        group[(s, p)].append((a, b))

    # pick a starting pair
    order = list(group.keys())
    rng.shuffle(order)
    used = set()          # used vertex-edges
    edges = []
    active = set(order)   # still-usable groups
    # start with a group that has many edges
    start = max(order, key=lambda k: len(group[k]))
    for a, b in group[start]:
        if (a, b) not in used:
            used.add((a, b)); edges.append((a, b))
            if len(edges) >= m:
                break
    if len(edges) >= m:
        return edges[:m]

    # repeatedly add an edge from an active group, preferring groups already used
    # and avoiding vertex overload
    deg = defaultdict(int)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    # order groups by (already-used desc, size desc)
    def gkey(k):
        cnt = sum(1 for a, b in group[k] if (a, b) in used)
        return (-cnt, -len(group[k]))
    # loop
    while len(edges) < m:
        # candidate groups: those with an unused edge
        cand = []
        for k in active:
            for a, b in group[k]:
                if (a, b) not in used:
                    cand.append((k, a, b))
                    break
        if not cand:
            return None
        # prefer groups with more already-used edges, then fewer high-degree vertices
        def ckey(item):
            k, a, b = item
            cnt = sum(1 for x, y in group[k] if (x, y) in used)
            return (-cnt, deg[a] + deg[b])
        cand.sort(key=ckey)
        chosen = cand[0]
        k, a, b = chosen
        used.add((a, b)); edges.append((a, b))
        deg[a] += 1; deg[b] += 1
        if len(edges) >= m:
            break
    return edges[:m]

def _local_search(values, edges, iters, seed):
    rng = random.Random(seed)
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    all_edges = [e for e in all_edges if values[e[0]] != values[e[1]]]
    cur = set(edges)
    cur_score = _score(values, list(cur))
    best_score = cur_score
    best = list(cur)
    for _ in range(iters):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 1:
            break
        u, v = rng.choice(list(cur))
        cand = rng.choice(all_edges)
        if cand in cur:
            continue
        new = set(cur); new.discard((u, v)); new.add(cand)
        ns = _score(values, list(new))
        if ns <= cur_score:
            cur = new; cur_score = ns
            if ns < best_score:
                best_score = ns; best = list(new)
    return best_score, best

def construct():
    n = 14
    m = 35
    best_score = m
    best_result = None
    rng = random.Random(7)

    # base pool of small highly-composite numbers
    pool = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 30, 32, 36, 40, 45, 48, 50, 60, 72, 80, 90]

    candidates = []

    # 1. pure small highly-composite multisets with heavy duplicates
    for d in [2, 3, 4]:
        base = pool[:d]
        for _ in range(30):
            vals = [base[rng.randrange(d)] for _ in range(n)]
            candidates.append(vals)

    # 2. two-value multisets (a, b)
    for a in pool[:12]:
        for b in pool[:12]:
            if a == b:
                continue
            for k in range(1, n):
                vals = [a]*k + [b]*(n-k)
                candidates.append(vals)

    # 3. three-value multisets
    for a in pool[:8]:
        for b in pool[:8]:
            for c in pool[:8]:
                if len({a, b, c}) < 2:
                    continue
                for _ in range(2):
                    vals = [a, b, c] + [rng.choice([a, b, c]) for _ in range(n-3)]
                    candidates.append(vals)

    # 4. consecutive small
    for start in range(1, 6):
        for step in range(1, 4):
            vals = [start + i*step for i in range(n)]
            if max(vals) <= 10000:
                candidates.append(vals)

    # 5. random small multisets
    for _ in range(60):
        vals = [rng.choice(pool[:10]) for _ in range(n)]
        candidates.append(vals)

    # 6. powers of 2 (baseline)
    candidates.append([2**i for i in range(n)])

    seen = set()
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        if len(set(vals)) < 2:
            continue
        if any(v < 1 or v > 10000 for v in vals):
            continue
        key = tuple(sorted(vals))
        if key in seen:
            continue
        seen.add(key)

        # quick pre-filter: distinct products among all value-pairs
        dvals = list(set(vals))
        prods = set()
        sums = set()
        for a, b in combinations(dvals, 2):
            prods.add(a*b); sums.add(a+b)
        # also same-value pairs can't exist (distinct values)
        if max(len(prods), len(sums)) >= best_score:
            continue

        # greedy build a few times with different seeds
        for s in range(3):
            if time.time() > _DEADLINE:
                break
            edges = _greedy_build(vals, m, 100 + s)
            if edges is None or len(edges) != m:
                continue
            sc = _score(vals, edges)
            # local search
            sc2, edges2 = _local_search(vals, edges, 400, 200 + s)
            if sc2 < best_score:
                best_score = sc2
                best_result = (list(vals), [[int(u), int(v)] for u, v in edges2])
                if best_score <= 1:
                    break

    if best_result is None:
        # fallback: consecutive, first m edges
        values = list(range(1, n+1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result

    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
