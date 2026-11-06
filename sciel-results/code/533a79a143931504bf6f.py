import time, random, math

def construct():
    start = time.time()
    DEADLINE = start + 78.0
    N = 180

    # Build all triples (a<b<c) with 1/a = 1/b + 1/c  <=>  (b-a)(c-a)=a^2
    triples = []
    seen = set()
    for a in range(1, N + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d != e:
                    b = a + d
                    c = a + e
                    if b <= N and c <= N:
                        t = (a, b, c) if b < c else (a, c, b)
                        if t not in seen:
                            seen.add(t)
                            triples.append(t)
            d += 1

    adj = [[] for _ in range(N + 1)]
    for (a, b, c) in triples:
        adj[a].append((b, c))
        adj[b].append((a, c))
        adj[c].append((a, b))

    def count_active(p):
        cnt = 0
        for (a, b, c) in triples:
            if p[a] and p[b] and p[c]:
                cnt += 1
        return cnt

    def make_present(lo, hi):
        p = [False] * (N + 1)
        for v in range(lo, hi + 1):
            p[v] = True
        return p

    def greedy_grow(p):
        p = p[:]
        improved = True
        while improved:
            improved = False
            for v in range(1, N + 1):
                if not p[v]:
                    ok = True
                    for (u, w) in adj[v]:
                        if p[u] and p[w]:
                            ok = False
                            break
                    if ok:
                        p[v] = True
                        improved = True
        return p

    p0 = make_present(91, 180)
    starts = [p0, greedy_grow(p0)]

    best_size = 0
    best_set = None
    for p in starts:
        act = count_active(p)
        sz = sum(p[1:])
        if act == 0 and sz > best_size:
            best_size = sz
            best_set = p[:]
    if best_set is None:
        best_set = p0[:]
        best_size = sum(p0[1:])

    random.seed(12345)
    for p0 in starts:
        if time.time() > DEADLINE:
            break
        present = p0[:]
        cur_active = count_active(present)
        cur_size = sum(present[1:])
        T = 3.0
        TMIN = 0.05
        while time.time() < DEADLINE and T > TMIN:
            v = random.randint(1, N)
            old = present[v]
            dact = 0
            for (u, w) in adj[v]:
                if present[u] and present[w]:
                    dact += (1 if not old else 0) - (1 if old else 0)
            dsize = 1 if not old else -1
            ds = dsize - 100000 * dact
            if ds >= 0 or random.random() < math.exp(ds / T):
                present[v] = not old
                cur_active += dact
                cur_size += dsize
                if cur_active == 0 and cur_size > best_size:
                    best_size = cur_size
                    best_set = present[:]
            T *= 0.9995

    if time.time() < DEADLINE:
        g = greedy_grow(best_set)
        if count_active(g) == 0 and sum(g[1:]) > best_size:
            best_size = sum(g[1:])
            best_set = g

    res = [v for v in range(1, N + 1) if best_set[v]]
    if not res:
        res = list(range(91, 181))
    return res
