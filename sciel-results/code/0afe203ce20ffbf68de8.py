import time
import random

N = 256
DEADLINE = time.time() + 75

div = [[False]*(2*N+1) for _ in range(N+1)]
for a in range(1, N+1):
    for m in range(2*a, 2*N+1, a):
        div[a][m] = True

def _try_add(S, x):
    conf = set()
    for b in S:
        if b <= x: continue
        for c in S:
            if c < b: continue
            if div[x][b+c]:
                conf.add(b); conf.add(c)
    for a in S:
        if a >= x: continue
        for c in S:
            if c < x: continue
            if div[a][x+c]:
                conf.add(a); conf.add(c)
    for a in S:
        if a >= x: continue
        for b in S:
            if b < a or b > x: continue
            if div[a][b+x]:
                conf.add(a); conf.add(b)
    return conf

def _greedy_adds(cur, deadline):
    best = set(cur); best_size = len(cur)
    while time.time() < deadline:
        changed = False
        for x in range(1, N+1):
            if x in cur: continue
            if not _try_add(cur, x):
                cur.add(x); changed = True
                if len(cur) > best_size:
                    best_size = len(cur); best = set(cur)
        if not changed:
            break
    return best

def _refine(cur, deadline):
    best = set(cur); best_size = len(cur)
    rng = random.Random(7)
    while time.time() < deadline:
        r = rng.random()
        if r < 0.5:
            x = rng.randint(1, N)
            if x not in cur and not _try_add(cur, x):
                cur.add(x)
                if len(cur) > best_size:
                    best_size = len(cur); best = set(cur)
        elif r < 0.85 and cur:
            x = rng.randint(1, N)
            if x not in cur:
                y = rng.choice(tuple(cur))
                if not _try_add(set(cur) - {y}, x):
                    cur.discard(y); cur.add(x)
                    if len(cur) > best_size:
                        best_size = len(cur); best = set(cur)
        else:
            if len(cur) > 20:
                y = rng.choice(tuple(cur))
                cur.discard(y)
    return best

def construct():
    random.seed(99)
    best = set(range(171, 257))
    starts = [best]
    # a couple random starts in upper region
    for _ in range(3):
        st = set(range(150, 257))
        # randomize by dropping some
        for e in list(st):
            if random.random() < 0.3:
                st.discard(e)
        starts.append(st)
    for st in starts:
        if time.time() > DEADLINE - 5: break
        g = _greedy_adds(set(st), DEADLINE - 2)
        r = _refine(g, DEADLINE)
        if len(r) > len(best):
            best = r
    return list(best)
