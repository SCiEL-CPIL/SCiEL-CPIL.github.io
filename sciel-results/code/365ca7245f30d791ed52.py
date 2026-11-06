import random, time, math
from collections import defaultdict

DEADLINE = 110.0

def _line_key(p, q):
    x1, y1 = p; x2, y2 = q
    dx = x2 - x1; dy = y2 - y1
    g = math.gcd(dx, dy)
    if g:
        dx //= g; dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx; dy = -dy
    A = dy; B = -dx; C = -(dy * x1 - dx * y1)
    g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g2:
        A //= g2; B //= g2; C //= g2
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A; B = -B; C = -C
    return (A, B, C)

def _score(pts):
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            k = _line_key(pi, pts[j])
            lines[k].add(i); lines[k].add(j)
    cnt = 0; valid = True
    for s in lines.values():
        L = len(s)
        if L == 4:
            cnt += 1
        elif L >= 5:
            valid = False
    return cnt, valid

def _inbounds(p):
    return abs(p[0]) <= 1000 and abs(p[1]) <= 1000

def construct():
    t0 = time.time()
    def time_left():
        return DEADLINE - (time.time() - t0)
    rng = random.Random(20240517)
    best = None; bestscore = -1
    def consider(pts):
        nonlocal best, bestscore
        s, v = _score(pts)
        if v and s > bestscore:
            bestscore = s; best = [list(p) for p in pts]

    n = 16
    # seeds
    seeds = []
    seeds.append([[x, y] for x in range(4) for y in range(4)])
    for ox in (-2, -1, 0, 1, 2):
        for oy in (-2, -1, 0, 1, 2):
            seeds.append([[x + ox, y + oy] for x in range(4) for y in range(4)])
    # scaled grid
    seeds.append([[2*x, 2*y] for x in range(4) for y in range(4)])
    for seed in seeds:
        consider(seed)

    def dedupe(pts):
        seen = set(); out = []
        for p in pts:
            t = tuple(p)
            if t not in seen:
                seen.add(t); out.append(list(p))
        return out

    def rand_seed():
        pts = []
        seen = set()
        while len(pts) < n:
            p = [rng.randint(-15, 15), rng.randint(-15, 15)]
            if tuple(p) not in seen:
                seen.add(tuple(p)); pts.append(p)
        return pts

    def gen_moves(cur):
        moves = []
        occ = {tuple(p) for p in cur}
        # 3-collinear completion
        lines = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                k = _line_key(cur[i], cur[j])
                lines[k].add(i)
        for k, idxs in lines.items():
            if len(idxs) == 3:
                il = list(idxs)
                a, b = cur[il[0]], cur[il[1]]
                dx = b[0] - a[0]; dy = b[1] - a[1]
                g = math.gcd(dx, dy)
                if g:
                    dx //= g; dy //= g
                for t in range(-8, 9):
                    cand = [a[0] + t * dx, a[1] + t * dy]
                    if _inbounds(cand) and tuple(cand) not in occ:
                        moves.append((il[0], cand))
        # local random
        for _ in range(60):
            i = rng.randrange(n)
            p = cur[i]
            cand = [p[0] + rng.randint(-4, 4), p[1] + rng.randint(-4, 4)]
            if _inbounds(cand) and tuple(cand) not in occ:
                moves.append((i, cand))
        # far kick
        i = rng.randrange(n)
        cand = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
        if tuple(cand) not in occ:
            moves.append((i, cand))
        return moves

    def polish(cur, curescore):
        # steepest single-point moves on line-through-pair candidates
        improved = True
        while improved and time_left() > 2.0:
            improved = False
            bestmove = None; bests = curescore
            occ = {tuple(p) for p in cur}
            cands = set()
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        a, b = cur[j], cur[k]
                        dx = b[0] - a[0]; dy = b[1] - a[1]
                        g = math.gcd(dx, dy)
                        if g:
                            dx //= g; dy //= g
                        for t in range(-3, 4):
                            cand = (a[0] + t * dx, a[1] + t * dy)
                            if _inbounds(cand) and cand not in occ:
                                cands.add((i, cand))
            rng.shuffle(list(cands))
            for i, cand in list(cands):
                if time_left() < 2.0:
                    break
                trial = [list(p) for p in cur]; trial[i] = list(cand)
                s, v = _score(trial)
                if v and s > bests:
                    bests = s; bestmove = (i, cand)
            if bestmove:
                i, cand = bestmove
                cur = [list(p) for p in cur]; cur[i] = list(cand)
                curescore = bests
                improved = True
                consider(cur)
        return cur, curescore

    # main search
    cur = [list(p) for p in best] if best else seeds[0]
    curescore = bestscore
    for restart in range(20000):
        if time_left() < 4.0:
            break
        cur, curescore = polish(cur, curescore)
        # local random-walk hill climb
        for _ in range(60):
            if time_left() < 4.0:
                break
            moves = gen_moves(cur)
            if not moves:
                break
            rng.shuffle(moves)
            improved = False
            for i, cand in moves[:400]:
                trial = [list(p) for p in cur]; trial[i] = cand
                s, v = _score(trial)
                if v and s > curescore:
                    cur = trial; curescore = s; improved = True
                    consider(cur)
                    break
            if not improved:
                if moves:
                    i, cand = rng.choice(moves)
                    trial = [list(p) for p in cur]; trial[i] = cand
                    s, v = _score(trial)
                    if v:
                        cur = trial; curescore = s
                if rng.random() < 0.15:
                    cur = rand_seed()
                    s, v = _score(cur)
                    curescore = s if v else -1
                    consider(cur)
        cur, curescore = polish(cur, curescore)

    if best is None:
        best = seeds[0]
    return best
