import time
import itertools

_DEADLINE = 150.0
_MARGIN = 6.0

def _encode(q, n, word):
    v = 0
    for x in word:
        v = v * q + x
    return v

def _ball(q, n, radius, code_idx):
    # decode
    c = [0]*n
    v = code_idx
    for i in range(n-1, -1, -1):
        c[i] = v % q
        v //= q
    res = [code_idx]
    if radius >= 1:
        for pos in range(n):
            orig = c[pos]
            for a in range(q):
                if a == orig:
                    continue
                c[pos] = a
                vv = 0
                for x in c:
                    vv = vv * q + x
                res.append(vv)
            c[pos] = orig
    return res

def _greedy(balls, total, order):
    # order: 'max' remove most redundant, 'min' least, 'first' index order
    cov = [0]*total
    for b in balls:
        for w in b:
            cov[w] += 1
    active = [True]*len(balls)
    def removable(i):
        for w in balls[i]:
            if cov[w] < 2:
                return False
        return True
    while True:
        if order == 'max':
            best_i = -1; best_s = 0
            for i in range(len(balls)):
                if not active[i]: continue
                if not removable(i): continue
                m = 10**9
                for w in balls[i]:
                    if cov[w] < m: m = cov[w]
                if m > best_s:
                    best_s = m; best_i = i
        elif order == 'min':
            best_i = -1; best_s = 10**9
            for i in range(len(balls)):
                if not active[i]: continue
                if not removable(i): continue
                m = 10**9
                for w in balls[i]:
                    if cov[w] < m: m = cov[w]
                if m < best_s:
                    best_s = m; best_i = i
        else:
            best_i = -1
            for i in range(len(balls)):
                if not active[i]: continue
                if removable(i):
                    best_i = i; break
        if best_i < 0:
            break
        active[best_i] = False
        for w in balls[best_i]:
            cov[w] -= 1
    # validate
    for w in range(total):
        if cov[w] < 1:
            return None
    return active

def construct(q, n, radius):
    t0 = time.time()
    total = q ** n
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        code = [list(w) for w in itertools.product(range(q), repeat=n)]
    seen = set()
    code = [c for c in code if not (tuple(c) in seen or seen.add(tuple(c)))]
    code_idx = [_encode(q, n, c) for c in code]
    balls = [_ball(q, n, radius, ci) for ci in code_idx]
    best = None
    for order in ('max', 'min', 'first'):
        if time.time() - t0 > _DEADLINE - _MARGIN:
            break
        act = _greedy(balls, total, order)
        if act is None:
            continue
        cnt = sum(act)
        if best is None or cnt < best[0]:
            best = (cnt, act)
    if best is None:
        # fallback: original code
        return code
    act = best[1]
    return [code[i] for i in range(len(code)) if act[i]]
