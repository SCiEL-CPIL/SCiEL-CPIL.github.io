import time

_T0 = time.time()
_DEADLINE = 130.0  # safety margin before 150 s wall-clock kill


def _tleft():
    return _DEADLINE - (time.time() - _T0)


def _lw26():
    # Lazebnik–Woldar algebraic construction: R_5(C4) >= 26.
    p = 5

    def mul(x, y):
        a1, b1 = x
        a2, b2 = y
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)

    def pw(x, n):
        r = (1, 0)
        base = x
        while n:
            if n & 1:
                r = mul(r, base)
            base = mul(base, base)
            n >>= 1
        return r

    # find primitive element in GF(25)^*
    prim = None
    for a in range(p):
        for b in range(p):
            if a == 0 and b == 0:
                continue
            e = (a, b)
            ok = True
            for d in (12, 8, 6, 4, 3, 2):
                if pw(e, d) == (1, 0):
                    ok = False
                    break
            if ok:
                prim = e
                break
        if prim:
            break

    if prim is None:
        return None

    # log table
    log = {}
    cur = (1, 0)
    for i in range(24):
        log[cur] = i
        cur = mul(cur, prim)

    # vertex order: 0 -> (0,0), 1..24 -> powers, 25 -> infinity
    elems = [(0, 0)] + list(log.keys())
    n = 26
    colors = []

    for i in range(n):
        for j in range(i + 1, n):
            if i == 25 or j == 25:
                other = j if i == 25 else i
                if other == 0:
                    colors.append(0)
                else:
                    colors.append((log[elems[other]] % 4) + 1)
            elif i == 0 or j == 0:
                colors.append(0)
            else:
                d = (log[elems[i]] - log[elems[j]]) % 24
                colors.append((d % 4) + 1)

    return {'n': n, 'colors': colors}


def _validate(n, cols):
    if len(cols) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in cols):
        return False
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = cols[k]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            k += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True


def construct():
    if _tleft() <= 0:
        return {'n': 1, 'colors': [0]}
    res = _lw26()
    if res is not None and _validate(26, res['colors']):
        return res
    return {'n': 1, 'colors': [0]}
