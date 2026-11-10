# EVOLVE-BLOCK-START
import time
import random

_N = 256
_KILL = 90.0
_MARGIN = 8.0

def _check3(a, b, x):
    return (2 * x == a + b) or (2 * a == x + b) or (2 * b == x + a)

def _greedy_init(rng, direction='asc'):
    S = []
    order = list(range(1, _N + 1))
    if direction == 'desc':
        order = order[::-1]
    elif direction == 'rand':
        rng.shuffle(order)
    for x in order:
        ok = True
        L = len(S)
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                if _check3(a, b, x):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            S.append(x)
    return S

def _conflict(S):
    conf = [0] * (_N + 1)
    S = list(S)
    m = len(S)
    for i in range(m):
        a = S[i]
        for j in range(i + 1, m):
            b = S[j]
            x = 2 * b - a
            if 1 <= x <= _N:
                conf[x] += 1
            x = 2 * a - b
            if 1 <= x <= _N:
                conf[x] += 1
            s = a + b
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= _N:
                    conf[x] += 1
    return conf

def _add_upd(conf, other, s):
    for a in other:
        x = 2 * s - a
        if 1 <= x <= _N:
            conf[x] += 1
        x = 2 * a - s
        if 1 <= x <= _N:
            conf[x] += 1
        t = s + a
        if t % 2 == 0:
            x = t // 2
            if 1 <= x <= _N:
                conf[x] += 1

def _rem_upd(conf, other, s):
    for a in other:
        x = 2 * s - a
        if 1 <= x <= _N:
            conf[x] -= 1
        x = 2 * a - s
        if 1 <= x <= _N:
            conf[x] -= 1
        t = s + a
        if t % 2 == 0:
            x = t // 2
            if 1 <= x <= _N:
                conf[x] -= 1

def _hill_climb(S0, inv, deadline):
    S = set(S0)
    conf = _conflict(S)
    weight = 0.0
    for x in S:
        weight += inv[x]
    best_S = set(S)
    best_w = weight
    eps = 1e-12
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = -1.0
        best_typ = None
        best_x = None
        best_y = None
        # candidate: best single add
        for x in range(1, _N + 1):
            if x not in S and conf[x] == 0:
                g = inv[x]
                if g > best_gain:
                    best_gain = g
                    best_typ = 'add'
                    best_x = x
                    best_y = None
        # candidate: best swap (remove x -> add smallest addable y)
        # Optimization: Pre-compute list of addable elements? No, conf changes.
        # We just iterate.
        for x in list(S):
            S.discard(x)
            _rem_upd(conf, S, x)
            w0 = weight - inv[x]
            # find smallest addable y (highest inv[y])
            # Since inv[y] is decreasing with y, we scan from 1 upwards
            for y in range(1, _N + 1):
                if y not in S and conf[y] == 0:
                    g = inv[y] - inv[x]
                    if g > best_gain:
                        best_gain = g
                        best_typ = 'swap'
                        best_x = x
                        best_y = y
                    break
            # restore
            _add_upd(conf, S, x)
            S.add(x)
            weight = w0 + inv[x]
        if best_gain > eps:
            if best_typ == 'add':
                _add_upd(conf, S, best_x)
                S.add(best_x)
                weight += inv[best_x]
            else:
                x = best_x
                y = best_y
                S.discard(x)
                _rem_upd(conf, S, x)
                weight -= inv[x]
                _add_upd(conf, S, y)
                S.add(y)
                weight += inv[y]
            if weight > best_w:
                best_w = weight
                best_S = set(S)
            improved = True
    return best_S

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    rng = random.Random(20240517)
    inv = [0.0] + [1.0 / x for x in range(1, _N + 1)]

    def w(S):
        return sum(inv[x] for x in S)

    best_S = None
    best_w = -1.0
    
    # Perform multiple restarts of Greedy + Hill Climb
    # The previous code only did 3. We can do many more.
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Vary the initialization strategy
        if num_restarts % 3 == 0:
            S_init = _greedy_init(rng, 'asc')
        elif num_restarts % 3 == 1:
            S_init = _greedy_init(rng, 'desc')
        else:
            S_init = _greedy_init(rng, 'rand')
        
        # Add some random noise to the initial set to escape local optima?
        # Or just rely on the hill climb.
        
        opt = _hill_climb(S_init, inv, deadline - 5.0)
        cw = w(opt)
        if cw > best_w:
            best_w = cw
            best_S = opt
        num_restarts += 1
        
        # If we've done a few and time is short, break
        if num_restarts > 50 and time.time() > deadline - 10.0:
            break

    if best_S is None or len(best_S) == 0:
        best_S = {1}

    res = sorted(best_S)
    # Defensive verification
    Sset = set(res)
    ok = True
    for a in res:
        for b in res:
            if a >= b:
                continue
            c = 2 * b - a
            if c in Sset and c > b:
                ok = False
    if not ok:
        S = _greedy_init(rng, 'asc')
        res = sorted(S)
    return res
# EVOLVE-BLOCK-END
