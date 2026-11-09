# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_MARGIN = 8.0

def _build():
    n = 10
    D = 15
    demands = [
        (0,5,3),(1,7,2),(2,9,4),(3,6,1),(0,4,5),(2,8,2),(5,9,3),(1,4,2),
        (6,0,4),(3,8,1),(7,2,3),(4,9,2),(8,1,5),(0,6,1),(5,2,2),
    ]
    s = np.array([d[0] for d in demands])
    t = np.array([d[1] for d in demands])
    w = np.array([d[2] for d in demands], dtype=float)

    cw = np.zeros((n, D), dtype=float)   # 1 if edge e is on clockwise path s->t
    ccw = np.zeros((n, D), dtype=float)  # 1 if edge e is on counter-clockwise path
    for e in range(n):
        for j in range(D):
            a, b = s[j], t[j]
            if a <= b:
                cw[e, j] = 1.0 if (a <= e < b) else 0.0
            else:
                cw[e, j] = 1.0 if (e >= a or e < b) else 0.0
            ccw[e, j] = 1.0 - cw[e, j]

    delta = ccw - cw   # [n, D]; flipping j adds w[j]*delta[:, j] to loads
    return n, D, s, t, w, cw, ccw, delta

def _loads(r, cw, w):
    return cw @ (w * r)

def _maxload(r, cw, w):
    return float(np.max(cw @ (w * r)))

def ring_routing() -> np.ndarray:
    t0 = time.time()
    n, D, s, t, w, cw, ccw, delta = _build()

    # ---- Phase 1: deterministic vectorized greedy local search ----
    r = np.zeros(D)
    L = _loads(r, cw, w)
    best = float(np.max(L))
    improved = True
    while improved and (time.time() - t0) < _DEADLINE - _MARGIN:
        improved = False
        best_e = int(np.argmax(L))
        # candidate gains for flipping each j (new max after flipping)
        cand = np.max(L + w * delta[best_e, :], axis=0)
        j = int(np.argmin(cand))
        new_max = float(cand[j])
        if new_max < best - 1e-12:
            r[j] = 1.0 - r[j]
            L = _loads(r, cw, w)
            best = new_max
            improved = True

    # ---- Phase 2: smooth epigraph polish (soft max via log-sum-exp) ----
    x = r.copy()
    beta = 4.0
    lr = 0.5
    for _ in range(20000):
        if (time.time() - t0) > _DEADLINE - _MARGIN:
            break
        L = cw @ (w * x)
        m = np.max(L)
        u = L - m
        sm = np.log(np.sum(np.exp(beta * u))) / beta   # soft max
        dsm = np.exp(beta * u) / np.sum(np.exp(beta * u))  # softmax weights
        g = cw.T @ dsm
        x = x + lr * g
        x = np.clip(x, 0.0, 1.0)

    r2 = (x >= 0.5).astype(float)
    m2 = _maxload(r2, cw, w)
    if m2 < best - 1e-12:
        r = r2
        best = m2

    # ---- Phase 3: exact local-search polish on the best integer point ----
    L = _loads(r, cw, w)
    best = float(np.max(L))
    improved = True
    while improved and (time.time() - t0) < _DEADLINE - _MARGIN:
        improved = False
        best_e = int(np.argmax(L))
        cand = np.max(L + w * delta[best_e, :], axis=0)
        j = int(np.argmin(cand))
        new_max = float(cand[j])
        if new_max < best - 1e-12:
            r[j] = 1.0 - r[j]
            L = _loads(r, cw, w)
            best = new_max
            improved = True

    return r

# EVOLVE-BLOCK-END
