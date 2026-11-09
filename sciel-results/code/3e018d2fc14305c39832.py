# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 10
_DEMANDS = [
    (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
    (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
    (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
]
_D = 15
# CW[d] = True -> route clockwise (edges s..t-1); False -> counter-clockwise (all other edges)
_CW = np.zeros(_D, dtype=bool)
_CCW = np.zeros(_D, dtype=bool)
for _d, (_s, _t, _w) in enumerate(_DEMANDS):
    _CW[_d] = True
    _CCW[_d] = False


def _edge_loads(cw):
    loads = np.zeros(_N)
    for d in range(_D):
        s, t, w = _DEMANDS[d]
        if cw[d]:
            e = s
            while e != t:
                loads[e] += w
                e = (e + 1) % _N
        else:
            e = t
            while e != s:
                loads[e] += w
                e = (e + 1) % _N
    return loads


def _flip(d, cw):
    cw = cw.copy()
    cw[d] = not cw[d]
    return cw


def _local_search(cw):
    loads = _edge_loads(cw)
    improved = True
    while improved:
        improved = False
        for d in range(_D):
            cand = _flip(d, cw)
            cand_loads = _edge_loads(cand)
            if cand_loads.max() < loads.max():
                cw = cand
                loads = cand_loads
                improved = True
    # pair flips
    improved = True
    while improved:
        improved = False
        for d1 in range(_D):
            for d2 in range(d1 + 1, _D):
                cand = _flip(d2, _flip(d1, cw))
                cand_loads = _edge_loads(cand)
                if cand_loads.max() < loads.max():
                    cw = cand
                    loads = cand_loads
                    improved = True
    return cw


def ring_routing() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0  # safety margin before 600s limit

    best_cw = np.zeros(_D, dtype=bool)
    best_loads = _edge_loads(best_cw)
    best_max = best_loads.max()

    rng = np.random.default_rng(0)

    def consider(cw):
        nonlocal best_cw, best_loads, best_max
        cw = _local_search(cw)
        loads = _edge_loads(cw)
        m = loads.max()
        if m < best_max:
            best_cw, best_loads, best_max = cw, loads, m

    # deterministic greedy initial solution
    cw = np.zeros(_D, dtype=bool)
    for d in range(_D):
        cw1 = cw.copy(); cw1[d] = True
        cw0 = cw.copy(); cw0[d] = False
        if _edge_loads(cw1).max() <= _edge_loads(cw0).max():
            cw[d] = True
        else:
            cw[d] = False
    consider(cw)

    # random restarts
    while time.time() < deadline:
        cw = (rng.random(_D) < 0.5).astype(bool)
        consider(cw)

    out = np.zeros(_D)
    out[best_cw] = 1.0  # >= 0.5 -> counter-clockwise
    out[~best_cw] = 0.0
    return out


# EVOLVE-BLOCK-END
