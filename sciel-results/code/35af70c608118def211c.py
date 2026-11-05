import numpy as np
import math

N = 16

_ALL = None
def _all_inputs():
    global _ALL
    if _ALL is None:
        bits = np.arange(1 << N, dtype=np.uint16)
        _ALL = ((bits[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
    return _ALL

def _verify(comps):
    net = _all_inputs().copy()
    for i, j in comps:
        a = net[:, i]; b = net[:, j]
        net[:, i] = np.minimum(a, b)
        net[:, j] = np.maximum(a, b)
    d = net[:, 1:] - net[:, :-1]
    return bool(np.all(d >= 0))

def _bitonic(lo, hi, asc):
    length = hi - lo
    if length <= 1:
        return []
    half = length // 2
    comps = []
    if asc:
        comps += _bitonic(lo, lo + half, True)
        comps += _bitonic(lo + half, hi, False)
    else:
        comps += _bitonic(lo, lo + half, False)
        comps += _bitonic(lo + half, hi, True)
    for i in range(lo, lo + half):
        if asc:
            comps.append((i, i + half))
        else:
            a = hi - 1 - (i + half)
            b = hi - 1 - i
            comps.append((min(a, b), max(a, b)))
    return comps

def _prune(comps):
    comps = list(comps)
    improved = True
    while improved:
        improved = False
        for idx in range(len(comps)):
            trial = comps[:idx] + comps[idx+1:]
            if _verify(trial):
                comps = trial
                improved = True
                break
    return comps

def construct():
    base = _bitonic(0, N, True)
    if not _verify(base):
        # fallback: odd-even transposition (known correct)
        base = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    pruned = _prune(base)
    if not _verify(pruned):
        pruned = base
    return pruned
