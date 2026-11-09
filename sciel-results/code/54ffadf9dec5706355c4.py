import time
import numpy as np

N = 16
TOTAL_ZERO_ONE = 1 << N

T_START = time.time()
DEADLINE = T_START + 82.0

def _verify(comp_list):
    if len(comp_list) == 0:
        return True
    comps = np.asarray(comp_list, dtype=np.int64)
    idx = np.arange(TOTAL_ZERO_ONE)
    # (65536, 16) uint8 array
    bits = (idx[:, None] >> np.arange(N)) & 1
    for i, j in comps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi
    return bool(np.all(bits[:, :-1] <= bits[:, 1:]))

def _batcher_oe_merge_sort(n):
    def oe_merge(left, right):
        comps = []
        h = len(left)
        if h == 1:
            comps.append((left[0], right[0]))
            return comps
        le = [left[k] for k in range(0, h, 2)]
        lo = [left[k] for k in range(1, h, 2)]
        re = [right[k] for k in range(0, h, 2)]
        ro = [right[k] for k in range(1, h, 2)]
        comps += oe_merge(le, re)
        comps += oe_merge(lo, ro)
        # Fixed: Compare left[k] and right[k-1] for k=1, 3, ...
        for k in range(1, h, 2):
            comps.append((left[k], right[k - 1]))
        return comps

    def sort_block(wires):
        m = len(wires)
        if m == 1:
            return []
        h = m // 2
        left = wires[:h]
        right = wires[h:]
        comps = []
        comps += sort_block(left)
        comps += sort_block(right)
        comps += oe_merge(left, right)
        return comps

    return sort_block(list(range(n)))

def _remove_redundant(comp_list):
    cur = list(comp_list)
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        i = 0
        while i < len(cur) and time.time() < DEADLINE:
            trial = cur[:i] + cur[i + 1:]
            if _verify(trial):
                cur = trial
                changed = True
            else:
                i += 1
    return cur

def construct():
    network = _batcher_oe_merge_sort(N)
    
    if not _verify(network):
        # Fallback
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
        
    best = network
    best = _remove_redundant(best)
    
    if time.time() < DEADLINE - 2.0:
        if not _verify(best):
            best = network
            
    return [(int(i), int(j)) for (i, j) in best]
