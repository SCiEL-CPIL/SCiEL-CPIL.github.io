import time
import numpy as np

_N = 256
_DEADLINE = 84.0


def _max_abs_ap_sum(signs):
    n = len(signs)
    if isinstance(signs, list):
        signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) <= 1:
                continue
            csum = np.cumsum(seq)
            rmin = csum[0]
            rmax = csum[0]
            for j in range(1, len(csum)):
                v = csum[j]
                if v - rmin > max_disc:
                    max_disc = v - rmin
                if rmax - v > max_disc:
                    max_disc = rmax - v
                if v < rmin:
                    rmin = v
                if v > rmax:
                    rmax = v
    return max_disc


def construct():
    t0 = time.time()
    n = _N
    
    # Greedy construction
    signs = np.zeros(n, dtype=np.int32)
    for i in range(n):
        best = 1
        best_val = 999
        for s in (1, -1):
            signs[i] = s
            val = _max_abs_ap_sum(signs[:i+1])
            if val < best_val:
                best_val = val
                best = s
        signs[i] = best
    
    signs_list = signs.tolist()
    disc = _max_abs_ap_sum(signs_list)
    
    # Local search
    deadline = _DEADLINE
    improved = True
    while improved and (time.time() - t0) < deadline - 2:
        improved = False
        best_pos = -1
        best_disc = disc
        for i in range(n):
            old = signs_list[i]
            signs_list[i] = -old
            d = _max_abs_ap_sum(signs_list)
            signs_list[i] = old
            if d < best_disc:
                best_disc = d
                best_pos = i
        if best_pos >= 0:
            signs_list[best_pos] = -signs_list[best_pos]
            disc = best_disc
            improved = True
    
    return signs_list
