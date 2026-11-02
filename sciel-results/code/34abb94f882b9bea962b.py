"""Second autocorrelation inequality: maximize ||h||2^2 / (||h||1 * ||h*h||inf)
via disjoint spikes on a Sidon set to keep autoconvolution peak minimal."""
import numpy as np

N = 50

def _sidon_positions(n, max_pos):
    """Greedily build a Sidon set (all pairwise sums distinct) within [0, max_pos]."""
    pos = []
    sums = set()
    for p in range(max_pos + 1):
        ok = True
        for i, a in enumerate(pos):
            s = a + p
            if s in sums:
                ok = False
                break
        if ok:
            # check p+p and p+existing not colliding with existing sums
            for a in pos:
                if (a + p) in sums:
                    ok = False
                    break
            if ok:
                pos.append(p)
                for a in pos:
                    sums.add(a + p)
        if len(pos) >= n:
            break
    return pos

def construct():
    # Use a Sidon set of spike positions to minimize autoconvolution overlap.
    # Greedy Sidon set from the largest positions tends to be sparse and valid.
    pos = _sidon_positions(50, N - 1)
    h = np.zeros(N)
    for p in pos:
        h[p] = 1.0
    return list(h)
