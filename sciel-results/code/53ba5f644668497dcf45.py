"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify(network):
    """Return True iff network sorts all 2^N Boolean inputs (zero-one principle)."""
    n_in = 1 << N
    idx = np.arange(n_in, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(N, dtype=np.uint32)) & 1).astype(np.uint8)
    for (i, j) in network:
        a = bits[:, i]
        b = bits[:, j]
        bits[:, i] = np.minimum(a, b)
        bits[:, j] = np.maximum(a, b)
    for k in range(N - 1):
        if np.any(bits[:, k] > bits[:, k + 1]):
            return False
    return True

def _bitonic_network():
    """Valid bitonic sort network for N=16 (ascending output)."""
    comps = []
    def merge(a, c, up):
        mid = a + (c - a) // 2
        for i in range((c - a) // 2):
            if up:
                comps.append((a + i, mid + i))
            else:
                comps.append((mid + i, a + i))
    def sort(a, c, up):
        length = c - a
        if length <= 1:
            return
        mid = a + length // 2
        sort(a, mid, True)
        sort(mid, c, False)
        merge(a, c, up)
    sort(0, N, True)
    return comps

def _reduce(network, deadline):
    """Remove redundant comparators greedily; return a valid reduced network."""
    net = list(network)
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() >= deadline - 2.0:
                return net
            cand = net[:idx] + net[idx + 1:]
            if _verify(cand):
                net = cand
                improved = True
                break
    return net

def construct():
    deadline = time.time() + 80.0
    net = _bitonic_network()
    net = _reduce(net, deadline)
    return net

# EVOLVE-BLOCK-END
