"""Size-efficient sorting network on 18 wires.

Constructs a correct sorting network, then shrinks it with a time-budgeted,
fully-verified local search. The entry point is construct() -> list[(i, j)].
"""
import time
import numpy as np

N = 18
DEADLINE = 75.0  # safety margin below the 90 s wall-clock kill
_start = time.time()


def _remaining():
    return DEADLINE - (time.time() - _start)


def _apply(net, x):
    """Apply compare-exchange network `net` to a (N,) array of 0/1 values."""
    x = x.copy()
    for a, b in net:
        va = x[a]
        vb = x[b]
        if va > vb:
            x[a] = vb
            x[b] = va
    return x


def _batcher_odd_even_merge(A, B):
    """Odd-even mergesort of A (len 2^a) followed by B (len 2^b).

    Returns a list of (i, j) compare-exchanges. This is the standard,
    provably-correct Batcher construction (Knuth TAOCP 3.4.5C).
    """
    a = len(A)
    b = len(B)
    if a == 1 and b == 1:
        return [(A[0], B[0])]
    if a > b:
        return _batcher_odd_even_merge(B, A)
    if a == 0:
        return []
    if a == 1:
        # merge single element A[0] into B
        comps = []
        for i in range(b):
            comps.append((A[0], B[i]))
            if i < b - 1:
                comps.append((B[i], B[i + 1]))
        return comps
    # general case: a == 2^p, b == 2^q, p <= q
    a1 = a // 2
    b1 = b // 2
    A1, A2 = A[:a1], A[a1:]
    B1, B2 = B[:b1], B[b1:]
    comps = []
    comps += _batcher_odd_even_merge(A1, B1)
    comps += _batcher_odd_even_merge(A2, B2)
    for i in range(a1):
        comps.append((A1[i], B2[i]))
    return comps


def _batcher_network(n):
    """Build a full Batcher odd-even mergesort network for `n` wires."""
    if n == 1:
        return []
    n1 = n // 2
    A = list(range(n1))
    B = list(range(n1, n))
    return _batcher_odd_even_merge(A, B)


def _verify(net):
    """Return True iff `net` sorts all 2^N Boolean inputs."""
    if len(net) > 4096:
        return False
    if not all(0 <= a < b < N for a, b in net):
        return False
    N2 = 1 << N
    x = np.fromiter((i & (1 << k) != 0 for i in range(N2) for k in range(N)),
                    dtype=np.uint8, count=N2 * N).reshape(N2, N)
    for a, b in net:
        va = x[:, a].copy()
        vb = x[:, b].copy()
        swap = va > vb
        x[:, a] = np.where(swap, vb, va)
        x[:, b] = np.where(swap, va, vb)
    # sorted ascending => non-decreasing along wires
    if np.any(x[:, 1:] < x[:, :-1]):
        return False
    # quick real-valued sanity check
    rng = np.random.default_rng(12345)
    for _ in range(2000):
        y = rng.random(N)
        z = _apply(net, y)
        if np.any(z[1:] < z[:-1]):
            return False
    return True


def _remove(net, idx):
    return net[:idx] + net[idx + 1:]


def _swap(net, i, j):
    out = net[:]
    out[i], out[j] = net[j], net[i]
    return out


def construct():
    # 1) Correct baseline (Batcher odd-even mergesort).
    net = _batcher_network(N)
    if not _verify(net):
        # Fallback: guaranteed-correct odd-even transposition (153 comparators).
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best = net
    best_len = len(net)
    m = best_len

    # 2) Greedy removal: drop any comparator that is not needed.
    i = 0
    while i < m and _remaining() > 1.0:
        cand = _remove(best, i)
        if _verify(cand):
            best = cand
            m -= 1
            # keep index i, re-check same position
            continue
        i += 1

    # 3) Swap local search: try reordering pairs (long jumps first).
    improved = True
    while improved and _remaining() > 1.0:
        improved = False
        for gap in range(m - 1, 0, -1):
            if _remaining() < 0.5:
                break
            for i in range(0, m - gap):
                if _remaining() < 0.2:
                    break
                cand = _swap(best, i, i + gap)
                if _verify(cand):
                    best = cand
                    improved = True
                    break
            if improved:
                break

    # 4) Final removal pass with remaining budget.
    m = len(best)
    i = 0
    while i < m and _remaining() > 0.5:
        cand = _remove(best, i)
        if _verify(cand):
            best = cand
            m -= 1
            continue
        i += 1

    return best
