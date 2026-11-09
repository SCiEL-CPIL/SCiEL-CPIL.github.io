import time
import numpy as np

N = 18
M = 1 << N
DEADLINE = 75.0  # safety margin before the 90s kill
_START = time.time()


def _time_left():
    return DEADLINE - (time.time() - _START)


def _bitonic_net(n):
    """Bitonic sorting network for n wires (n must be a power of two)."""
    net = []
    def merge(lo, width, asc):
        if width == 1:
            return
        merge(lo, width // 2, True)
        merge(lo + width // 2, width // 2, False)
        step = width // 2
        for i in range(0, width, step):
            for j in range(step):
                a, b = lo + i + j, lo + i + j + step
                if not asc:
                    a, b = b, a
                net.append((a, b))
    def sort(lo, width, asc):
        if width == 1:
            return
        sort(lo, width // 2, True)
        sort(lo + width // 2, width // 2, False)
        merge(lo, width, asc)
    sort(0, n, True)
    return net


def _bitvector(net):
    """Return (18 x M) uint8 matrix: bit r,c = whether wire r is 1 on input c."""
    mat = np.zeros((N, M), dtype=np.uint8)
    for r in range(N):
        mask = 1 << r
        mat[r] = (np.arange(M, dtype=np.uint64) & mask) >> r
    for (a, b) in net:
        ma, mb = mat[a], mat[b]
        mat[a] = np.minimum(ma, mb)
        mat[b] = np.maximum(ma, mb)
    return mat


def _is_sorted(net):
    mat = _bitvector(net)
    for r in range(N - 1):
        if np.any(mat[r] < mat[r + 1]):
            return False
    return True


def _count(net):
    return len(net)


def _best_by_removal(net, budget):
    """Try removing each comparator; keep the smallest valid network."""
    best = net
    n = len(net)
    for i in range(n):
        if _time_left() < 3.0:
            break
        cand = net[:i] + net[i + 1:]
        if _is_sorted(cand):
            best = cand
            n = len(cand)
            budget -= 1
    return best


def _local_search(net, budget):
    """Iteratively remove redundant comparators until stable."""
    improved = True
    while improved and budget > 0 and _time_left() > 5.0:
        new_net = _best_by_removal(net, budget)
        if len(new_net) < len(net):
            net = new_net
            improved = True
        else:
            improved = False
    return net


def construct():
    t0 = time.time()

    # Start from a strong, correct network: bitonic sort on 32 wires,
    # restricted to the first 18 wires (comparators only within 0..17).
    base = [c for c in _bitonic_net(32) if c[1] < N]
    if not _is_sorted(base):
        # Fallback to a guaranteed-correct (large) network.
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    best = base
    budget = 200  # max network re-evaluations in the optimization loop

    while _time_left() > 6.0 and budget > 0:
        improved = True
        while improved and _time_left() > 6.0 and budget > 0:
            cand = _best_by_removal(best, budget)
            if len(cand) < len(best):
                best = cand
                improved = True
            else:
                improved = False
        # Once removal is stuck, stop (we have no cheap insertion heuristic
        # that is guaranteed to beat removal-only on this task within budget).
        if not improved:
            break

    return list(best)


# Optional: keep a module-level helper name from the seed lineage if referenced.
def _time_left_check():
    return _time_left()
