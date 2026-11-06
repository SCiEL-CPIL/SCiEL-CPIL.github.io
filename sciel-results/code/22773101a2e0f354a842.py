"""Size-efficient sorting networks on 18 wires.
Fresh design: build a known network, verify over all 2^18 Boolean inputs (0-1 principle),
then greedily reduce and locally search for smaller valid networks under a wall-clock deadline.
"""
import time
import random
import numpy as np

N = 18
DEADLINE = 85.0  # safety margin before 90s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _odd_even_transposition():
    # N phases; phase p uses comparators (i, i+1) for i in range(p%2, N-1, 2)
    return [(i, i + 1) for p in range(N) for i in range(p % 2, N - 1, 2)]


def _verify(net, arr):
    """Return True iff `net` sorts every row of the 2^N x N boolean matrix `arr`.
    Uses the 0-1 principle: sorting all Boolean inputs is necessary and sufficient."""
    a = arr.copy()
    for i, j in net:
        a[:, i], a[:, j] = np.minimum(a[:, i], a[:, j]), np.maximum(a[:, i], a[:, j])
    # sorted iff each row is non-increasing (wire0 >= wire1 >= ... )
    d = np.diff(a, axis=1)
    return not np.any(d > 0)


def _build_matrix():
    # all 2^N boolean inputs
    bits = np.arange(1 << N, dtype=np.uint8)
    m = np.zeros((1 << N, N), dtype=np.uint8)
    for k in range(N):
        m[:, k] = (bits >> (N - 1 - k)) & 1
    return m


def _count(net):
    return len(net)


def _try_remove(net, arr):
    """Try to remove one comparator; return new net if valid, else None."""
    n = len(net)
    for idx in range(n):
        cand = net[:idx] + net[idx + 1:]
        if _verify(cand, arr):
            return cand
    return None


def _try_replace(net, arr, i, j):
    """Try replacing comparator i with (i,j); return new net if valid else None."""
    if (i, j) == (net[0], net[1]) or (i, j) == (net[-1], net[-2]):
        return None
    cand = net[:i] + [(i, j)] + net[i + 1:]
    if _verify(cand, arr):
        return cand
    return None


def _perturb(net):
    """Randomly swap adjacent comparators to escape local minima."""
    net = list(net)
    for _ in range(random.randint(1, 3)):
        if len(net) < 2:
            break
        a = random.randrange(len(net) - 1)
        net[a], net[a + 1] = net[a + 1], net[a]
    return net


def construct():
    global _START
    _START = time.time()

    arr = _build_matrix()

    # seed network
    net = _odd_even_transposition()
    assert _verify(net, arr), "seed network must be valid"
    best = net
    best_len = len(best)

    rng = random.Random(12345)

    def deadline():
        return (_elapsed() > DEADLINE - 3.0)

    # Phase 1: greedy removal loop (fast, linear scan, no immediate restart)
    while not deadline():
        newnet = _try_remove(best, arr)
        if newnet is None:
            break
        best = newnet
        best_len = len(best)

    # Phase 2: local replacement search + occasional restarts
    while not deadline():
        # try to remove again
        newnet = _try_remove(best, arr)
        if newnet is not None:
            best = newnet
            best_len = len(best)
            continue

        # try a few random replacements
        improved = False
        if len(best) >= 2:
            for _ in range(200):
                if deadline():
                    break
                i = rng.randrange(len(best) - 1)
                a, b = best[i]
                # propose a different comparator
                c = rng.randrange(N)
                d = rng.randrange(N)
                if c == d:
                    continue
                if c > d:
                    c, d = d, c
                if (c, d) == (a, b):
                    continue
                cand = best[:i] + [(c, d)] + best[i + 1:]
                if _verify(cand, arr):
                    best = cand
                    improved = True
                    break
        if not improved:
            # perturbation + restart
            best = _perturb(best)
            if not _verify(best, arr):
                # revert to a known-good smaller seed and re-reduce
                best = _odd_even_transposition()
                # quick re-reduce
                while not deadline():
                    nn = _try_remove(best, arr)
                    if nn is None:
                        break
                    best = nn

    return list(best)


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
