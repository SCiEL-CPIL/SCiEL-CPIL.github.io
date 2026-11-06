"""Size-efficient sorting networks on 16 wires.

Fresh design: build a known-valid network (Batcher odd-even merge), then
prune redundant comparators using the zero-one principle over all 2^16
Boolean inputs.  Always returns a verified sorting network; never worse
than the seed.
"""
import time

N = 16
DEADLINE = time.time() + 84.0  # safety margin under the 90 s wall-clock kill


def _batcher_oom(n, offset=0):
    """Comparator list for Batcher's odd-even merge sort on n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    if n % 2 == 0:
        left = _batcher_oom(n // 2, offset)
        right = _batcher_oom(n // 2, offset + n // 2)

        def merge_odd_even(m, base):
            if m <= 1:
                return []
            if m == 2:
                return [(base, base + 1)]
            a = merge_odd_even(m // 2, base)
            b = merge_odd_even(m // 2, base + m // 2)
            cross = [(base + i, base + i + m // 2) for i in range(m // 2)]
            return a + b + cross

        return left + right + merge_odd_even(n, offset)
    else:
        left = _batcher_oom(n // 2, offset)
        right = _batcher_oom(n // 2 + 1, offset + n // 2)
        return left + right


def _simulate(comps, x):
    v = list(x)
    for i, j in comps:
        a, b = v[i], v[j]
        if a > b:
            v[i], v[j] = b, a
    return tuple(v)


def _is_sorted(comps):
    """Zero-one principle: verify all 2^N inputs sort correctly."""
    for mask in range(1 << N):
        x = tuple((mask >> i) & 1 for i in range(N))
        y = _simulate(comps, x)
        k = sum(x)
        expected = (0,) * (N - k) + (1,) * k
        if y != expected:
            return False
    return True


def _prune(comps, deadline):
    """Remove redundant comparators one at a time (zero-one principle)."""
    comps = list(comps)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        i = 0
        while i < len(comps) and time.time() < deadline:
            candidate = comps[:i] + comps[i + 1:]
            if _is_sorted(candidate):
                comps = candidate
                changed = True
            else:
                i += 1
    return comps


def construct():
    net = _batcher_oom(N)
    if not _is_sorted(net):
        # Fallback to the seed odd-even transposition network.
        net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    net = _prune(net, DEADLINE)
    return net
