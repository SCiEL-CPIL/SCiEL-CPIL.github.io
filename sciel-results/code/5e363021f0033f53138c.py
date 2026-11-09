"""Snake-in-the-box in Q_9: greedy build + gap-insertion growth."""
import random
import time

N = 9
V = 1 << N
MASK = V - 1

# Precompute neighbor masks (512-bit) for each vertex.
_ADJ = [0] * V
for _v in range(V):
    m = 0
    for _b in range(N):
        m |= 1 << (_v ^ (1 << _b))
    _ADJ[_v] = m

_DEADLINE = time.time() + 110.0   # safety margin under the 120 s limit


def construct():
    rng = random.Random(12345)

    # ---- Phase 1: greedy build ----
    path = [0]
    used = 1
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            w = cur ^ (1 << b)
            if not (used >> w) & 1:
                # must not be adjacent to any non-consecutive vertex
                if (_ADJ[w] & used) == (1 << cur):
                    nxt = w
                    break
        if nxt is None:
            break
        path.append(nxt)
        used |= 1 << nxt

    def to_used(p):
        u = 0
        for x in p:
            u |= 1 << x
        return u

    best = path[:]
    best_len = len(path)

    def try_insert(path, used):
        """Attempt to grow by inserting into a 2-bit gap. Return (path, used, ok)."""
        n = len(path)
        # candidate gaps: consecutive pair differing in exactly 2 bits
        gaps = []
        for i in range(n - 1):
            d = path[i] ^ path[i + 1]
            if d & (d - 1):  # not a single bit -> exactly 2 bits (since consecutive)
                gaps.append(i)
        if not gaps:
            return path, used, False
        rng.shuffle(gaps)
        for i in gaps:
            a, b = path[i], path[i + 1]
            d = a ^ b
            if d & (d - 1) == 0:
                continue
            # flip a random non-empty subset of the 2 gap bits
            cands = []
            for s in range(1, 4):
                if s & 1:
                    cands.append(a ^ (d & 1))
                if s & 2:
                    cands.append(a ^ ((d >> 1) & 1) << 1)
            rng.shuffle(cands)
            for w in cands:
                if (used >> w) & 1:
                    continue
                if (_ADJ[w] & used) == ((1 << a) | (1 << b)):
                    path.insert(i + 1, w)
                    used |= 1 << w
                    return path, used, True
        return path, used, False

    # ---- Phase 2: iterative gap insertion ----
    while time.time() < _DEADLINE:
        p, u, ok = try_insert(path, used)
        if ok:
            if len(p) > best_len:
                best = p[:]
                best_len = len(p)
            path, used = p, u
        else:
            # stuck: try a few randomized restarts of the build to escape
            if time.time() < _DEADLINE:
                for _ in range(50):
                    rng2 = random.Random(rng.randrange(1 << 30))
                    path2 = [0]
                    used2 = 1
                    while True:
                        cur = path2[-1]
                        nxt = None
                        for b in range(N):
                            w = cur ^ (1 << b)
                            if not (used2 >> w) & 1:
                                if (_ADJ[w] & used2) == (1 << cur):
                                    nxt = w
                                    break
                        if nxt is None:
                            break
                        path2.append(nxt)
                        used2 |= 1 << nxt
                    if len(path2) > len(path):
                        path, used = path2, used2
                        if len(path) > best_len:
                            best = path[:]
                            best_len = len(path)

    return best
