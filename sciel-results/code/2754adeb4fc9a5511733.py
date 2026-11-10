# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_TIME_MARGIN = 60.0


def _time_left():
    return _DEADLINE - _TIME_MARGIN - time.time()


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total = q * q  # 2401

    # F_49 = F_7[alpha], alpha^2 = 3 (3 is a non-residue mod 7).
    # element is integer index e = a + b*7 (a,b in 0..6)
    K = 3
    # precompute add/mul tables
    add = np.zeros((q, q), dtype=np.int32)
    mul = np.zeros((q, q), dtype=np.int32)
    idx = 0
    for a in range(p):
        for b in range(p):
            for a2 in range(p):
                for b2 in range(p):
                    j = a2 + b2 * p
                    # add
                    sa = (a + a2) % p
                    sb = (b + b2) % p
                    add[idx, j] = sa + sb * p
                    # mul: (a+b a)(a2+b2 a) = (a a2 + b b2 K) + (a b2 + b a2) a
                    ca = (a * a2 + b * b2 * K) % p
                    cb = (a * b2 + b * a2) % p
                    mul[idx, j] = ca + cb * p
            idx += 1

    zero = 0
    one = 1
    # nonzero elements
    nz = [e for e in range(q) if e != zero]

    # line directions
    dirs = []
    for m in range(q):
        dirs.append((1, m))
    dirs.append((0, 1))
    ndirs = len(dirs)  # 50

    # point id = x + y*q
    def pid(x, y):
        return x + y * q

    # For each (point, dir) requirement, the set of OTHER points on the line.
    # other_mask[r] is a bitset over 2401 points (the punctured line).
    other = np.zeros((total, ndirs), dtype=object)
    for x in range(q):
        for y in range(q):
            pt = pid(x, y)
            for di, (vx, vy) in enumerate(dirs):
                mask = 0
                for t in nz:
                    ox = int(add[x, vx]) if vx != 0 else x
                    # careful: x + t*vx
                    ox = int(add[x, int(mul[t, vx])])
                    oy = int(add[y, int(mul[t, vy])])
                    mask |= 1 << pid(ox, oy)
                other[pt, di] = mask

    # greedy set cover over requirements r=(pt,di)
    req_count = total * ndirs
    # start: all requirements uncovered
    covered = np.zeros(req_count, dtype=bool)
    N_set = set()
    # remaining requirement count per point (number of directions not yet satisfied)
    rem_per_pt = np.full(total, ndirs, dtype=np.int32)

    # Precompute for each point the list of requirement indices
    def ridx(pt, di):
        return pt * ndirs + di

    # We maintain, for each candidate point, how many currently-uncovered
    # requirements it would cover if added. Recompute incrementally.
    # gain[point] = number of uncovered requirements r for which point in other[r]
    gain = np.zeros(total, dtype=np.int32)
    # build gain by iterating requirements and their masks
    for r in range(req_count):
        pt = r // ndirs
        di = r % ndirs
        mask = other[pt, di]
        # add 1 to gain of each point in mask
        m = mask
        while m:
            low = m & -m
            bit = (low.bit_length() - 1)
            gain[bit] += 1
            m ^= low

    # iterative greedy
    import heapq
    # max-heap via negative gain
    heap = [(-int(gain[i]), i) for i in range(total) if gain[i] > 0]
    heapq.heapify(heap)
    # track current gain to skip stale entries
    cur_gain = gain.copy()

    # map requirement -> list of points that cover it (for efficient uncover)
    # We'll instead, when we add a point, recompute uncovered status.
    # For efficiency, keep for each point the set of requirements it covers.
    covers = [set() for _ in range(total)]
    for r in range(req_count):
        pt = r // ndirs
        di = r % ndirs
        mask = other[pt, di]
        m = mask
        while m:
            low = m & -m
            bit = (low.bit_length() - 1)
            covers[bit].add(r)
            m ^= low

    while heap and _time_left() > 0:
        ng, pt = heapq.heappop(heap)
        if -ng != cur_gain[pt] or pt in N_set:
            continue
        if cur_gain[pt] <= 0:
            break
        N_set.add(pt)
        # uncover all requirements covered by pt
        for r in covers[pt]:
            if not covered[r]:
                covered[r] = True
                rpt = r // ndirs
                rem_per_pt[rpt] -= 1
                # this requirement is now satisfied; remove pt's contribution to
                # other points' gains? No: gain counts uncovered reqs a point covers.
                # Since req r is now covered, for every point q in other[r], its gain
                # decreases by 1 (one fewer uncovered req it covers).
                mask = other[rpt, r % ndirs]
                m = mask
                while m:
                    low = m & -m
                    bit = (low.bit_length() - 1)
                    cur_gain[bit] -= 1
                    if cur_gain[bit] > 0 and bit not in N_set:
                        heapq.heappush(heap, (-int(cur_gain[bit]), bit))
                    m ^= low
        cur_gain[pt] = 0

    # Build N as numpy array of shape (k,2,2)
    pts = list(N_set)
    if not pts:
        # fallback: everything
        pts = list(range(total))
    arr = np.zeros((len(pts), 2, 2), dtype=np.int64)
    for i, pt in enumerate(pts):
        x = pt % q
        y = pt // q
        arr[i, 0, 0] = x % p
        arr[i, 0, 1] = (x // p) % p
        arr[i, 1, 0] = y % p
        arr[i, 1, 1] = (y // p) % p
    return arr


# EVOLVE-BLOCK-END
