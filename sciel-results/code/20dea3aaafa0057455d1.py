import time
import random
import numpy as np

# ---------- precomputation specialized to q=3, n=6, radius=1 ----------
Q = 3
N = 6
TOTAL = Q ** N                       # 729
NWORDS = 1 + N * (Q - 1)             # 13 words per codeword ball

# linear index of every word
_w = np.arange(TOTAL, dtype=np.int64)
POW = np.array([Q ** (N - 1 - d) for d in range(N)], dtype=np.int64)

# base coordinate of every word
base = np.zeros((TOTAL, N), dtype=np.int64)
for d in range(N):
    base[:, d] = (_w // POW[d]) % Q

# For each candidate codeword (as linear index) precompute its 13 covered word-indices
def _ball(widx):
    """Return the 13 word-indices at Hamming distance <=1 of word widx."""
    coords = [int((widx // POW[d]) % Q) for d in range(N)]
    out = [widx]
    for d in range(N):
        for v in range(Q):
            if v == coords[d]:
                continue
            out.append(widx + (v - coords[d]) * POW[d])
    return out

# tetracode direct sum with 2-symbol tail -> 81 candidate codewords
_tetracode = [[a, b, (a + b) % Q, (a + 2 * b) % Q] for a in range(Q) for b in range(Q)]
cand = []
for h in _tetracode:
    for x in range(Q):
        for y in range(Q):
            cand.append(h + [x, y])
M = len(cand)
cand_idx = np.array([int(np.dot(c, POW)) for c in cand], dtype=np.int64)
cand_set = set(int(x) for x in cand_idx)

# covers[ci] = boolean length-TOTAL mask of words covered by candidate ci
covers_mask = np.zeros((M, TOTAL), dtype=np.int8)
for ci in range(M):
    for wi in _ball(int(cand_idx[ci])):
        covers_mask[ci, wi] = 1

# For fast swap search: word -> list of candidates covering it
word_to_cand = [set() for _ in range(TOTAL)]
for ci in range(M):
    for wi in range(TOTAL):
        if covers_mask[ci, wi]:
            word_to_cand[wi].add(ci)


def _cov_of(codes):
    """Coverage count vector for a list of linear-index codewords."""
    cov = np.zeros(TOTAL, dtype=np.int32)
    for c in codes:
        for wi in _ball(c):
            cov[wi] += 1
    return cov


def _reduce(codes, deadline):
    """Local swap-reduction. codes = list of linear indices (valid cover)."""
    codes = list(codes)
    cov = _cov_of(codes)
    while True:
        if time.time() > deadline:
            break
        m = len(codes)
        moved = False
        order = list(range(m))
        random.shuffle(order)
        for ri in order:
            if time.time() > deadline:
                break
            # unique words lost if we remove codes[ri]
            lost = []
            for d in range(N):
                c = codes[ri]
                coords = [int((c // POW[d2]) % Q) for d2 in range(N)]
                # only iterate ball of ri
            for wi in _ball(codes[ri]):
                if cov[wi] == 1:
                    lost.append(wi)
            if not lost:
                # pure removal
                codes.pop(ri)
                cov = _cov_of(codes)
                moved = True
                break
            else:
                # need a replacement covering all lost
                lost_set = set(lost)
                codes_set = set(codes)
                best = -1
                best_score = -1
                # candidates that cover every lost word
                cand_list = set()
                for wi in lost:
                    cand_list |= word_to_cand[wi]
                for ci in cand_list:
                    if ci in codes_set:
                        continue
                    # score: how many currently-1 words it also covers (tightness)
                    score = 0
                    for wi in _ball(ci):
                        if cov[wi] == 1 and wi not in lost_set:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best = ci
                if best >= 0:
                    codes[ri] = best
                    cov = _cov_of(codes)
                    moved = True
                    break
        if not moved:
            break
    return codes


def _greedy(codes, deadline):
    """Greedy set cover from a candidate pool, then reduce."""
    cov = np.zeros(TOTAL, dtype=np.int8)
    chosen = []
    pool = list(cand_idx)
    while True:
        if time.time() > deadline:
            break
        if cov.all():
            break
        best = None
        best_gain = 0
        for c in pool:
            gain = 0
            for wi in _ball(c):
                if cov[wi] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best = c
        if best is None or best_gain == 0:
            break
        chosen.append(int(best))
        pool = [c for c in pool if c != best]
        for wi in _ball(best):
            cov[wi] = 1
    if not cov.all():
        # fall back to full candidate set
        return _reduce([int(c) for c in cand_idx], deadline)
    return _reduce(chosen, deadline)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    # Fallback for any non-specialized parameters: return a valid (large) full-space code.
    if (q, n, radius) != (Q, N, 1):
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    random.seed(7)
    best = [int(c) for c in cand_idx]
    best_count = M

    # seed with the tetracode-sum reduction first
    codes, _ = _reduce([int(c) for c in cand_idx], deadline - 2.0)
    if len(codes) < best_count:
        best, best_count = codes, len(codes)

    # alternating: greedy-from-scratch and randomized full-set reductions
    it = 0
    while time.time() < deadline - 2.0:
        if it % 3 == 0:
            codes = _greedy(cand_idx, deadline - 2.0)
        else:
            pool = [int(c) for c in cand_idx]
            random.shuffle(pool)
            codes = _reduce(pool, deadline - 2.0)
        if len(codes) < best_count:
            best, best_count = codes, len(codes)
            if best_count <= 71:  # can't beat the lower bound
                break
        it += 1

    # final polish on the best found
    best = _reduce(best, deadline - 0.5)
    return [[(c // POW[d]) % Q for d in range(N)] for c in best]
