import time
import random

# Precompute the index <-> word mapping for q=3, n=7
_Q, _N = 3, 7
_SPACE = _Q ** _N  # 2187
# word -> index
_WORD_TO_IDX = {}
_IDX_TO_WORD = []
for _i in range(_SPACE):
    _w = []
    _x = _i
    for _ in range(_N):
        _w.append(_x % _Q)
        _x //= _Q
    _WORD_TO_IDX[tuple(_w)] = _i
    _IDX_TO_WORD.append(_w)

# Precompute the radius-1 ball (as a bitmask) for every word in the space.
# ball[i] is an int with bit j set iff word j is within distance <=1 of word i.
_BALLS = [0] * _SPACE
for _i in range(_SPACE):
    _w = _IDX_TO_WORD[_i]
    _m = 1 << _i
    for _p in range(_N):
        _old = _w[_p]
        for _v in range(_Q):
            if _v == _old:
                continue
            _w2 = list(_w)
            _w2[_p] = _v
            _m |= 1 << _WORD_TO_IDX[tuple(_w2)]
    _BALLS[_i] = _m


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before the 120s kill
    rng = random.Random(12345)

    if q != _Q or n != _N or radius != 1:
        # Fallback: greedy set cover for the general case (best effort).
        space = q ** n
        if space > 2_000_000:
            # Too large; fall back to a simple structured answer.
            return [[0] * n]
        # Enumerate all words and their radius balls
        words = []
        for i in range(space):
            w = []
            x = i
            for _ in range(n):
                w.append(x % q)
                x //= q
            words.append(w)
        idx = {tuple(w): i for i, w in enumerate(words)}
        balls = []
        for i, w in enumerate(words):
            m = 1 << i
            for p in range(n):
                old = w[p]
                for v in range(q):
                    if v == old:
                        continue
                    w2 = list(w)
                    w2[p] = v
                    m |= 1 << idx[tuple(w2)]
            balls.append(m)
        full = (1 << space) - 1
        covered = 0
        chosen = []
        for _ in range(space):
            if time.time() > deadline:
                break
            if covered == full:
                break
            best_i = -1
            best_gain = 0
            for i in range(space):
                gain = (balls[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i < 0:
                break
            chosen.append(words[best_i])
            covered |= balls[best_i]
        return chosen

    # Specialized fast path for q=3, n=7, radius=1
    full = (1 << _SPACE) - 1

    def greedy_cover(seed_order):
        covered = 0
        chosen = []
        # candidate order: iterate in given order, but pick best marginal each step
        # To be efficient, we scan all candidates each step (2187 * ~186 ~ 400k ops)
        for _ in range(_SPACE):
            if covered == full:
                break
            best_i = -1
            best_gain = 0
            for i in seed_order:
                gain = (_BALLS[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i < 0:
                break
            chosen.append(best_i)
            covered |= _BALLS[best_i]
        return chosen

    def greedy_cover_random():
        order = list(range(_SPACE))
        rng.shuffle(order)
        return greedy_cover(order)

    def prune(indices):
        # Remove redundant codewords: if a codeword's ball is covered by the union
        # of the others' balls, drop it. Iterate until stable.
        cur = list(indices)
        changed = True
        while changed:
            changed = False
            m = len(cur)
            for i in range(m):
                # union of all balls except i
                others = 0
                for j in range(m):
                    if j != i:
                        others |= _BALLS[cur[j]]
                if (_BALLS[cur[i]] & ~others) == 0:
                    # redundant
                    cur.pop(i)
                    m -= 1
                    changed = True
                    break
        return cur

    # Phase 1: deterministic greedy (natural order)
    best = prune(greedy_cover(list(range(_SPACE))))
    best_size = len(best)

    # Phase 2: randomized greedy restarts to try to beat it
    budget_restarts = 200
    for r in range(budget_restarts):
        if time.time() > deadline - 10.0:
            break
        cand = greedy_cover_random()
        cand = prune(cand)
        if len(cand) < best_size:
            best = cand
            best_size = len(cand)
            if best_size <= 186:
                break

    # Phase 3: local pair-compression on the best code (structural compression)
    # Try to replace two codewords with one if the new ball covers what they covered
    # that isn't covered by the rest.
    def pair_compress(code):
        code = list(code)
        improved = True
        while improved:
            improved = False
            m = len(code)
            for i in range(m):
                if time.time() > deadline - 5.0:
                    return code
                for j in range(i + 1, m):
                    # rest = union of balls of all except i and j
                    rest = 0
                    for k in range(m):
                        if k != i and k != j:
                            rest |= _BALLS[code[k]]
                    # what i and j cover that rest does not
                    need = (_BALLS[code[i]] | _BALLS[code[j]]) & ~rest
                    # find a single word whose ball covers 'need'
                    found = -1
                    for c in range(_SPACE):
                        if (_BALLS[c] & need) == need:
                            found = c
                            break
                    if found >= 0:
                        # replace i and j with found
                        newcode = []
                        for k in range(m):
                            if k == i or k == j:
                                continue
                            newcode.append(code[k])
                        newcode.append(found)
                        code = newcode
                        improved = True
                        break
                if improved:
                    break
        return code

    if time.time() < deadline - 5.0:
        compressed = pair_compress(best)
        compressed = prune(compressed)
        if len(compressed) < best_size:
            best = compressed
            best_size = len(compressed)

    return [_IDX_TO_WORD[i] for i in best]
