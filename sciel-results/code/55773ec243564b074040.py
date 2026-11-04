import time
import itertools
import numpy as np

# Ternary tetracode: 9 words of length 4, a perfect covering code of radius 1.
_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _ball_indices(cw, n, q, radius):
    """Integer (base-q) indices of all words within Hamming distance <= radius of cw."""
    cw = np.asarray(cw, dtype=np.int64)
    if radius == 0:
        return np.array([int(np.dot(cw, q ** np.arange(n - 1, -1, -1)))], dtype=np.int64)
    if radius == 1:
        base = int(np.dot(cw, q ** np.arange(n - 1, -1, -1)))
        inc = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        out = [base]
        for i in range(n):
            v = cw[i]
            for d in range(q):
                if d != v:
                    out.append(base + (d - v) * inc[i])
        return np.array(out, dtype=np.int64)
    # General radius (small n/radius): enumerate subsets of positions.
    idxs = set()
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for changes in itertools.product(range(q), repeat=r):
                w = list(cw)
                for pos, ch in zip(positions, changes):
                    w[pos] = ch
                idxs.add(int(np.dot(np.asarray(w, dtype=np.int64), q ** np.arange(n - 1, -1, -1))))
    return np.array(sorted(idxs), dtype=np.int64)


def _tetracode_solution(q, n, radius):
    """Direct sum of the ternary tetracode with every word on the remaining coords."""
    if n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    return [h + list(tail) for h in _TETRA
            for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_cover(q, n, radius, deadline):
    """Greedy set-cover over the whole space, then remove redundant codewords."""
    total = q ** n
    full = np.fromiter(itertools.product(range(q), repeat=n), dtype=np.int8, count=total)
    full_u = full.view(np.dtype((np.void, full.dtype.itemsize * n)))
    full_idx = {w: i for i, w in enumerate(full_u)}

    # Candidate pool: tetracode direct-sum (covers the whole space) plus random samples.
    init = _tetracode_solution(q, n, radius)
    cand_words = list(init)
    seen = {tuple(w) for w in cand_words}
    rng = np.random.default_rng(12345)
    n_rand = min(2000, max(0, total - len(cand_words)))
    for idx in rng.integers(0, total, size=n_rand):
        w = full[idx].tolist()
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            cand_words.append(w)

    # Precompute each candidate's ball as an index array.
    balls = [_ball_indices(cw, n, q, radius) for cw in cand_words]

    covered = np.zeros(total, dtype=bool)
    covered_count = 0
    chosen = []

    while covered_count < total and time.time() < deadline:
        # gain[c] = number of currently-uncovered words in candidate c's ball
        gains = np.array([np.count_nonzero(balls[c] & ~covered) for c in range(len(cand_words))])
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best)
        covered[balls[best]] = True
        covered_count = int(np.count_nonzero(covered))

    # If greedy could not finish (time), fall back to the guaranteed tetracode solution.
    if covered_count < total:
        return _tetracode_solution(q, n, radius)

    selected_words = [cand_words[i] for i in chosen]
    sel_idx = np.array([full_idx[tuple(w)] for w in selected_words], dtype=np.int64)

    # Redundancy removal: drop a codeword if none of its ball words are unique to it.
    keep = np.ones(len(sel_idx), dtype=bool)
    if len(sel_idx) > 0:
        for a in range(len(sel_idx)):
            if not keep[a]:
                continue
            if time.time() > deadline:
                break
            na = sel_idx[a]
            ok = True
            for b in range(len(sel_idx)):
                if b == a or not keep[b]:
                    continue
                if np.any(sel_idx[b] == na):
                    ok = False
                    break
            if ok:
                keep[a] = False

    return [selected_words[i] for i in range(len(selected_words)) if keep[i]]


def construct(q, n, radius):
    deadline = time.time() + 120.0  # safety margin before the 150 s wall-clock limit

    try:
        result = _greedy_cover(q, n, radius, deadline)
        if len(result) > 0:
            return result
    except Exception:
        pass

    return _tetracode_solution(q, n, radius)
