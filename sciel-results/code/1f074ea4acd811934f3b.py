# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 118.0  # safety margin before 150s kill
_RNG = np.random.default_rng(12345)


def _time_left():
    return _DEADLINE - time.time()


def _init_cover(codewords, q, n, radius):
    """coverage[i] = number of codewords within radius of word i."""
    if not codewords:
        return np.zeros(q ** n, dtype=np.int32)
    cd = np.array(codewords, dtype=np.int64)
    cov = np.zeros(q ** n, dtype=np.int32)
    for cw in cd:
        base = np.zeros(n, dtype=np.int64)
        for pos in range(n):
            for v in range(q):
                base[pos] = v
                cov[np.ravel_multi_index(base, (q,) * n)] += 1
    return cov


def _greedy_construct(q, n, radius, budget_s, rng):
    """Randomized greedy: repeatedly add the word covering the most uncovered words."""
    total = q ** n
    cov = np.zeros(total, dtype=np.int32)
    cd = []
    # random tie-break pool size
    tie_pool = int(rng.integers(2, 9))
    while _time_left() > budget_s:
        uncovered = np.flatnonzero(cov == 0)
        if uncovered.size == 0:
            break
        # sample a subset to keep cost bounded
        if uncovered.size > 400:
            sample = rng.choice(uncovered, size=400, replace=False)
        else:
            sample = uncovered
        # score each candidate by how many currently-uncovered words it covers
        best = -1
        best_idx = -1
        best_cand = None
        # evaluate candidates = sample words; for each, count coverage of uncovered
        for cand in sample:
            base = np.unravel_index(cand, (q,) * n)
            # count how many uncovered words are within radius
            cnt = 0
            # radius 1: the word itself + single-position changes
            cnt += 1 if cov[cand] == 0 else 0
            for pos in range(n):
                old = base[pos]
                for v in range(q):
                    if v == old:
                        continue
                    base[pos] = v
                    w = np.ravel_multi_index(base, (q,) * n)
                    if cov[w] == 0:
                        cnt += 1
                base[pos] = old
            if cnt > best:
                best = cnt
                best_idx = cand
                best_cand = base.copy()
        if best_cand is None:
            break
        # add best_cand
        cd.append(list(best_cand))
        # update coverage for this codeword
        base = best_cand
        cov[cand] += 1
        for pos in range(n):
            old = base[pos]
            for v in range(q):
                if v == old:
                    continue
                base[pos] = v
                cov[np.ravel_multi_index(base, (q,) * n)] += 1
            base[pos] = old
    return cd


def _local_search(cd, q, n, radius, cov, budget_s, rng):
    """Swap + deletion local search with delta evaluation."""
    if not cd:
        return cd, cov
    cd = list(cd)
    cov = cov.copy()
    improved = True
    while improved and _time_left() > budget_s:
        improved = False
        # --- deletion pass ---
        for i in range(len(cd) - 1, -1, -1):
            cw = cd[i]
            base = list(cw)
            # count words that ONLY this codeword covers (coverage would drop to 0)
            would_uncover = 0
            cand = np.ravel_multi_index(base, (q,) * n)
            if cov[cand] <= 1:
                would_uncover += 1
            for pos in range(n):
                old = base[pos]
                for v in range(q):
                    if v == old:
                        continue
                    base[pos] = v
                    w = np.ravel_multi_index(base, (q,) * n)
                    if cov[w] <= 1:
                        would_uncover += 1
                base[pos] = old
            if would_uncover == 0:
                # safe to delete
                _remove_cw(cov, cw, q, n)
                del cd[i]
                improved = True
        # --- swap pass (replace one codeword with a better one) ---
        for i in range(len(cd)):
            if _time_left() < budget_s:
                break
            cw = cd[i]
            # candidate new words: random words, and words near uncovered regions
            for _ in range(30):
                neww = [int(rng.integers(0, q)) for _ in range(n)]
                # compute delta: gain - loss
                gain = 0
                loss = 0
                base_new = list(neww)
                # gain: words covered by neww that are currently uncovered
                for w in _radius_words(base_new, q, n):
                    if cov[w] == 0:
                        gain += 1
                # loss: words covered by old cw that would become uncovered
                base_old = list(cw)
                for w in _radius_words(base_old, q, n):
                    if cov[w] == 1:
                        loss += 1
                if gain - loss > 0:
                    _remove_cw(cov, cw, q, n)
                    _add_cw(cov, neww, q, n)
                    cd[i] = neww
                    improved = True
                    break
    return cd, cov


def _radius_words(word, q, n):
    """Yield all words within Hamming distance 1 of word (inclusive)."""
    base = list(word)
    yield np.ravel_multi_index(base, (q,) * n)
    for pos in range(n):
        old = base[pos]
        for v in range(q):
            if v == old:
                continue
            base[pos] = v
            yield np.ravel_multi_index(base, (q,) * n)
        base[pos] = old


def _add_cw(cov, cw, q, n):
    base = list(cw)
    cov[np.ravel_multi_index(base, (q,) * n)] += 1
    for pos in range(n):
        old = base[pos]
        for v in range(q):
            if v == old:
                continue
            base[pos] = v
            cov[np.ravel_multi_index(base, (q,) * n)] += 1
        base[pos] = old


def _remove_cw(cov, cw, q, n):
    base = list(cw)
    cov[np.ravel_multi_index(base, (q,) * n)] -= 1
    for pos in range(n):
        old = base[pos]
        for v in range(q):
            if v == old:
                continue
            base[pos] = v
            cov[np.ravel_multi_index(base, (q,) * n)] -= 1
        base[pos] = old


def _is_valid(cd, q, n, radius):
    total = q ** n
    cov = np.zeros(total, dtype=np.int32)
    for cw in cd:
        _add_cw(cov, cw, q, n)
    return bool(np.all(cov > 0))


def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 118.0
    rng = _RNG
    total = q ** n

    # --- initial solution: direct sum (the seed) ---
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(q) for b in range(q)]
        init = [h + [0] * (n - 4) for h in tetracode]
    else:
        init = [[0] * n]

    best = None
    best_len = len(init)

    # try to get a valid baseline via greedy if init isn't (it should be)
    # --- ILS loop ---
    it = 0
    while _time_left() > 5.0:
        it += 1
        # --- construct a candidate ---
        if it == 1:
            cand = list(init)
        else:
            # start from a perturbation of best, or fresh greedy
            if best is not None and rng.random() < 0.6:
                cand = [list(w) for w in best]
                # perturb: drop a few, add a few random
                k = max(1, len(cand) // 10)
                for _ in range(k):
                    if cand:
                        del cand[int(rng.integers(0, len(cand)))]
                    cand.append([int(rng.integers(0, q)) for _ in range(n)])
            else:
                cand = _greedy_construct(q, n, radius, budget_s=4.0, rng=rng)

        # local search
        cov = _init_cover(cand, q, n, radius)
        cand, cov = _local_search(cand, q, n, radius, budget_s=4.0, rng=rng)

        # validate
        if _is_valid(cand, q, n, radius):
            if best is None or len(cand) < best_len:
                best = [list(w) for w in cand]
                best_len = len(cand)

    if best is None:
        # fallback: ensure validity
        if _is_valid(init, q, n, radius):
            return [list(w) for w in init]
        # last resort: full space
        return [[int(d) for d in np.unravel_index(i, (q,) * n)] for i in range(total)]
    return best
# EVOLVE-BLOCK-END
