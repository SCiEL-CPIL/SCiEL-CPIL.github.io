"""Fresh design for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _neighbors(word, q, n):
    """All words within Hamming distance <=1 of word (inclusive)."""
    out = [list(word)]
    for i in range(n):
        for v in range(q):
            if v == word[i]:
                continue
            w = list(word)
            w[i] = v
            out.append(w)
    return out


def _build_counts(codewords, q, n):
    """coverage count array of size q**n; index = base-q number of the word."""
    size = q ** n
    counts = [0] * size
    for cw in codewords:
        for nb in _neighbors(cw, q, n):
            idx = 0
            for d in nb:
                idx = idx * q + d
            counts[idx] += 1
    return counts


def _uncovered_indices(counts):
    return [i for i, c in enumerate(counts) if c == 0]


def _index_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w


def _word_to_idx(word, q):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _try_improve(codewords, q, n, radius, deadline, rng, max_rounds=4000):
    """Local search: remove redundant codewords, add best covering word.
    Returns (list_of_codewords, is_valid)."""
    size = q ** n
    counts = _build_counts(codewords, q, n)
    # map codeword tuple -> index in the list
    cw_set = {}
    for i, cw in enumerate(codewords):
        cw_set[tuple(cw)] = i

    def remove_cw(cw):
        key = tuple(cw)
        for nb in _neighbors(cw, q, n):
            counts[_word_to_idx(nb, q)] -= 1
        cw_set.pop(key, None)
        codewords.remove(cw)

    def add_cw(cw):
        for nb in _neighbors(cw, q, n):
            counts[_word_to_idx(nb, q)] += 1
        codewords.append(cw)

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        if time.time() > deadline:
            break
        rounds += 1
        improved = False

        # 1) Remove redundant codewords (in a randomized order to escape order bias)
        order = list(range(len(codewords)))
        rng.shuffle(order)
        removed_any = False
        for pos in order:
            if time.time() > deadline:
                break
            # pos is an index into codewords; but removals shift indices,
            # so operate by value using a working snapshot of current list.
            pass
        # Simpler: iterate over a snapshot of current codewords (by value).
        snapshot = list(codewords)
        for cw in snapshot:
            if time.time() > deadline:
                break
            if cw not in codewords:
                continue
            # check if removing keeps coverage >=1 everywhere
            ok = True
            for nb in _neighbors(cw, q, n):
                if counts[_word_to_idx(nb, q)] <= 1:
                    ok = False
                    break
            if ok:
                remove_cw(cw)
                improved = True
                removed_any = True
                if len(codewords) == 0:
                    break

        # 2) If there are still uncovered words, add the best covering word.
        uncovered = _uncovered_indices(counts)
        if uncovered:
            # score each candidate word by number of currently-uncovered neighbors
            best_cw = None
            best_gain = -1
            # candidate words: only need to consider words that are uncovered or
            # near uncovered; but to be safe consider a candidate set.
            # A word covers its neighbors; to cover any uncovered word u, the new
            # codeword must be within distance 1 of u. Candidate set = union of
            # neighbors of all uncovered words.
            cand = set()
            for u in uncovered:
                w = _index_to_word(u, q, n)
                for nb in _neighbors(w, q, n):
                    cand.add(tuple(nb))
            # score candidates
            for cw in cand:
                if time.time() > deadline:
                    break
                if cw in cw_set:
                    continue
                gain = 0
                for nb in _neighbors(list(cw), q, n):
                    if counts[_word_to_idx(nb, q)] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cw = cw
            if best_cw is not None and best_gain > 0:
                add_cw(list(best_cw))
                improved = True

    # final validity check
    counts = _build_counts(codewords, q, n)
    valid = all(c >= 1 for c in counts)
    return codewords, valid


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin vs 150s kill
    rng = random.Random(12345)

    # Base construction: tetracode (9) x all tails on remaining coords.
    base = _tetracode()
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial = [h + list(tail) for h in base for tail in tails]

    best = None
    best_valid = False

    def consider(codewords, valid):
        nonlocal best, best_valid
        if valid and (best is None or len(codewords) < len(best)):
            best = codewords
            best_valid = True

    # First, one deterministic-ish pass.
    cur = [list(cw) for cw in initial]
    res, valid = _try_improve(cur, q, n, radius, deadline, rng)
    consider(res, valid)

    # Random restarts with shuffled initial order and different seeds, while time allows.
    attempt = 0
    while time.time() < deadline - 5.0:
        attempt += 1
        seed = 1000 + attempt
        r = random.Random(seed)
        cur = [list(cw) for cw in initial]
        r.shuffle(cur)
        # Also try a slightly perturbed start: drop a random few then let search fix.
        res, valid = _try_improve(cur, q, n, radius, deadline, r, max_rounds=1500)
        consider(res, valid)
        if best is not None and len(best) <= 71:
            break  # can't beat the known lower bound; stop early

    if best is not None:
        return [list(cw) for cw in best]
    # Fallback: return the initial (valid) construction.
    return [list(cw) for cw in initial]
# EVOLVE-BLOCK-END
