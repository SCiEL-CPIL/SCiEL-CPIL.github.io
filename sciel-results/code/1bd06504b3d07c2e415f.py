"""Covering code K_3(8,1) via greedy construction + removal + swap local search."""
import time
import itertools
import random

_DEADLINE = 110.0  # wall-clock safety margin before the 120s limit


def _to_int(word, n):
    """Encode a length-n word over 0..q-1 as an integer (big-endian)."""
    v = 0
    for x in word:
        v = v * 3 + x
    return v


def _to_word(v, n):
    """Decode an integer back to a length-n word over 0..2."""
    out = [0] * n
    for i in range(n - 1, -1, -1):
        out[i] = v % 3
        v //= 3
    return out


def _neighbors(v, n):
    """All words within Hamming distance 1 of v (including v), as ints."""
    out = [v]
    for i in range(n):
        base = v // (3 ** (n - 1 - i))
        cur = v % (3 ** (n - 1 - i))
        pos = 3 ** (n - 1 - i)
        for a in range(3):
            if a != base:
                out.append((v - base * pos) + a * pos + cur)
    return out


def _cover_set(codes, n, total):
    """Union of balls radius 1 around each code, as a Python set of ints."""
    cov = set()
    for v in codes:
        for u in _neighbors(v, n):
            cov.add(u)
    return cov


def _is_valid(codes, n, total):
    """True iff every word of {0,1,2}^n is within distance 1 of some code."""
    return len(_cover_set(codes, n, total)) == total


def _greedy_build(n, total, deadline):
    """Greedy set cover: repeatedly add the word covering the most uncovered words."""
    all_words = list(range(total))
    uncovered = set(all_words)
    codes = []
    start = time.time()
    while uncovered and (time.time() - start) < deadline:
        best_v = None
        best_gain = -1
        # Sample candidate centers: pick from uncovered words (each is worth covering)
        # Use a deterministic-ish scan over a shuffled subset to keep it fast.
        cand = list(uncovered)
        random.shuffle(cand)
        for v in cand:
            gain = 0
            for u in _neighbors(v, n):
                if u in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_v = v
            if best_gain == 72:  # can't do better than a full ball
                break
        if best_v is None:
            break
        codes.append(best_v)
        for u in _neighbors(best_v, n):
            uncovered.discard(u)
    return codes


def _remove_pass(codes, n, total, deadline):
    """Iteratively remove any code whose ball is redundant."""
    codes = list(codes)
    start = time.time()
    changed = True
    while changed and (time.time() - start) < deadline:
        changed = False
        for i in range(len(codes)):
            v = codes[i]
            # Is v's ball covered by the rest?
            rest = codes[:i] + codes[i + 1:]
            if not rest:
                continue
            rest_cov = set()
            for w in rest:
                for u in _neighbors(w, n):
                    rest_cov.add(u)
            redundant = True
            for u in _neighbors(v, n):
                if u not in rest_cov:
                    redundant = False
                    break
            if redundant:
                codes = codes[:i] + codes[i + 1:]
                changed = True
                break
    return codes


def _swap_pass(codes, n, total, deadline, rng):
    """Try replacing a code with a random candidate; keep if it enables removals."""
    codes = list(codes)
    start = time.time()
    best = len(codes)
    while (time.time() - start) < deadline and len(codes) >= best:
        i = rng.randrange(len(codes))
        # Candidate: a random word from the space (or from a neighbor of an existing code)
        if rng.random() < 0.5:
            new_v = rng.randrange(total)
        else:
            base = codes[rng.randrange(len(codes))]
            new_v = rng.choice(_neighbors(base, n))
        if new_v in codes:
            continue
        cand = codes[:i] + [new_v] + codes[i + 1:]
        # Must remain a valid cover
        if len(_cover_set(cand, n, total)) != total:
            continue
        # Try to remove redundancies from the candidate
        cand = _remove_pass(cand, n, total, min(deadline, time.time() - start))
        if len(cand) < best:
            codes = cand
            best = len(codes)
    return codes


def construct(q, n, radius):
    start = time.time()
    deadline = start + _DEADLINE
    total = q ** n
    rng = random.Random(12345)

    # 1. Greedy construction
    codes = _greedy_build(n, total, deadline - (time.time() - start) - 20)
    codes = _remove_pass(codes, n, total, deadline - (time.time() - start) - 10)

    # 2. Swap local search
    remaining = deadline - time.time()
    if remaining > 5:
        codes = _swap_pass(codes, n, total, remaining - 5, rng)

    # 3. Final removal pass
    codes = _remove_pass(codes, n, total, max(0, deadline - time.time() - 2))

    # Guarantee validity; if not valid, fall back to the direct-sum construction.
    if not _is_valid(codes, n, total):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codes = [h + list(tail) for h in tetracode
                 for tail in itertools.product(range(q), repeat=n - 4)]

    # Convert ints back to word lists
    return [_to_word(v, n) for v in codes]
