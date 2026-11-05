"""Greedy set-cover construction for the q-ary covering code K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _initial_code(q, n):
    """Direct sum of the ternary tetracode with the full cube on the rest.

    This is a valid covering code of size 9 * q ** (n - 4) for q >= 3.
    """
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _encode(words, q, n):
    """Encode a list of length-n words as a base-q integer array."""
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    return np.array([int(np.dot(w, powers)) for w in words], dtype=np.int64)


def _decode(idx, q, n):
    """Decode a base-q integer back into a length-n word."""
    w = [0] * n
    for i in range(n):
        w[i] = idx % q
        idx //= q
    return w


def _all_neighbors(codes, q, n, radius):
    """All words within Hamming distance <= radius of any codeword, as a set of ints."""
    codes = _encode(codes, q, n)
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    out = set()
    for c in codes:
        out.add(int(c))
        for pos in range(n):
            base = int(c - (c % q) * powers[pos])
            for v in range(q):
                if v == c % q:
                    continue
                out.add(base + v * powers[pos])
    return out


def _coverage(codes, q, n, radius, deadline, target=None):
    """Return (covered_set, uncovered_list) for the given codes."""
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    covered = set()
    for c in codes:
        covered.add(int(c))
        for pos in range(n):
            base = int(c - (c % q) * powers[pos])
            for v in range(q):
                if v == c % q:
                    continue
                covered.add(base + v * powers[pos])
    total = q ** n
    uncovered = [w for w in range(total) if w not in covered]
    if target is not None:
        uncovered = uncovered[:target]
    return covered, uncovered


def _greedy_construct(q, n, radius, deadline, start_codes=None):
    """Greedy set-cover: repeatedly add the codeword covering the most uncovered words."""
    total = q ** n
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)

    codes = list(start_codes) if start_codes is not None else []
    if codes:
        covered, uncovered = _coverage(codes, q, n, radius, deadline)
    else:
        covered = set()
        uncovered = list(range(total))

    while uncovered and time.time() < deadline:
        # Pick a pivot uncovered word; it must be covered by a codeword within radius.
        pivot = uncovered[0]
        # Enumerate all candidate codewords within radius of the pivot.
        candidates = []
        pw = [0] * n
        for i in range(n):
            pw[i] = pivot % q
            pivot //= q
        # radius 1: pivot itself + all single-position changes
        candidates.append(tuple(pw))
        for pos in range(n):
            for v in range(q):
                if v == pw[pos]:
                    continue
                cand = list(pw)
                cand[pos] = v
                candidates.append(tuple(cand))

        # For each candidate, count how many uncovered words it covers.
        best_gain = -1
        best_cand = None
        for cand in candidates:
            cand_int = int(np.dot(cand, powers))
            gain = 1  # covers itself
            for pos in range(n):
                base = cand_int - (cand_int % q) * powers[pos]
                for v in range(q):
                    if v == cand[pos]:
                        continue
                    nb = base + v * powers[pos]
                    if nb in uncovered:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cand = cand

        codes.append(list(best_cand))
        covered, uncovered = _coverage(codes, q, n, radius, deadline)

    return codes


def construct(q, n, radius):
    deadline = time.time() + 110.0  # safety margin before the 120s wall-clock kill

    # Trivial small cases: cover the whole space directly.
    if q ** n <= 4000:
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Start from the direct-sum construction, which is already valid and small.
    codes = _initial_code(q, n)
    best = list(codes)

    # Greedy refinement: keep adding codewords only if it never hurts; we instead
    # try to *replace* by running greedy from scratch and keeping the smaller result.
    try:
        greedy = _greedy_construct(q, n, radius, deadline, start_codes=None)
        if len(greedy) < len(best):
            best = greedy
    except Exception:
        pass

    # If we still have budget, attempt a greedy run seeded from the initial code.
    if time.time() < deadline:
        try:
            seeded = _greedy_construct(q, n, radius, deadline, start_codes=best)
            if len(seeded) < len(best):
                best = seeded
        except Exception:
            pass

    return best
# EVOLVE-BLOCK-END
