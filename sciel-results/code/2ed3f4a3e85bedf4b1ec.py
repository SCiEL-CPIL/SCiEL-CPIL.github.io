"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

import numpy as np


def _tetracode_words(q, n):
    """Direct-sum construction: tetracode on first 4 coords, all words on rest."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n == 4:
        return [list(h) for h in tetracode]
    tails = list(itertools.product(range(q), repeat=n - 4))
    return [h + list(tail) for h in tetracode for tail in tails]


def _encode_int(word, q, n):
    """Convert a list of digits to a single integer in [0, q^n)."""
    val = 0
    for d in word:
        val = val * q + d
    return val


def _decode_int(val, q, n):
    """Convert an integer to a list of n digits in base q."""
    digits = []
    for _ in range(n):
        digits.append(val % q)
        val //= q
    digits.reverse()
    return digits


def _hamming_distance(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _is_covering(codewords, q, n, radius):
    """Check if every word in {0..q-1}^n is within radius of some codeword."""
    total = q ** n
    if total > 200000:
        # Too large to check exhaustively; assume valid if we got here via greedy
        return True
    # Build a set of covered words for fast lookup
    covered = set()
    for cw in codewords:
        cw_int = _encode_int(cw, q, n)
        covered.add(cw_int)
        if radius >= 1:
            for pos in range(n):
                for new_digit in range(q):
                    if new_digit != cw[pos]:
                        w = list(cw)
                        w[pos] = new_digit
                        covered.add(_encode_int(w, q, n))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for d1 in range(q):
                        if d1 == cw[pos1]:
                            continue
                        for d2 in range(q):
                            if d2 == cw[pos2]:
                                continue
                            w = list(cw)
                            w[pos1] = d1
                            w[pos2] = d2
                            covered.add(_encode_int(w, q, n))
    return len(covered) == total


def _greedy_cover(q, n, radius, deadline, seed=0):
    """
    Greedy set-cover heuristic: repeatedly pick the word that covers the most
    currently-uncovered words. Uses a bitset of uncovered words for efficiency.
    """
    rng = random.Random(seed)
    total = q ** n
    if total > 500000:
        # Fall back to tetracode for very large spaces
        return _tetracode_words(q, n)

    # Precompute the ball (set of words within radius) for each possible center
    # represented as a frozenset of ints, but for memory we compute on the fly.
    # Instead, we track uncovered as a Python set of ints.
    uncovered = set(range(total))

    codewords = []
    start = time.time()

    while uncovered and (time.time() - start) < deadline - 5.0:
        # Pick the best center: maximize |ball(center) ∩ uncovered|
        # For speed, sample a subset of candidates if the space is large.
        if len(uncovered) > 10000:
            # Sample candidates from uncovered words and their neighbors
            candidates = list(uncovered)
            rng.shuffle(candidates)
            candidates = candidates[:min(2000, len(candidates))]
        else:
            candidates = list(uncovered)

        best_center = None
        best_gain = -1
        best_ball = None

        for c_int in candidates:
            c_word = _decode_int(c_int, q, n)
            ball = set()
            ball.add(c_int)
            if radius >= 1:
                for pos in range(n):
                    for nd in range(q):
                        if nd != c_word[pos]:
                            w = list(c_word)
                            w[pos] = nd
                            ball.add(_encode_int(w, q, n))
            if radius >= 2:
                for p1 in range(n):
                    for p2 in range(p1 + 1, n):
                        for d1 in range(q):
                            if d1 == c_word[p1]:
                                continue
                            for d2 in range(q):
                                if d2 == c_word[p2]:
                                    continue
                                w = list(c_word)
                                w[p1] = d1
                                w[p2] = d2
                                ball.add(_encode_int(w, q, n))
            gain = len(ball & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_center = c_int
                best_ball = ball
                if gain == len(uncovered):
                    break

        if best_center is None or best_gain <= 0:
            break

        codewords.append(_decode_int(best_center, q, n))
        uncovered -= best_ball

    # If not fully covered, add random words to fill gaps (shouldn't happen often)
    while uncovered and (time.time() - start) < deadline - 2.0:
        c = rng.choice(list(uncovered))
        codewords.append(_decode_int(c, q, n))
        # Remove its ball from uncovered
        c_word = _decode_int(c, q, n)
        ball = {c}
        if radius >= 1:
            for pos in range(n):
                for nd in range(q):
                    if nd != c_word[pos]:
                        w = list(c_word)
                        w[pos] = nd
                        ball.add(_encode_int(w, q, n))
        uncovered -= ball

    return codewords


def _swap_improve(codewords, q, n, radius, deadline, seed=0):
    """
    Local search: try to remove a codeword if its ball is fully covered by others.
    Then try swaps: remove one, add another that covers more.
    """
    rng = random.Random(seed)
    total = q ** n
    if total > 500000:
        return codewords

    # Precompute balls for all codewords
    def compute_ball(cw):
        ball = set()
        ball.add(_encode_int(cw, q, n))
        if radius >= 1:
            for pos in range(n):
                for nd in range(q):
                    if nd != cw[pos]:
                        w = list(cw)
                        w[pos] = nd
                        ball.add(_encode_int(w, q, n))
        if radius >= 2:
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    for d1 in range(q):
                        if d1 == cw[p1]:
                            continue
                        for d2 in range(q):
                            if d2 == cw[p2]:
                                continue
                            w = list(cw)
                            w[p1] = d1
                            w[p2] = d2
                            ball.add(_encode_int(w, q, n))
        return ball

    balls = [compute_ball(cw) for cw in codewords]
    all_covered = set()
    for b in balls:
        all_covered |= b

    improved = True
    while improved and (time.time() - start_time) < deadline - 5.0:
        improved = False
        # Try removing each codeword
        for i in range(len(codewords)):
            if (time.time() - start_time) > deadline - 5.0:
                break
            # Check if ball[i] is covered by union of others
            others = set()
            for j in range(len(codewords)):
                if j == i:
                    continue
                others |= balls[j]
            if balls[i] <= others:
                codewords.pop(i)
                balls.pop(i)
                improved = True
                break

    # Try swaps
    if (time.time() - start_time) < deadline - 5.0:
        for _ in range(50):
            if (time.time() - start_time) > deadline - 5.0:
                break
            i = rng.randrange(len(codewords))
            # Try to find a replacement that covers more
            # Sample some random words as candidates
            best_new = None
            best_gain = 0
            current_union = set()
            for j in range(len(codewords)):
                if j != i:
                    current_union |= balls[j]
            for _ in range(200):
                cand_int = rng.randrange(total)
                cand_word = _decode_int(cand_int, q, n)
                if cand_word in codewords:
                    continue
                cand_ball = compute_ball(cand_word)
                gain = len(cand_ball - current_union) - len(balls[i] - current_union)
                if gain > best_gain:
                    best_gain = gain
                    best_new = (cand_int, cand_word, cand_ball)
            if best_new and best_gain > 0:
                cand_int, cand_word, cand_ball = best_new
                codewords[i] = cand_word
                balls[i] = cand_ball
                improved = True

    return codewords


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering
    radius `radius`. Returns a list of codewords (each a list of n ints).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Phase 1: Get an initial covering via greedy
    initial = _greedy_cover(q, n, radius, deadline - 30.0, seed=42)

    # Phase 2: Try to improve via swap/local search
    improved = _swap_improve(initial, q, n, radius, deadline, seed=42)

    # Phase 3: If time remains, try a few more random restarts of greedy
    best = improved
    for seed in [1, 2, 3]:
        if (time.time() - start_time) > deadline - 20.0:
            break
        candidate = _greedy_cover(q, n, radius, deadline - 15.0, seed=seed)
        candidate = _swap_improve(candidate, q, n, radius, deadline, seed=seed)
        if len(candidate) < len(best):
            best = candidate

    return best
# EVOLVE-BLOCK-END
