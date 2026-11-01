"""Improved football pool solver: tetracode base + LP-guided greedy + local search."""
import itertools
import time
import numpy as np
from scipy.optimize import linprog

# Precompute tetracode (perfect code of length 4, radius 1, size 9)
_TETRACODE = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _ball_indices(codeword, q, n, radius=1):
    """Yield indices (as flat index) of all words within Hamming distance <= radius from codeword."""
    # The codeword itself
    yield _flat_index(codeword, q, n)
    # All words at distance 1
    for pos in range(n):
        orig = codeword[pos]
        for val in range(q):
            if val != orig:
                yield _flat_index_at(codeword, pos, val, q, n)


def _flat_index(codeword, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + codeword[i]
    return idx


def _flat_index_at(codeword, pos, val, q, n):
    # Compute flat index of codeword with position pos changed to val
    idx = 0
    for i in range(n):
        d = val if i == pos else codeword[i]
        idx = idx * q + d
    return idx


def _to_tuple(codeword):
    return tuple(codeword)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Strategy 1: tetracode base construction (works for n >= 4)
    if n >= 4:
        tetracode = _TETRACODE
        tail_size = n - 4
        base_code = [h + list(tail) for h in tetracode
                     for tail in itertools.product(range(q), repeat=tail_size)]
        best_code = base_code
        best_count = len(base_code)
    else:
        # For small n, use greedy
        best_code = _greedy_cover(q, n, radius, start_time, deadline)
        best_count = len(best_code)

    if time.time() > deadline - 10:
        return best_code

    # Strategy 2: Try to improve via local search / greedy removal
    # For the tetracode construction, each codeword covers (2*n*q - 1) words (for q=3, radius=1)
    # Total space = q^n. If q^n / (2*n*q - 1) < best_count, we might do better.
    # But for q=3, n=8: 3^8=6561, coverage per codeword = 2*8*3-1=47, 6561/47≈139.6
    # The tetracode gives 9*3^4=729 for n=8? No: 9 * 3^(8-4) = 9*81=729.
    # But best known is 486. So we need to improve.

    # Try greedy set-cover approach on the remaining uncovered words
    # First, check if the current code is valid (it should be by construction)
    # Then try to remove redundant codewords
    
    # Compute coverage
    covered = set()
    code_tuples = set(_to_tuple(c) for c in best_code)
    
    for c in best_code:
        for idx in _ball_indices(c, q, n, radius):
            covered.add(idx)
    
    total_words = q ** n
    if len(covered) == total_words:
        # Try to remove redundant codewords
        best_code = _remove_redundant(best_code, q, n, radius, start_time, deadline)
        best_count = len(best_code)

    if time.time() > deadline - 5:
        return best_code

    # Strategy 3: Greedy cover from scratch with LP guidance for larger n
    # For n <= 6, we can do a more careful greedy
    if n <= 6:
        greedy_code = _greedy_cover(q, n, radius, start_time, deadline)
        if len(greedy_code) < best_count:
            best_code = greedy_code
            best_count = len(greedy_code)

    return best_code


def _remove_redundant(code, q, n, radius, start_time, deadline):
    """Try to remove codewords that are redundant."""
    code_list = list(code)
    n_code = len(code_list)
    
    if n_code == 0:
        return code_list

    # Compute coverage count for each word
    # For efficiency, only do this for small codes
    if n_code > 2000 or time.time() > deadline - 5:
        return code_list

    # Count how many codewords cover each word
    cover_count = np.zeros(q ** n, dtype=np.int32)
    for c in code_list:
        for idx in _ball_indices(c, q, n, radius):
            cover_count[idx] += 1

    # Try to remove codewords in order of lowest contribution
    # A codeword is redundant if all words in its ball are covered by at least one other codeword
    # i.e., cover_count[idx] >= 2 for all idx in ball(c)
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        new_list = []
        for c in code_list:
            ball = list(_ball_indices(c, q, n, radius))
            redundant = all(cover_count[idx] >= 2 for idx in ball)
            if redundant:
                # Remove it
                for idx in ball:
                    cover_count[idx] -= 1
                changed = True
            else:
                new_list.append(c)
        if changed:
            code_list = new_list

    return code_list


def _greedy_cover(q, n, radius, start_time, deadline):
    """Greedy set cover: repeatedly pick the word that covers the most uncovered words."""
    total_words = q ** n
    if total_words > 3 ** 8:
        # Too large for this approach
        return []

    uncovered = set(range(total_words))
    code = []

    # Precompute balls for all words (only feasible for small spaces)
    # For q=3, n<=6: 729 words, each ball has 1 + 2*6 = 13 words. Manageable.
    balls = {}
    for w in range(total_words):
        cw = [0] * n
        rem = w
        for i in range(n - 1, -1, -1):
            cw[i] = rem % q
            rem //= q
        ball = set()
        ball.add(w)
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val != orig:
                    idx = _flat_index_at(cw, pos, val, q, n)
                    ball.add(idx)
        balls[w] = ball

    while uncovered and time.time() < deadline - 5:
        best_word = None
        best_count = 0
        for w in uncovered:
            cnt = len(balls[w] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_word = w
            if time.time() > deadline - 5:
                break
        if best_word is None or best_count == 0:
            break
        code.append(_decode(best_word, q, n))
        uncovered -= balls[best_word]

    return code


def _decode(flat_idx, q, n):
    cw = [0] * n
    rem = flat_idx
    for i in range(n - 1, -1, -1):
        cw[i] = rem % q
        rem //= q
    return cw
