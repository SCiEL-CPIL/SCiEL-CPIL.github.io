"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
from collections import Counter


def _encode(word, q, n):
    """Encode a word as an integer in base q."""
    val = 0
    for i, digit in enumerate(word):
        val = val * q + digit
    return val


def _decode(val, q, n):
    """Decode an integer back to a word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = val % q
        val //= q
    return word


def _hamming_dist(w1, w2):
    """Hamming distance between two words."""
    return sum(1 for a, b in zip(w1, w2) if a != b)


def _generate_balls(q, n, radius):
    """Precompute the ball (set of words within radius) for each word."""
    total = q ** n
    balls = []
    for idx in range(total):
        center = _decode(idx, q, n)
        ball = set()
        for word in itertools.product(range(q), repeat=n):
            if _hamming_dist(center, word) <= radius:
                ball.add(_encode(word, q, n))
        balls.append(ball)
    return balls


def _greedy_cover(q, n, radius, balls, random_seed=0):
    """Greedy set cover to build a covering code."""
    total = q ** n
    uncovered = set(range(total))
    codewords = []
    
    # Precompute coverage count for each candidate
    # For each word, count how many uncovered words it covers
    while uncovered:
        best_word = -1
        best_count = -1
        for word in range(total):
            if word in codewords:
                continue
            count = len(balls[word] & uncovered)
            if count > best_count:
                best_count = count
                best_word = word
            if best_count == len(uncovered):
                break
        codewords.append(best_word)
        uncovered -= balls[best_word]
    
    return codewords


def _local_search_remove(codewords, q, n, radius, balls):
    """Try to remove codewords that are redundant."""
    total = q ** n
    covered = set()
    for cw in codewords:
        covered |= balls[cw]
    
    changed = True
    while changed:
        changed = False
        for i in range(len(codewords)):
            # Check if removing codeword[i] still covers everything
            remaining = set(codewords[:i] + codewords[i+1:])
            new_covered = set()
            for cw in remaining:
                new_covered |= balls[cw]
            if new_covered == set(range(total)):
                codewords.pop(i)
                covered = new_covered
                changed = True
                break
    return codewords


def construct(q, n, radius):
    total = q ** n
    
    # For small instances, use exact/greedy + local search
    if total <= 10000:
        balls = _generate_balls(q, n, radius)
        
        # Greedy construction
        codewords = _greedy_cover(q, n, radius, balls)
        
        # Local search to remove redundant codewords
        codewords = _local_search_remove(codewords, q, n, radius, balls)
        
        # Convert back to list form
        return [_decode(cw, q, n) for cw in codewords]
    else:
        # Fallback: direct sum with tetracode for larger n
        if n < 4:
            # For very small n, brute force
            return [list(word) for word in itertools.product(range(q), repeat=n)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
# EVOLVE-BLOCK-END
