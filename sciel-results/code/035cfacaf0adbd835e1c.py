"""Covering code construction for K_q(n, radius) with a time-bounded greedy heuristic."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s wall-clock limit

    # Trivial cases
    if n == 0:
        return [[]]
    if radius == 0:
        return [[0] * n]

    # For small spaces, exact greedy (which is optimal for covering)
    total_words = q ** n
    if total_words <= 500000:
        # Enumerate all words, greedy set cover
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        # Precompute distance-1 neighbors for each word (ball of radius 1)
        # Actually, for greedy we want to pick the word that covers the most uncovered
        # But that's expensive. Instead, use a simpler greedy: pick word, mark all words
        # within radius as covered, repeat.
        
        covered = [False] * total_words
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # For each word, precompute its ball (all words within Hamming distance <= radius)
        # This is expensive for large radius, but for small total_words it's fine
        balls = {}
        for idx, w in enumerate(all_words):
            ball = set()
            ball.add(idx)
            if radius >= 1:
                for pos in range(n):
                    for val in range(q):
                        if val == w[pos]:
                            continue
                        new_word = list(w)
                        new_word[pos] = val
                        new_idx = word_to_idx[tuple(new_word)]
                        ball.add(new_idx)
            if radius >= 2 and n >= 2:
                for pos1 in range(n):
                    for pos2 in range(n):
                        if pos1 >= pos2:
                            continue
                        for val1 in range(q):
                            if val1 == w[pos1]:
                                continue
                            for val2 in range(q):
                                if val2 == w[pos2]:
                                    continue
                                new_word = list(w)
                                new_word[pos1] = val1
                                new_word[pos2] = val2
                                new_idx = word_to_idx[tuple(new_word)]
                                ball.add(new_idx)
            balls[idx] = ball
        
        code = []
        while any(covered) == False or not all(covered):
            # Check if all covered
            if all(covered):
                break
            # Find the word whose ball covers the most uncovered words
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                if all(covered[i] for i in balls[idx]):
                    continue
                count = sum(1 for i in balls[idx] if not covered[i])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            code.append(list(all_words[best_idx]))
            for i in balls[best_idx]:
                covered[i] = True
            if time.time() > deadline:
                break
        
        if all(covered) or len(code) > 0:
            # Verify coverage
            if not _verify_coverage(code, q, n, radius):
                # Fallback to direct sum
                return _direct_sum(q, n, radius)
            return code
        
        return _direct_sum(q, n, radius)

    # For larger spaces: use direct sum as base, then try to improve with greedy
    base_code = _direct_sum(q, n, radius)
    
    if time.time() > deadline - 5:
        return base_code

    # Try to improve: greedy removal
    # For each codeword, check if removing it still leaves full coverage
    # This is expensive, so we do it only if we have time
    
    code = list(base_code)
    if len(code) > 100 and time.time() < deadline - 10:
        code = _try_remove_redundant(code, q, n, radius, deadline)
    
    return code


def _direct_sum(q, n, radius):
    """Direct sum construction using a perfect code on a small block."""
    if q == 3:
        # Ternary tetracode: 9 words of length 4, perfect code radius 1
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return [list(h) for h in tetracode]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # n < 4, use all words
            return [list(w) for w in itertools.product(range(3), repeat=n)]
    elif q == 2:
        # Binary Hamming code or repetition
        if n >= 3:
            # Hamming code length 3: 4 words, perfect radius 1
            hamming = [[a, b, a ^ b] for a in range(2) for b in range(2)]
            if n == 3:
                return [list(h) for h in hamming]
            # For larger n, use product with all-words on remaining coordinates
            # But that gives 4 * 2^(n-3)
            # Better: use Hamming code on blocks of 3, and handle remainder
            full_blocks = n // 3
            remainder = n % 3
            code = []
            for block_combos in itertools.product(range(4), repeat=full_blocks):
                for tail in itertools.product(range(2), repeat=remainder):
                    word = []
                    for bi in block_combos:
                        a, b = divmod(bi, 2)
                        word.extend([a, b, a ^ b])
                    word.extend(tail)
                    code.append(word)
            return code
        else:
            return [list(w) for w in itertools.product(range(2), repeat=n)]
    else:
        # General case: use all words (trivial covering)
        return [list(w) for w in itertools.product(range(q), repeat=n)]


def _verify_coverage(code, q, n, radius):
    """Verify that every word is within Hamming distance radius of some codeword."""
    if not code:
        return n == 0
    # For large spaces, use a randomized check
    total_words = q ** n
    if total_words <= 1000000:
        # Exact check
        code_set = set(tuple(c) for c in code)
        for word in itertools.product(range(q), repeat=n):
            min_dist = min(_hamming(word, c) for c in code_set)
            if min_dist > radius:
                return False
        return True
    else:
        # Randomized check
        for _ in range(10000):
            word = tuple(random.randint(0, q - 1) for _ in range(n))
            min_dist = min(_hamming(word, c) for c in code)
            if min_dist > radius:
                return False
        return True


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _try_remove_redundant(code, q, n, radius, deadline):
    """Try to remove codewords that are not needed for coverage."""
    code = list(code)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        # Try removing each codeword in random order
        indices = list(range(len(code)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline - 5:
                break
            candidate = code[:idx] + code[idx+1:]
            if len(candidate) == 0:
                continue
            if _verify_coverage(candidate, q, n, radius):
                code = candidate
                changed = True
                break
    return code
# EVOLVE-BLOCK-END
