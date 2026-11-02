"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _ball_indices(q, n, radius):
    """Precompute the ball of radius `radius` around each word in {0..q-1}^n."""
    total = q ** n
    balls = [set() for _ in range(total)]
    
    # Generate all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # For each word, compute its ball
    for i, word in enumerate(all_words):
        balls[i].add(i)
        # Add words at distance 1
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_word = word[:]
                    new_word[pos] = new_val
                    # Convert to index
                    idx = 0
                    for j in range(n):
                        idx = idx * q + new_word[j]
                    balls[i].add(idx)
        # Add words at distance 2
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1 = word[pos1]
                    orig2 = word[pos2]
                    for new_val1 in range(q):
                        for new_val2 in range(q):
                            if new_val1 == orig1 and new_val2 == orig2:
                                continue
                            new_word = word[:]
                            new_word[pos1] = new_val1
                            new_word[pos2] = new_val2
                            idx = 0
                            for j in range(n):
                                idx = idx * q + new_word[j]
                            balls[i].add(idx)
    
    return balls


def _word_to_idx(word, q):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with the direct-sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to indices for faster computation
    code_indices = [_word_to_idx(c, q) for c in code]
    code_set = set(code_indices)
    
    # Precompute balls
    balls = _ball_indices(q, n, radius)
    
    # Greedy removal: try to remove codewords that are redundant
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code_indices) - 1, -1, -1):
            if time.time() >= deadline:
                break
            # Check if removing codeword i still covers everything
            candidate_set = code_set - {code_indices[i]}
            
            # Check coverage
            covered = set()
            for idx in candidate_set:
                covered.update(balls[idx])
            
            if len(covered) == q ** n:
                # Safe to remove
                code_set.discard(code_indices[i])
                code_indices.pop(i)
                code.pop(i)
                improved = True
    
    # Convert back to list of words
    return [_idx_to_word(idx, q, n) for idx in code_indices]
# EVOLVE-BLOCK-END
